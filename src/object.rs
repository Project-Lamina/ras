//! Object file generation: ELF (Linux/BSD), Mach-O (macOS), COFF/PE (Windows).

use crate::parser::ArmAttributeValue;
#[cfg(test)]
use crate::parser::SectionFlags;
use crate::{error::RasError, parser::Section};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::collections::HashMap;
use std::io::Write;

/// Extra sections and metadata for [`ObjectWriter`] backends.
#[derive(Debug, Clone)]
pub struct ObjectWriteOptions {
    /// Emit `.debug_line`, `.debug_abbrev` and `.debug_info` even without `.loc` rows.
    /// Rows from `.file`/`.loc` always produce DWARF on ELF targets.
    pub emit_minimal_dwarf_sections: bool,
    /// Retain temporary local symbols in ELF output.
    pub keep_locals: bool,
    /// Compile unit name used when the source declares no `.file` entry.
    pub dwarf_decl_file_name: String,
    /// Name CodeView's `S_OBJNAME` gives this object.
    pub debug_object_name: String,
    /// Symbols that define a function. Each is emitted as `STT_FUNC` with the byte
    /// length of its code, independently of whether debug info is requested.
    pub function_symbols: Vec<String>,
    /// Subprograms and their types for `.debug_info`. A name with no definition in
    /// this object is skipped.
    pub debug_info: Option<crate::dwarf::DebugInfoRequest>,
}

impl Default for ObjectWriteOptions {
    fn default() -> Self {
        Self {
            emit_minimal_dwarf_sections: false,
            keep_locals: false,
            dwarf_decl_file_name: "lamina.s".to_string(),
            debug_object_name: "lamina.obj".to_string(),
            function_symbols: Vec::new(),
            debug_info: None,
        }
    }
}

/// A symbol with its resolved offset within its defining section, ready for the
/// object file symbol table. Built by the assembler after two-pass resolution.
#[derive(Debug, Clone)]
pub struct ObjectSymbol {
    pub attributes: crate::parser::SymbolAttributes,
    pub name: String,
    pub global: bool,
    pub section: String,
    pub value: u64,
}

/// A relocation entry for an unresolved (external) symbol reference.
/// `offset` is the byte offset within the code section of the 4-byte displacement field.
#[derive(Debug, Clone)]
pub struct ExternalReloc {
    pub offset: usize,
    pub symbol: String,
    pub section: String,
    pub kind: u32,
    pub addend: i64,
    /// x86_64: bytes that follow the 4-byte displacement in the instruction (an immediate).
    pub trailing: usize,
    /// x86_64: the reference is a call or jump target, encoded as a Mach-O branch.
    pub branch: bool,
    /// `symbol` names a section, and the relocation targets that section's symbol
    /// with `addend` measured from the section start.
    pub section_symbol: bool,
}

/// All data needed to write a single relocatable object file.
pub struct ObjectWriteRequest<'a> {
    pub code: &'a [u8],
    pub sections: &'a [Section],
    pub symbols: &'a [ObjectSymbol],
    pub relocations: &'a [ExternalReloc],
    pub target_arch: TargetArchitecture,
    pub target_os: TargetOperatingSystem,
    pub opts: &'a ObjectWriteOptions,
}

/// Backend for writing relocatable object files (ELF, Mach-O, or COFF/PE).
pub trait ObjectWriter {
    /// Build a relocatable object file in memory.
    fn object_bytes(&mut self, req: &ObjectWriteRequest<'_>) -> Result<Vec<u8>, RasError>;

    /// Write a relocatable object file to `path`.
    fn write_object_file(
        &mut self,
        path: &std::path::Path,
        req: &ObjectWriteRequest<'_>,
    ) -> Result<(), RasError> {
        let bytes = self.object_bytes(req)?;
        std::fs::write(path, bytes).map_err(|e| RasError::ObjectError(e.to_string()))
    }
}

const ELF64_HEADER_SIZE: usize = 64;
const ELF64_SECTION_HEADER_SIZE: usize = 64;
const ELF32_HEADER_SIZE: usize = 52;
const ELF32_SECTION_HEADER_SIZE: usize = 40;
const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ET_REL: u16 = 1;
const EM_X86_64: u16 = 62;
const EM_386: u16 = 3;
const EM_ARM: u16 = 40;
const EM_AARCH64: u16 = 183;
const EM_RISCV: u16 = 243;

/// The float ABI an object was built for. A module that leaves this at soft-float
/// cannot be merged with a double-float one, which is what glibc ships.
const EF_RISCV_FLOAT_ABI_DOUBLE: u32 = 0x4;
const EM_PPC64: u16 = 21;
#[cfg(feature = "nightly")]
const EM_ARX64: u16 = 0xa064;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_REL: u32 = 9;
const SHT_ARM_ATTRIBUTES: u32 = 0x7000_0003;
const SHF_INFO_LINK: u64 = 0x40;
const SHF_TLS: u64 = 0x400;
const ELF64_SYM_SIZE: u64 = 24;
const ELF64_RELA_SIZE: u64 = 24;
const ELF32_SYM_SIZE: u64 = 16;
const ELF32_RELA_SIZE: u64 = 12;
const ELF32_REL_SIZE: u64 = 8;
const STB_GLOBAL: u8 = 1;
const STT_TLS: u8 = 6;

const R_386_32: u32 = 1;
const R_386_PC32: u32 = 2;
const R_386_PLT32: u32 = 4;
const R_386_16: u32 = 20;
/// Intel i386 psABI, relocation types: 22. 23 is `R_386_PC8`.
const R_386_8: u32 = 22;

const R_ARM_ABS32: u32 = 2;
const R_ARM_REL32: u32 = 3;
const R_ARM_CALL: u32 = 28;
const R_ARM_JUMP24: u32 = 29;
const R_ARM_MOVW_ABS_NC: u32 = 43;
const R_ARM_MOVT_ABS: u32 = 44;

/// The width of the field an i386 relocation patches. The addend is stored
/// there, so a kind whose field width is unknown cannot be written at all.
fn i386_rel_width(kind: u32) -> Result<usize, RasError> {
    match kind {
        R_386_32 | R_386_PC32 | R_386_PLT32 => Ok(4),
        R_386_16 => Ok(2),
        R_386_8 => Ok(1),
        _ => Err(RasError::ObjectError(format!(
            "i386 relocation type {kind} is not implemented"
        ))),
    }
}

/// Where an ARM relocation's addend lives. ARM uses `Elf32_Rel`, so the addend
/// is always in the instruction or data the relocation patches: for a plain
/// data word that is a field of a known width, and for an instruction it is a
/// bit field the assembler has already filled in.
enum ArmAddend {
    Field(usize),
    InPlace,
}

fn arm_rel_addend(kind: u32) -> Result<ArmAddend, RasError> {
    match kind {
        R_ARM_ABS32 | R_ARM_REL32 => Ok(ArmAddend::Field(4)),
        R_ARM_CALL | R_ARM_JUMP24 | R_ARM_MOVW_ABS_NC | R_ARM_MOVT_ABS => Ok(ArmAddend::InPlace),
        _ => Err(RasError::ObjectError(format!(
            "ARM relocation type {kind} is not implemented"
        ))),
    }
}

pub const ARM_ATTRIBUTES_SECTION: &str = ".ARM.attributes";

/// Contents of an `.ARM.attributes` section: format version `A`, then one
/// `aeabi` vendor subsection holding a single file-scope attribute list.
pub fn arm_attributes(attributes: &[(u32, ArmAttributeValue)]) -> Vec<u8> {
    let ordered: std::collections::BTreeMap<u32, &ArmAttributeValue> = attributes
        .iter()
        .map(|(tag, value)| (*tag, value))
        .collect();
    let mut body = Vec::new();
    for (tag, value) in ordered {
        body.extend(crate::parser::leb128(i128::from(tag), false));
        match value {
            ArmAttributeValue::Number(number) => {
                body.extend(crate::parser::leb128(i128::from(*number), false));
            }
            ArmAttributeValue::Text(text) => {
                body.extend_from_slice(text.as_bytes());
                body.push(0);
            }
        }
    }
    let mut vendor = b"aeabi\0".to_vec();
    vendor.push(1);
    vendor.extend(((body.len() + 5) as u32).to_le_bytes());
    vendor.extend(body);
    let mut out = vec![b'A'];
    out.extend(((vendor.len() + 4) as u32).to_le_bytes());
    out.extend(vendor);
    out
}

fn elf64_e_machine(arch: TargetArchitecture) -> Option<u16> {
    match arch {
        TargetArchitecture::X86_64 => Some(EM_X86_64),
        TargetArchitecture::I686 => Some(EM_386),
        TargetArchitecture::Armv7 => Some(EM_ARM),
        TargetArchitecture::Aarch64 => Some(EM_AARCH64),
        #[cfg(feature = "nightly")]
        TargetArchitecture::Arx64 => Some(EM_ARX64),
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => Some(EM_RISCV),
        TargetArchitecture::PowerPC64 => Some(EM_PPC64),
        _ => None,
    }
}

fn write_elf_header_with_shnum<W: Write>(
    w: &mut W,
    class32: bool,
    e_machine: u16,
    e_shnum: u16,
    e_shstrndx: u16,
) -> Result<(), RasError> {
    let map = |e: std::io::Error| RasError::ObjectError(format!("ELF write: {}", e));
    let mut e_ident = [0u8; 16];
    e_ident[0..4].copy_from_slice(&ELF_MAGIC);
    e_ident[4] = if class32 { 1 } else { 2 };
    e_ident[5] = 1;
    e_ident[6] = 1;
    w.write_all(&e_ident).map_err(map)?;
    w.write_all(&ET_REL.to_le_bytes()).map_err(map)?;
    w.write_all(&e_machine.to_le_bytes()).map_err(map)?;
    w.write_all(&1u32.to_le_bytes()).map_err(map)?;
    let (header_size, shdr_size) = if class32 {
        (ELF32_HEADER_SIZE, ELF32_SECTION_HEADER_SIZE)
    } else {
        (ELF64_HEADER_SIZE, ELF64_SECTION_HEADER_SIZE)
    };
    if class32 {
        for value in [0u32, 0, header_size as u32] {
            w.write_all(&value.to_le_bytes()).map_err(map)?;
        }
    } else {
        for value in [0u64, 0, header_size as u64] {
            w.write_all(&value.to_le_bytes()).map_err(map)?;
        }
    }
    // EF_ARM_EABI_VER5; the float ABI is described by .ARM.attributes, and the
    // linker sets EF_ARM_ABI_FLOAT_* on the output from those.
    let e_flags: u32 = match e_machine {
        EM_PPC64 => 2,
        EM_ARM => 0x0500_0000,
        EM_RISCV => EF_RISCV_FLOAT_ABI_DOUBLE,
        _ => 0,
    };
    w.write_all(&e_flags.to_le_bytes()).map_err(map)?;
    w.write_all(&(header_size as u16).to_le_bytes())
        .map_err(map)?;
    w.write_all(&0u16.to_le_bytes()).map_err(map)?;
    w.write_all(&0u16.to_le_bytes()).map_err(map)?;
    w.write_all(&(shdr_size as u16).to_le_bytes())
        .map_err(map)?;
    w.write_all(&e_shnum.to_le_bytes()).map_err(map)?;
    w.write_all(&e_shstrndx.to_le_bytes()).map_err(map)?;
    Ok(())
}

/// Full ELF64 section header fields. `sh_addr` is always zero for relocatable
/// objects, so it is not exposed. Unset fields default to zero.
#[derive(Default)]
struct Elf64SectionHeader {
    name: u32,
    section_type: u32,
    flags: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
}

fn write_elf_section_header_full<W: Write>(
    w: &mut W,
    class32: bool,
    header: &Elf64SectionHeader,
) -> Result<(), RasError> {
    let map = |e: std::io::Error| RasError::ObjectError(format!("ELF write: {}", e));
    w.write_all(&header.name.to_le_bytes()).map_err(map)?;
    w.write_all(&header.section_type.to_le_bytes())
        .map_err(map)?;
    let wide = [header.flags, 0, header.offset, header.size];
    if class32 {
        for value in wide {
            w.write_all(&(value as u32).to_le_bytes()).map_err(map)?;
        }
    } else {
        for value in wide {
            w.write_all(&value.to_le_bytes()).map_err(map)?;
        }
    }
    w.write_all(&header.link.to_le_bytes()).map_err(map)?;
    w.write_all(&header.info.to_le_bytes()).map_err(map)?;
    if class32 {
        w.write_all(&(header.addralign as u32).to_le_bytes())
            .map_err(map)?;
        w.write_all(&(header.entsize as u32).to_le_bytes())
            .map_err(map)?;
    } else {
        w.write_all(&header.addralign.to_le_bytes()).map_err(map)?;
        w.write_all(&header.entsize.to_le_bytes()).map_err(map)?;
    }
    Ok(())
}

/// ELFv2 st_other bits 5-7 hold log2 of the local entry offset; 0 means none.
fn elf_local_entry(
    attributes: &crate::parser::SymbolAttributes,
    arch: TargetArchitecture,
) -> Result<u8, RasError> {
    let Some(value) = attributes.local_entry.as_deref() else {
        return Ok(0);
    };
    if arch != TargetArchitecture::PowerPC64 {
        return Err(RasError::ObjectError(
            ".localentry is only defined for ppc64le".into(),
        ));
    }
    let offset: u64 = value
        .parse()
        .map_err(|_| RasError::ObjectError(format!("Invalid .localentry offset: {value}")))?;
    match offset {
        0 => Ok(0),
        4 | 8 | 16 | 32 | 64 | 128 => Ok((offset.trailing_zeros() as u8) << 5),
        _ => Err(RasError::ObjectError(format!(
            "Unrepresentable .localentry offset {offset}; ELFv2 allows 0 or a power of two from 4 to 128"
        ))),
    }
}

fn elf_visibility(attributes: &crate::parser::SymbolAttributes) -> u8 {
    if attributes.internal_visibility {
        1
    } else if attributes.hidden {
        2
    } else if attributes.protected {
        3
    } else {
        0
    }
}

fn write_elf_symbol<W: Write>(
    w: &mut W,
    class32: bool,
    name_offset: u32,
    info: u8,
    visibility: u8,
    section_index: u16,
    value: u64,
) -> Result<(), RasError> {
    let map = |e: std::io::Error| RasError::ObjectError(format!("ELF write: {}", e));
    w.write_all(&name_offset.to_le_bytes()).map_err(map)?;
    if class32 {
        w.write_all(&(value as u32).to_le_bytes()).map_err(map)?;
        w.write_all(&0u32.to_le_bytes()).map_err(map)?;
        w.write_all(&[info, visibility]).map_err(map)?;
        w.write_all(&section_index.to_le_bytes()).map_err(map)?;
    } else {
        w.write_all(&[info, visibility]).map_err(map)?;
        w.write_all(&section_index.to_le_bytes()).map_err(map)?;
        w.write_all(&value.to_le_bytes()).map_err(map)?;
        w.write_all(&0u64.to_le_bytes()).map_err(map)?;
    }
    Ok(())
}

pub struct ElfWriter;

impl Default for ElfWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ElfWriter {
    pub fn new() -> Self {
        Self
    }
}

/// Only COFF has an image base to be relative to.
fn refuse_image_relative(req: &ObjectWriteRequest<'_>, format: &str) -> Result<(), RasError> {
    match req.relocations.iter().find(|r| r.kind == COFF_IMGREL32) {
        Some(r) => Err(RasError::ObjectError(format!(
            "image-relative address of {} is only supported in COFF objects, not {format}",
            r.symbol
        ))),
        None => Ok(()),
    }
}

impl ObjectWriter for ElfWriter {
    fn object_bytes(&mut self, req: &ObjectWriteRequest<'_>) -> Result<Vec<u8>, RasError> {
        refuse_image_relative(req, "ELF")?;
        if req.target_arch == TargetArchitecture::X86_64
            && let Some(r) = req.relocations.iter().find(|r| r.kind == X86_64_SECREL32)
        {
            return Err(RasError::ObjectError(format!(
                "x86_64 @SECREL32 for {} is only supported in COFF objects",
                r.symbol
            )));
        }
        if req.target_arch == TargetArchitecture::Aarch64
            && let Some(r) = req.relocations.iter().find(|r| matches!(r.kind, 313 | 314))
        {
            return Err(RasError::ObjectError(format!(
                "arm64 @TLVP relocation for {} is only supported in Mach-O objects",
                r.symbol
            )));
        }
        if let Some(s) = req.sections.iter().find(|s| s.macho.is_some()) {
            return Err(RasError::ObjectError(format!(
                "Mach-O section {} cannot be written to an ELF object",
                s.name
            )));
        }
        let machine = elf64_e_machine(req.target_arch)
            .ok_or_else(|| RasError::ObjectError("Unsupported ELF architecture".into()))?;
        let class32 = matches!(
            req.target_arch,
            TargetArchitecture::Riscv32 | TargetArchitecture::I686 | TargetArchitecture::Armv7
        );
        // The i386 and ARM psABIs define Elf32_Rel only: the addend lives in
        // the field the relocation patches, not in the relocation entry.
        let use_rel = matches!(
            req.target_arch,
            TargetArchitecture::I686 | TargetArchitecture::Armv7
        );
        let sym_size = if class32 {
            ELF32_SYM_SIZE
        } else {
            ELF64_SYM_SIZE
        };
        let rela_size = if use_rel {
            ELF32_REL_SIZE
        } else if class32 {
            ELF32_RELA_SIZE
        } else {
            ELF64_RELA_SIZE
        };
        let table_align = if class32 { 4 } else { 8 };
        let mut sections = req.sections.to_vec();
        if sections.is_empty() {
            sections.push(Section {
                group: None,
                name: ".text".into(),
                flags: crate::parser::SectionFlags {
                    alloc: true,
                    exec: true,
                    write: false,
                    tls: false,
                },
                data: req.code.to_vec(),
                alignment: 1,
                macho: None,
            });
        }
        if matches!(
            req.target_arch,
            TargetArchitecture::X86_64 | TargetArchitecture::I686 | TargetArchitecture::Armv7
        ) {
            for name in [".data", ".bss"] {
                if !sections.iter().any(|s| s.name == name) {
                    sections.push(Section {
                        group: None,
                        name: name.into(),
                        flags: crate::parser::SectionFlags {
                            alloc: true,
                            exec: false,
                            write: true,
                            tls: false,
                        },
                        data: Vec::new(),
                        alignment: 1,
                        macho: None,
                    });
                }
            }
        }
        let mut names: Vec<String> = sections.iter().map(|s| s.name.clone()).collect();
        let mut data: Vec<Vec<u8>> = sections.iter().map(|s| s.data.clone()).collect();
        let mut headers: Vec<Elf64SectionHeader> = sections
            .iter()
            .map(|s| Elf64SectionHeader {
                section_type: if s.name == ARM_ATTRIBUTES_SECTION {
                    SHT_ARM_ATTRIBUTES
                } else if matches!(s.name.as_str(), ".bss" | ".tbss") {
                    8
                } else if s.name.starts_with(".note.") && s.name != ".note.GNU-stack" {
                    7
                } else {
                    SHT_PROGBITS
                },
                flags: u64::from(s.flags.write)
                    | (u64::from(s.flags.alloc) << 1)
                    | (u64::from(s.flags.exec) << 2)
                    | if s.group.is_some() { 0x200 } else { 0 }
                    | if s.flags.tls { SHF_TLS } else { 0 },
                addralign: s.alignment as u64,
                ..Default::default()
            })
            .collect();
        let mut strtab = vec![0];
        let mut symtab = vec![0; sym_size as usize];
        let mut sym_indices = std::collections::HashMap::new();
        let mut object_symbols = req.symbols.to_vec();
        for section in &sections {
            if let Some(signature) = &section.group
                && !object_symbols.iter().any(|s| &s.name == signature)
            {
                object_symbols.push(ObjectSymbol {
                    name: signature.clone(),
                    global: false,
                    section: section.name.clone(),
                    value: 0,
                    attributes: Default::default(),
                });
            }
        }
        let merge_relocs: std::collections::HashSet<_> = req
            .relocations
            .iter()
            .filter(|r| {
                !matches!(
                    (req.target_arch, r.kind),
                    (TargetArchitecture::X86_64, 1 | 10 | 11 | 12 | 14)
                        | (TargetArchitecture::I686, R_386_32 | R_386_16 | R_386_8)
                        | (TargetArchitecture::Armv7, R_ARM_ABS32)
                        | (TargetArchitecture::Aarch64, 257..=259)
                )
            })
            .map(|r| r.symbol.as_str())
            .collect();
        let identity_relocs: std::collections::HashSet<_> = req
            .relocations
            .iter()
            .filter(|r| {
                matches!(
                    (req.target_arch, r.kind),
                    (TargetArchitecture::X86_64, 9 | 41 | 42)
                        | (TargetArchitecture::Aarch64, 311 | 312)
                )
            })
            .map(|r| r.symbol.as_str())
            .collect();
        let omitted: std::collections::HashMap<_, _> = object_symbols
            .iter()
            .filter(|s| {
                s.attributes.internal
                    || (!req.opts.keep_locals
                        && s.name.starts_with(".L")
                        && !s.global
                        && !s.attributes.weak
                        && !s.section.is_empty()
                        && s.section != "*COM*"
                        && !sections
                            .iter()
                            .any(|section| section.group.as_ref() == Some(&s.name))
                        && !(s.attributes.retain && merge_relocs.contains(s.name.as_str()))
                        && !identity_relocs.contains(s.name.as_str()))
            })
            .map(|s| (s.name.as_str(), s))
            .collect();
        let mut section_symbols = std::collections::HashMap::new();
        for reloc in req.relocations.iter().filter(|r| r.section_symbol) {
            if section_symbols.contains_key(reloc.symbol.as_str()) {
                continue;
            }
            let section = names
                .iter()
                .position(|n| n == &reloc.symbol)
                .ok_or_else(|| {
                    RasError::ObjectError(format!("Unknown relocation section: {}", reloc.symbol))
                })?
                + 1;
            let index = symtab.len() / sym_size as usize;
            section_symbols.insert(reloc.symbol.as_str(), index as u32);
            write_elf_symbol(&mut symtab, class32, 0, 3, 0, section as u16, 0)?;
        }
        for reloc in req.relocations {
            let Some(symbol) = omitted.get(reloc.symbol.as_str()) else {
                continue;
            };
            if symbol.section == "*ABS*" {
                section_symbols.insert(symbol.section.as_str(), 0);
                continue;
            }
            if section_symbols.contains_key(symbol.section.as_str()) {
                continue;
            }
            let section = names
                .iter()
                .position(|n| n == &symbol.section)
                .ok_or_else(|| {
                    RasError::ObjectError(format!("Unknown symbol section: {}", symbol.section))
                })?
                + 1;
            let index = symtab.len() / sym_size as usize;
            section_symbols.insert(symbol.section.as_str(), index as u32);
            write_elf_symbol(&mut symtab, class32, 0, 3, 0, section as u16, 0)?;
        }
        let mut defined: Vec<_> = object_symbols
            .iter()
            .filter(|s| !omitted.contains_key(s.name.as_str()))
            .collect();
        defined.sort_by_key(|s| s.global || s.attributes.weak);
        let first_global = (symtab.len() / sym_size as usize) as u32
            + defined
                .iter()
                .filter(|s| !s.global && !s.attributes.weak)
                .count() as u32;
        let tls_sections: std::collections::HashSet<&str> = sections
            .iter()
            .filter(|s| s.flags.tls)
            .map(|s| s.name.as_str())
            .collect();
        for sym in defined {
            let index = symtab.len() / sym_size as usize;
            sym_indices.insert(sym.name.as_str(), index as u32);
            let name = strtab.len() as u32;
            strtab.extend_from_slice(sym.name.as_bytes());
            strtab.push(0);
            let section = if sym.section == "*ABS*" {
                0xfff1
            } else if sym.section == "*COM*" {
                0xfff2
            } else if sym.section.is_empty() {
                0
            } else {
                names
                    .iter()
                    .position(|n| n == &sym.section)
                    .ok_or_else(|| {
                        RasError::ObjectError(format!("Unknown symbol section: {}", sym.section))
                    })?
                    + 1
            };
            write_elf_symbol(
                &mut symtab,
                class32,
                name,
                (if sym.attributes.weak {
                    2
                } else {
                    u8::from(sym.global)
                } << 4)
                    | (if tls_sections.contains(sym.section.as_str()) {
                        STT_TLS
                    } else {
                        sym.attributes.symbol_type
                    } & 0xf),
                elf_visibility(&sym.attributes)
                    | elf_local_entry(&sym.attributes, req.target_arch)?,
                section as u16,
                sym.value,
            )?;
            if let Some(size) = sym
                .attributes
                .size
                .as_ref()
                .and_then(|s| s.parse::<u64>().ok())
                .or_else(|| sym.attributes.common.map(|c| c.0))
            {
                let end = symtab.len();
                if class32 {
                    let at = end - 8;
                    let size = u32::try_from(size).map_err(|_| {
                        RasError::ObjectError(format!("Symbol size {size} is not 32-bit"))
                    })?;
                    symtab[at..at + 4].copy_from_slice(&size.to_le_bytes());
                } else {
                    symtab[end - 8..].copy_from_slice(&size.to_le_bytes());
                }
            }
        }
        for reloc in req.relocations.iter().filter(|r| !r.section_symbol) {
            if !sym_indices.contains_key(reloc.symbol.as_str())
                && !omitted.contains_key(reloc.symbol.as_str())
            {
                sym_indices.insert(
                    reloc.symbol.as_str(),
                    (symtab.len() / sym_size as usize) as u32,
                );
                let name = strtab.len() as u32;
                strtab.extend_from_slice(reloc.symbol.as_bytes());
                strtab.push(0);
                write_elf_symbol(&mut symtab, class32, name, STB_GLOBAL << 4, 0, 0, 0)?;
            }
        }
        let mut groups: std::collections::BTreeMap<&str, Vec<u32>> =
            std::collections::BTreeMap::new();
        for (i, section) in sections.iter().enumerate() {
            if let Some(signature) = &section.group {
                groups.entry(signature).or_default().push((i + 1) as u32);
            }
        }
        let mut rela_sections = Vec::new();
        for (i, section) in sections.iter().enumerate() {
            let mut bytes = Vec::new();
            for r in req.relocations.iter().filter(|r| r.section == section.name) {
                if class32 {
                    bytes.extend_from_slice(&(r.offset as u32).to_le_bytes());
                } else {
                    bytes.extend_from_slice(&(r.offset as u64).to_le_bytes());
                }
                let (index, addend) = if r.section_symbol {
                    (section_symbols[r.symbol.as_str()], r.addend)
                } else if let Some(symbol) = omitted.get(r.symbol.as_str()) {
                    (
                        section_symbols[symbol.section.as_str()],
                        r.addend.wrapping_add(symbol.value as i64),
                    )
                } else {
                    (sym_indices[r.symbol.as_str()], r.addend)
                };
                if use_rel {
                    let info = (index << 8) | (r.kind & 0xff);
                    bytes.extend_from_slice(&info.to_le_bytes());
                    let width = if req.target_arch == TargetArchitecture::Armv7 {
                        match arm_rel_addend(r.kind)? {
                            ArmAddend::Field(width) => Some(width),
                            ArmAddend::InPlace if addend == 0 => None,
                            ArmAddend::InPlace => {
                                return Err(RasError::ObjectError(format!(
                                    "ARM relocation type {} for {} carries addend {addend}, \
                                     which belongs in the instruction field",
                                    r.kind, r.symbol
                                )));
                            }
                        }
                    } else {
                        Some(i386_rel_width(r.kind)?)
                    };
                    if let Some(width) = width {
                        let field =
                            data[i].get_mut(r.offset..r.offset + width).ok_or_else(|| {
                                RasError::ObjectError(format!(
                                    "Relocation for {} lies outside {}",
                                    r.symbol, section.name
                                ))
                            })?;
                        field.copy_from_slice(&addend.to_le_bytes()[..width]);
                    }
                } else if class32 {
                    let info = (index << 8) | (r.kind & 0xff);
                    bytes.extend_from_slice(&info.to_le_bytes());
                    let addend = i32::try_from(addend).map_err(|_| {
                        RasError::ObjectError(format!("Relocation addend {addend} is not 32-bit"))
                    })?;
                    bytes.extend_from_slice(&addend.to_le_bytes());
                } else {
                    let info = (u64::from(index) << 32) | u64::from(r.kind);
                    bytes.extend_from_slice(&info.to_le_bytes());
                    bytes.extend_from_slice(&addend.to_le_bytes());
                }
            }
            if !bytes.is_empty() {
                if let Some(signature) = &section.group {
                    groups
                        .get_mut(signature.as_str())
                        .unwrap()
                        .push((headers.len() + 1) as u32);
                }
                rela_sections.push(headers.len());
                names.push(format!(
                    "{}{}",
                    if use_rel { ".rel" } else { ".rela" },
                    section.name
                ));
                data.push(bytes);
                headers.push(Elf64SectionHeader {
                    section_type: if use_rel { SHT_REL } else { SHT_RELA },
                    flags: SHF_INFO_LINK | if section.group.is_some() { 0x200 } else { 0 },
                    info: (i + 1) as u32,
                    addralign: table_align,
                    entsize: rela_size,
                    ..Default::default()
                });
            }
        }
        let mut group_sections = Vec::new();
        for (signature, members) in groups {
            if !req.symbols.iter().any(|s| s.name == signature) {
                let offset = sym_indices[signature] as usize * sym_size as usize
                    + if class32 { 14 } else { 6 };
                symtab[offset..offset + 2]
                    .copy_from_slice(&((headers.len() + 1) as u16).to_le_bytes());
            }
            group_sections.push(headers.len());
            names.push(".group".into());
            let mut bytes = 1u32.to_le_bytes().to_vec();
            for member in members {
                bytes.extend_from_slice(&member.to_le_bytes());
            }
            data.push(bytes);
            headers.push(Elf64SectionHeader {
                section_type: 17,
                info: sym_indices[signature],
                addralign: 4,
                entsize: 4,
                ..Default::default()
            });
        }
        let symtab_index = headers.len() + 1;
        for i in group_sections {
            headers[i].link = symtab_index as u32;
        }
        for i in rela_sections {
            headers[i].link = symtab_index as u32;
        }
        names.push(".symtab".into());
        data.push(symtab);
        headers.push(Elf64SectionHeader {
            section_type: SHT_SYMTAB,
            link: (symtab_index + 1) as u32,
            info: first_global,
            addralign: table_align,
            entsize: sym_size,
            ..Default::default()
        });
        names.push(".strtab".into());
        data.push(strtab);
        headers.push(Elf64SectionHeader {
            section_type: SHT_STRTAB,
            addralign: 1,
            ..Default::default()
        });
        names.push(".shstrtab".into());
        let mut shstrtab = vec![0];
        for name in names {
            shstrtab.extend_from_slice(name.as_bytes());
            shstrtab.push(0);
        }
        headers.push(Elf64SectionHeader {
            section_type: SHT_STRTAB,
            addralign: 1,
            ..Default::default()
        });
        let mut name_offset = 1;
        for header in &mut headers {
            header.name = name_offset as u32;
            name_offset += shstrtab[name_offset..]
                .iter()
                .position(|b| *b == 0)
                .unwrap()
                + 1;
        }
        data.push(shstrtab);
        let (header_size, shdr_size) = if class32 {
            (ELF32_HEADER_SIZE, ELF32_SECTION_HEADER_SIZE)
        } else {
            (ELF64_HEADER_SIZE, ELF64_SECTION_HEADER_SIZE)
        };
        let mut offset = header_size + (headers.len() + 1) * shdr_size;
        let mut payload = Vec::new();
        for (header, bytes) in headers.iter_mut().zip(data) {
            let padding = (header.addralign as usize - offset % header.addralign as usize)
                % header.addralign as usize;
            payload.resize(payload.len() + padding, 0);
            offset += padding;
            header.offset = offset as u64;
            header.size = bytes.len() as u64;
            if header.section_type != 8 {
                offset += bytes.len();
                payload.extend_from_slice(&bytes);
            }
        }
        let mut f: Vec<u8> = Vec::new();
        write_elf_header_with_shnum(
            &mut f,
            class32,
            machine,
            (headers.len() + 1) as u16,
            headers.len() as u16,
        )?;
        write_elf_section_header_full(&mut f, class32, &Elf64SectionHeader::default())?;
        for header in headers {
            write_elf_section_header_full(&mut f, class32, &header)?;
        }
        f.write_all(&payload)
            .map_err(|e| RasError::ObjectError(e.to_string()))?;
        Ok(f)
    }
}

const MH_MAGIC_64: u32 = 0xfeed_facf;
const MH_OBJECT: u32 = 1;
const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;
const LC_DYSYMTAB: u32 = 0xb;
const LC_BUILD_VERSION: u32 = 0x32;
const PLATFORM_MACOS: u32 = 1;
/// Deployment target recorded in LC_BUILD_VERSION (11.0.0). arm64 Mach-O requires at least 11.0.
const MACOS_MINOS: u32 = 11 << 16;
const CPU_TYPE_X86_64: u32 = 0x0100_0007;
const CPU_SUBTYPE_X86_64_ALL: u32 = 3;
const CPU_TYPE_ARM64: u32 = 0x0100_000c;
const CPU_SUBTYPE_ARM64_ALL: u32 = 0;
const S_ZEROFILL: u32 = 0x1;
const S_THREAD_LOCAL_ZEROFILL: u32 = 0x12;
const S_ATTR_PURE_INSTRUCTIONS: u32 = 0x8000_0000;
const S_ATTR_SOME_INSTRUCTIONS: u32 = 0x0000_0400;
const N_UNDF: u8 = 0x0;
const N_EXT: u8 = 0x1;
const N_ABS: u8 = 0x2;
const N_SECT: u8 = 0xe;
const N_PEXT: u8 = 0x10;
const N_WEAK_DEF: u16 = 0x0080;
const N_WEAK_REF: u16 = 0x0040;
/// Absolute symbols carry N_NO_DEAD_STRIP in clang's output.
const N_NO_DEAD_STRIP: u16 = 0x0020;
const X86_64_RELOC_UNSIGNED: u32 = 0;
const X86_64_RELOC_SIGNED: u32 = 1;
const X86_64_RELOC_BRANCH: u32 = 2;
const X86_64_RELOC_GOT_LOAD: u32 = 3;
const X86_64_RELOC_SIGNED_1: u32 = 6;
const X86_64_RELOC_SIGNED_2: u32 = 7;
const X86_64_RELOC_SIGNED_4: u32 = 8;
const ARM64_RELOC_UNSIGNED: u32 = 0;
const ARM64_RELOC_BRANCH26: u32 = 2;
const ARM64_RELOC_PAGE21: u32 = 3;
const ARM64_RELOC_PAGEOFF12: u32 = 4;
const ARM64_RELOC_GOT_LOAD_PAGE21: u32 = 5;
const ARM64_RELOC_GOT_LOAD_PAGEOFF12: u32 = 6;
const ARM64_RELOC_TLVP_LOAD_PAGE21: u32 = 8;
const ARM64_RELOC_TLVP_LOAD_PAGEOFF12: u32 = 9;
const ARM64_RELOC_ADDEND: u32 = 10;

const NLIST_GROUP_LOCAL: u8 = 0;
const NLIST_GROUP_EXTDEF: u8 = 1;
const NLIST_GROUP_UNDEF: u8 = 2;

fn macho_is_zerofill(flags: u32) -> bool {
    matches!(flags & 0xff, S_ZEROFILL | S_THREAD_LOCAL_ZEROFILL)
}

fn macho_section_kind(name: &str) -> Option<(&'static str, &'static str, u32)> {
    match name {
        ".text" => Some((
            "__TEXT",
            "__text",
            S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS,
        )),
        ".rodata" => Some(("__TEXT", "__const", 0)),
        ".data" => Some(("__DATA", "__data", 0)),
        ".bss" => Some(("__DATA", "__bss", S_ZEROFILL)),
        _ => None,
    }
}

/// Mach-O requires a symbol to be private (never in the symbol table) when its name is
/// `L`-prefixed and it is not global. Internal CFI labels are private as well.
fn macho_is_private(sym: &ObjectSymbol) -> bool {
    sym.attributes.internal || (sym.name.starts_with('L') && !sym.global && !sym.attributes.weak)
}

struct MachORelocSpec {
    r_type: u32,
    width_log2: u32,
    pcrel: bool,
}

fn arm64_reloc_spec(kind: u32) -> Result<MachORelocSpec, RasError> {
    let spec = |r_type: u32, width_log2: u32, pcrel: bool| MachORelocSpec {
        r_type,
        width_log2,
        pcrel,
    };
    match kind {
        257 => Ok(spec(ARM64_RELOC_UNSIGNED, 3, false)),
        258 => Ok(spec(ARM64_RELOC_UNSIGNED, 2, false)),
        259 => Ok(spec(ARM64_RELOC_UNSIGNED, 1, false)),
        282 | 283 => Ok(spec(ARM64_RELOC_BRANCH26, 2, true)),
        275 => Ok(spec(ARM64_RELOC_PAGE21, 2, true)),
        // Darwin has one page offset relocation, the linker reads the scale off the instruction.
        277 | 278 | 284 | 285 | 286 | 299 => Ok(spec(ARM64_RELOC_PAGEOFF12, 2, false)),
        311 => Ok(spec(ARM64_RELOC_GOT_LOAD_PAGE21, 2, true)),
        312 => Ok(spec(ARM64_RELOC_GOT_LOAD_PAGEOFF12, 2, false)),
        313 => Ok(spec(ARM64_RELOC_TLVP_LOAD_PAGE21, 2, true)),
        314 => Ok(spec(ARM64_RELOC_TLVP_LOAD_PAGEOFF12, 2, false)),
        _ => Err(RasError::ObjectError(format!(
            "{} (kind {kind}) has no Mach-O equivalent",
            aarch64_reloc_name(kind)
        ))),
    }
}

fn x86_64_reloc_spec(
    kind: u32,
    addend: i64,
    trailing: usize,
    branch: bool,
) -> Result<MachORelocSpec, RasError> {
    let abs = |width_log2| MachORelocSpec {
        r_type: X86_64_RELOC_UNSIGNED,
        width_log2,
        pcrel: false,
    };
    let unsupported = || {
        RasError::ObjectError(format!(
            "x86_64 relocation kind {kind} (trailing {trailing}, branch {branch}) has no Mach-O equivalent"
        ))
    };
    match kind {
        1 => Ok(abs(3)),
        10 => Ok(abs(2)),
        12 => Ok(abs(1)),
        14 => Ok(abs(0)),
        2 | 4 if branch || kind == 4 => {
            if trailing != 0 {
                return Err(unsupported());
            }
            Ok(MachORelocSpec {
                r_type: X86_64_RELOC_BRANCH,
                width_log2: 2,
                pcrel: true,
            })
        }
        2 => {
            // clang keeps the plain SIGNED type whenever the reference has an explicit
            // addend; SIGNED_n is used only for a bare reference.
            let explicit_zero = addend + 4 + trailing as i64 == 0;
            let r_type = match trailing {
                0 => X86_64_RELOC_SIGNED,
                1 if explicit_zero => X86_64_RELOC_SIGNED_1,
                2 if explicit_zero => X86_64_RELOC_SIGNED_2,
                4 if explicit_zero => X86_64_RELOC_SIGNED_4,
                1 | 2 | 4 => X86_64_RELOC_SIGNED,
                _ => return Err(unsupported()),
            };
            Ok(MachORelocSpec {
                r_type,
                width_log2: 2,
                pcrel: true,
            })
        }
        42 if trailing == 0 => Ok(MachORelocSpec {
            r_type: X86_64_RELOC_GOT_LOAD,
            width_log2: 2,
            pcrel: true,
        }),
        _ => Err(unsupported()),
    }
}

struct MachONlist {
    name: String,
    n_type: u8,
    n_sect: u8,
    n_desc: u16,
    n_value: u64,
    group: u8,
}

struct MachOSection {
    segment: String,
    name: String,
    flags: u32,
    align: u64,
    addr: u64,
    data: Vec<u8>,
    relocs: Vec<(u32, u32)>,
}

impl MachOSection {
    fn is_zerofill(&self) -> bool {
        macho_is_zerofill(self.flags)
    }
}

enum MachORef {
    Extern { index: u32, delta: i64 },
    Section { ord: usize, value: u64 },
}

fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_name16(buf: &mut Vec<u8>, name: &str) {
    let mut field = [0u8; 16];
    field[..name.len()].copy_from_slice(name.as_bytes());
    buf.extend_from_slice(&field);
}

fn write_field(data: &mut [u8], offset: usize, width: usize, value: i64) -> Result<(), RasError> {
    let bits = width as u32 * 8;
    let value = i128::from(value);
    if value < -(1i128 << (bits - 1)) || value >= (1i128 << bits) {
        return Err(RasError::ObjectError(format!(
            "Mach-O relocation value {value} does not fit in {width} bytes"
        )));
    }
    data[offset..offset + width].copy_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

fn macho_object_bytes(req: &ObjectWriteRequest<'_>) -> Result<Vec<u8>, RasError> {
    if let Some(sym) = req.symbols.iter().find(|s| s.attributes.protected) {
        return Err(RasError::ObjectError(format!(
            "protected visibility is not supported for Mach-O: {}",
            sym.name
        )));
    }
    let arm64 = req.target_arch == TargetArchitecture::Aarch64;
    let (cputype, cpusubtype) = match req.target_arch {
        TargetArchitecture::X86_64 => (CPU_TYPE_X86_64, CPU_SUBTYPE_X86_64_ALL),
        TargetArchitecture::Aarch64 => (CPU_TYPE_ARM64, CPU_SUBTYPE_ARM64_ALL),
        other => {
            return Err(RasError::ObjectError(format!(
                "Mach-O objects are written for x86_64 and arm64 only, not {other:?}"
            )));
        }
    };

    let mut source = req.sections.to_vec();
    if source.is_empty() {
        source.push(Section {
            group: None,
            name: ".text".into(),
            flags: crate::parser::SectionFlags {
                alloc: true,
                exec: true,
                write: false,
                tls: false,
            },
            data: req.code.to_vec(),
            alignment: 1,
            macho: None,
        });
    }

    let mut sections = Vec::with_capacity(source.len());
    let mut section_ord: HashMap<&str, usize> = HashMap::new();
    let mut addr = 0u64;
    for s in &source {
        let (segment, name, flags) = match &s.macho {
            Some(section_type) => {
                let (segment, name) = s.name.split_once(',').ok_or_else(|| {
                    RasError::ObjectError(format!("Mach-O section {} has no segment", s.name))
                })?;
                if segment.is_empty() || name.is_empty() || segment.len() > 16 || name.len() > 16 {
                    return Err(RasError::ObjectError(format!(
                        "Mach-O section name {} needs a non-empty segment and section of at most 16 bytes each",
                        s.name
                    )));
                }
                (
                    segment.to_string(),
                    name.to_string(),
                    section_type.kind | section_type.attributes,
                )
            }
            None => {
                let (segment, name, flags) = macho_section_kind(&s.name).ok_or_else(|| {
                    RasError::ObjectError(format!(
                        "Mach-O has no section for {}; write Mach-O sections as segment,section",
                        s.name
                    ))
                })?;
                (segment.to_string(), name.to_string(), flags)
            }
        };
        if s.group.is_some() {
            return Err(RasError::ObjectError(format!(
                "Mach-O has no section groups; {} is in a comdat group",
                s.name
            )));
        }
        if !s.alignment.is_power_of_two() {
            return Err(RasError::ObjectError(format!(
                "section {} alignment {} is not a power of two",
                s.name, s.alignment
            )));
        }
        let align = s.alignment as u64;
        let zerofill = macho_is_zerofill(flags);
        if zerofill && s.data.iter().any(|&b| b != 0) {
            return Err(RasError::ObjectError(format!(
                "{} holds initialized bytes but Mach-O zero-fill sections cannot",
                s.name
            )));
        }
        let flags = if flags & S_ATTR_PURE_INSTRUCTIONS != 0 && s.data.is_empty() {
            flags & !S_ATTR_SOME_INSTRUCTIONS
        } else {
            flags
        };
        addr = align_up(addr, align);
        section_ord.insert(s.name.as_str(), sections.len());
        sections.push(MachOSection {
            segment,
            name,
            flags,
            align,
            addr,
            data: s.data.clone(),
            relocs: Vec::new(),
        });
        addr += s.data.len() as u64;
    }
    let sections_end = addr;

    let symbol_by_name = {
        let mut map: HashMap<&str, &ObjectSymbol> = HashMap::new();
        for sym in req.symbols {
            map.entry(sym.name.as_str()).or_insert(sym);
        }
        map
    };

    let mut nlists: Vec<MachONlist> = Vec::new();
    let mut index_of: HashMap<String, u32> = HashMap::new();
    if arm64 {
        for (ord, sec) in sections.iter().enumerate() {
            nlists.push(MachONlist {
                name: format!("ltmp{ord}"),
                n_type: N_SECT,
                n_sect: (ord + 1) as u8,
                n_desc: 0,
                n_value: sec.addr,
                group: NLIST_GROUP_LOCAL,
            });
        }
    }
    for sym in req.symbols.iter().filter(|s| !macho_is_private(s)) {
        let ext = sym.global || sym.attributes.weak;
        let nlist = match sym.section.as_str() {
            "" => {
                if !ext {
                    return Err(RasError::ObjectError(format!(
                        "local symbol {} is never defined",
                        sym.name
                    )));
                }
                MachONlist {
                    name: sym.name.clone(),
                    n_type: N_UNDF | N_EXT,
                    n_sect: 0,
                    n_desc: if sym.attributes.weak { N_WEAK_REF } else { 0 },
                    n_value: 0,
                    group: NLIST_GROUP_UNDEF,
                }
            }
            "*COM*" => {
                let (size, align) = sym.attributes.common.unwrap_or((0, sym.value));
                if !align.is_power_of_two() || align > 1 << 15 {
                    return Err(RasError::ObjectError(format!(
                        "common symbol {} has unsupported alignment {align}",
                        sym.name
                    )));
                }
                MachONlist {
                    name: sym.name.clone(),
                    n_type: N_UNDF | N_EXT,
                    n_sect: 0,
                    n_desc: (align.trailing_zeros() as u16) << 8,
                    n_value: size,
                    group: NLIST_GROUP_UNDEF,
                }
            }
            "*ABS*" => MachONlist {
                name: sym.name.clone(),
                n_type: N_ABS | if ext { N_EXT } else { 0 },
                n_sect: 0,
                n_desc: N_NO_DEAD_STRIP,
                n_value: sym.value,
                group: if ext {
                    NLIST_GROUP_EXTDEF
                } else {
                    NLIST_GROUP_LOCAL
                },
            },
            section => {
                let ord = *section_ord.get(section).ok_or_else(|| {
                    RasError::ObjectError(format!(
                        "symbol {} is in unknown section {section}",
                        sym.name
                    ))
                })?;
                let mut n_type = N_SECT;
                let mut n_desc = 0;
                if ext {
                    n_type |= N_EXT;
                    if sym.attributes.hidden
                        || sym.attributes.internal_visibility
                        || sym.attributes.private_extern
                    {
                        n_type |= N_PEXT;
                    }
                    if sym.attributes.weak {
                        n_desc |= N_WEAK_DEF;
                    }
                }
                MachONlist {
                    name: sym.name.clone(),
                    n_type,
                    n_sect: (ord + 1) as u8,
                    n_desc,
                    n_value: sections[ord].addr + sym.value,
                    group: if ext {
                        NLIST_GROUP_EXTDEF
                    } else {
                        NLIST_GROUP_LOCAL
                    },
                }
            }
        };
        nlists.push(nlist);
    }
    for reloc in req.relocations {
        if reloc.section_symbol {
            continue;
        }
        let defined_here = symbol_by_name
            .get(reloc.symbol.as_str())
            .is_some_and(|s| !s.section.is_empty() && s.section != "*COM*" && s.section != "*ABS*");
        if !defined_here && !nlists.iter().any(|n| n.name == reloc.symbol) {
            nlists.push(MachONlist {
                name: reloc.symbol.clone(),
                n_type: N_UNDF | N_EXT,
                n_sect: 0,
                n_desc: 0,
                n_value: 0,
                group: NLIST_GROUP_UNDEF,
            });
        }
    }
    nlists.sort_by_key(|n| n.group);
    for (i, nlist) in nlists.iter().enumerate() {
        index_of.entry(nlist.name.clone()).or_insert(i as u32);
    }
    let count = |group| nlists.iter().filter(|n| n.group == group).count() as u32;
    let (n_local, n_extdef, n_undef) = (
        count(NLIST_GROUP_LOCAL),
        count(NLIST_GROUP_EXTDEF),
        count(NLIST_GROUP_UNDEF),
    );

    let atoms: Vec<(usize, u64, &str)> = req
        .symbols
        .iter()
        .filter(|s| !macho_is_private(s))
        .filter_map(|s| {
            section_ord
                .get(s.section.as_str())
                .map(|&ord| (ord, s.value, s.name.as_str()))
        })
        .collect();

    for reloc in req.relocations {
        let ord = *section_ord.get(reloc.section.as_str()).ok_or_else(|| {
            RasError::ObjectError(format!(
                "relocation in section {} which has no Mach-O section",
                reloc.section
            ))
        })?;
        if sections[ord].is_zerofill() {
            return Err(RasError::ObjectError(format!(
                "relocation inside zero-fill section {}",
                reloc.section
            )));
        }
        let spec = if arm64 {
            arm64_reloc_spec(reloc.kind)?
        } else {
            x86_64_reloc_spec(reloc.kind, reloc.addend, reloc.trailing, reloc.branch)?
        };
        // clang stores addend minus the immediate width, so the content bias stays 4.
        let bias = if spec.pcrel && !arm64 { 4 } else { 0 };
        let field = reloc.offset as u64;
        let field_addr = sections[ord].addr + field;
        let width = 1usize << spec.width_log2;
        let addend = reloc.addend + bias;

        if reloc.section_symbol {
            let target_ord = *section_ord.get(reloc.symbol.as_str()).ok_or_else(|| {
                RasError::ObjectError(format!(
                    "relocation against unknown section {}",
                    reloc.symbol
                ))
            })?;
            if spec.pcrel {
                return Err(RasError::ObjectError(format!(
                    "pc-relative relocation against section {} is not supported for Mach-O",
                    reloc.symbol
                )));
            }
            let content = sections[target_ord].addr as i64 + reloc.addend;
            write_field(&mut sections[ord].data, reloc.offset, width, content)?;
            let r_address = u32::try_from(reloc.offset)
                .map_err(|_| RasError::ObjectError("relocation offset too large".into()))?;
            let info = ((target_ord as u32 + 1) & 0x00ff_ffff)
                | (spec.width_log2 << 25)
                | (spec.r_type << 28);
            sections[ord].relocs.push((r_address, info));
            continue;
        }

        let target = match symbol_by_name.get(reloc.symbol.as_str()) {
            Some(sym) if sym.section.as_str() == "*ABS*" => {
                return Err(RasError::ObjectError(format!(
                    "relocation against absolute symbol {} is not supported for Mach-O",
                    sym.name
                )));
            }
            Some(sym)
                if macho_is_private(sym) && !sym.section.is_empty() && sym.section != "*COM*" =>
            {
                let sym_ord = *section_ord.get(sym.section.as_str()).ok_or_else(|| {
                    RasError::ObjectError(format!("symbol {} has unknown section", sym.name))
                })?;
                if arm64 {
                    MachORef::Extern {
                        index: index_of[format!("ltmp{sym_ord}").as_str()],
                        delta: sym.value as i64,
                    }
                } else {
                    let atom = atoms
                        .iter()
                        .filter(|(o, value, _)| *o == sym_ord && *value <= sym.value)
                        .max_by_key(|(_, value, _)| *value);
                    match atom {
                        Some(&(_, atom_value, atom_name)) => MachORef::Extern {
                            index: index_of[atom_name],
                            delta: (sym.value - atom_value) as i64,
                        },
                        None => MachORef::Section {
                            ord: sym_ord,
                            value: sym.value,
                        },
                    }
                }
            }
            Some(sym) if macho_is_private(sym) => {
                return Err(RasError::ObjectError(format!(
                    "private symbol {} is undefined",
                    sym.name
                )));
            }
            _ => MachORef::Extern {
                index: index_of[reloc.symbol.as_str()],
                delta: 0,
            },
        };

        let r_address = u32::try_from(reloc.offset)
            .map_err(|_| RasError::ObjectError("relocation offset too large".into()))?;
        let (symbol_num, is_extern) = if arm64 {
            let MachORef::Extern { index, delta } = target else {
                return Err(RasError::ObjectError(format!(
                    "arm64 relocation in {} has no symbol to bind",
                    reloc.section
                )));
            };
            let total = reloc.addend + delta;
            if spec.r_type == ARM64_RELOC_UNSIGNED {
                write_field(&mut sections[ord].data, reloc.offset, width, total)?;
            } else if total != 0 {
                // The instruction keeps a zero immediate; the addend travels in an ADDEND entry.
                let value = u32::try_from(total)
                    .ok()
                    .filter(|v| *v <= 0x00ff_ffff)
                    .ok_or_else(|| {
                        RasError::ObjectError(format!(
                            "arm64 addend {total} in {} is outside 0..2^24",
                            reloc.section
                        ))
                    })?;
                sections[ord]
                    .relocs
                    .push((r_address, value | (2 << 25) | (ARM64_RELOC_ADDEND << 28)));
            }
            (index, true)
        } else {
            let (symbol_num, is_extern, content) = match target {
                MachORef::Extern { index, delta } => (index, true, addend + delta),
                MachORef::Section {
                    ord: target_ord,
                    value,
                } => {
                    let target_addr = sections[target_ord].addr as i64 + value as i64 + addend;
                    let content = if spec.pcrel {
                        target_addr - (field_addr as i64 + 4)
                    } else {
                        target_addr
                    };
                    ((target_ord + 1) as u32, false, content)
                }
            };
            write_field(&mut sections[ord].data, reloc.offset, width, content)?;
            (symbol_num, is_extern)
        };
        let info = (symbol_num & 0x00ff_ffff)
            | (u32::from(spec.pcrel) << 24)
            | (spec.width_log2 << 25)
            | (u32::from(is_extern) << 27)
            | (spec.r_type << 28);
        sections[ord].relocs.push((r_address, info));
    }

    let mut strtab = vec![0u8];
    let mut symtab = Vec::with_capacity(nlists.len() * 16);
    for nlist in &nlists {
        put_u32(&mut symtab, strtab.len() as u32);
        strtab.extend_from_slice(nlist.name.as_bytes());
        strtab.push(0);
        symtab.push(nlist.n_type);
        symtab.push(nlist.n_sect);
        symtab.extend_from_slice(&nlist.n_desc.to_le_bytes());
        symtab.extend_from_slice(&nlist.n_value.to_le_bytes());
    }

    let ncmds = 4u32;
    let sizeofcmds = (72 + 80 * sections.len() as u32) + 24 + 24 + 80;
    let max_align = sections.iter().map(|s| s.align).max().unwrap_or(1);
    let data_start = align_up(32 + u64::from(sizeofcmds), max_align);
    let data_end = data_start + sections_end;
    let file_offsets: Vec<u64> = sections.iter().map(|s| data_start + s.addr).collect();
    let mut cursor = data_end;
    let mut reloc_offsets = vec![0u64; sections.len()];
    for (i, sec) in sections.iter().enumerate() {
        if sec.relocs.is_empty() {
            continue;
        }
        cursor = align_up(cursor, 8);
        reloc_offsets[i] = cursor;
        cursor += 8 * sec.relocs.len() as u64;
    }
    let symoff = align_up(cursor, 8);
    let stroff = symoff + symtab.len() as u64;
    let strsize = strtab.len() as u64;

    let mut out = Vec::with_capacity((stroff + strsize) as usize);
    put_u32(&mut out, MH_MAGIC_64);
    put_u32(&mut out, cputype);
    put_u32(&mut out, cpusubtype);
    put_u32(&mut out, MH_OBJECT);
    put_u32(&mut out, ncmds);
    put_u32(&mut out, sizeofcmds);
    put_u32(&mut out, 0);
    put_u32(&mut out, 0);

    put_u32(&mut out, LC_SEGMENT_64);
    put_u32(&mut out, 72 + 80 * sections.len() as u32);
    put_name16(&mut out, "");
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&sections_end.to_le_bytes());
    out.extend_from_slice(&data_start.to_le_bytes());
    out.extend_from_slice(&sections_end.to_le_bytes());
    put_u32(&mut out, 7);
    put_u32(&mut out, 7);
    put_u32(&mut out, sections.len() as u32);
    put_u32(&mut out, 0);
    for (i, sec) in sections.iter().enumerate() {
        put_name16(&mut out, &sec.name);
        put_name16(&mut out, &sec.segment);
        out.extend_from_slice(&sec.addr.to_le_bytes());
        out.extend_from_slice(&(sec.data.len() as u64).to_le_bytes());
        put_u32(&mut out, file_offsets[i] as u32);
        put_u32(&mut out, sec.align.trailing_zeros());
        put_u32(&mut out, reloc_offsets[i] as u32);
        put_u32(&mut out, sec.relocs.len() as u32);
        put_u32(&mut out, sec.flags);
        put_u32(&mut out, 0);
        put_u32(&mut out, 0);
        put_u32(&mut out, 0);
    }

    put_u32(&mut out, LC_BUILD_VERSION);
    put_u32(&mut out, 24);
    put_u32(&mut out, PLATFORM_MACOS);
    put_u32(&mut out, MACOS_MINOS);
    put_u32(&mut out, 0);
    put_u32(&mut out, 0);

    put_u32(&mut out, LC_SYMTAB);
    put_u32(&mut out, 24);
    put_u32(&mut out, symoff as u32);
    put_u32(&mut out, nlists.len() as u32);
    put_u32(&mut out, stroff as u32);
    put_u32(&mut out, strsize as u32);

    put_u32(&mut out, LC_DYSYMTAB);
    put_u32(&mut out, 80);
    put_u32(&mut out, 0);
    put_u32(&mut out, n_local);
    put_u32(&mut out, n_local);
    put_u32(&mut out, n_extdef);
    put_u32(&mut out, n_local + n_extdef);
    put_u32(&mut out, n_undef);
    for _ in 0..12 {
        put_u32(&mut out, 0);
    }

    out.resize(data_start as usize, 0);
    for (i, sec) in sections.iter().enumerate() {
        if sec.is_zerofill() {
            continue;
        }
        out.resize(file_offsets[i] as usize, 0);
        out.extend_from_slice(&sec.data);
    }
    out.resize(data_end as usize, 0);
    for sec in &sections {
        if sec.relocs.is_empty() {
            continue;
        }
        out.resize(align_up(out.len() as u64, 8) as usize, 0);
        for &(r_address, info) in &sec.relocs {
            put_u32(&mut out, r_address);
            put_u32(&mut out, info);
        }
    }
    out.resize(symoff as usize, 0);
    out.extend_from_slice(&symtab);
    out.extend_from_slice(&strtab);
    Ok(out)
}

pub struct MachOWriter;

impl Default for MachOWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl MachOWriter {
    pub fn new() -> Self {
        Self
    }
}

impl ObjectWriter for MachOWriter {
    fn object_bytes(&mut self, req: &ObjectWriteRequest<'_>) -> Result<Vec<u8>, RasError> {
        refuse_image_relative(req, "Mach-O")?;
        macho_object_bytes(req)
    }
}

pub(crate) const X86_64_SECREL32: u32 = 270;

/// Fixup kinds CodeView needs that no ELF relocation number names: the offset of a
/// symbol within its section, and that section's one-based index. Numbered above every
/// real relocation of every architecture this writer serves so they cannot collide.
pub(crate) const COFF_SECREL32: u32 = 0x1000_0001;
pub(crate) const COFF_SECTION: u32 = 0x1000_0002;
/// An address relative to the image base, which `.rva` and `@IMGREL` ask for.
pub(crate) const COFF_IMGREL32: u32 = 0x1000_0003;

const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
const IMAGE_SCN_MEM_DISCARDABLE: u32 = 0x0200_0000;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;
const IMAGE_SYM_UNDEFINED: i16 = 0;
const IMAGE_SYM_ABSOLUTE: i16 = -1;
const IMAGE_SYM_CLASS_EXTERNAL: u8 = 2;
const IMAGE_SYM_CLASS_STATIC: u8 = 3;
const IMAGE_SYM_CLASS_WEAK_EXTERNAL: u8 = 105;
const IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY: u32 = 1;
const IMAGE_SYM_DTYPE_FUNCTION: u16 = 0x20;

const IMAGE_FILE_MACHINE_ARM64: u16 = 0xAA64;

/// Type indicators from the Microsoft PE/COFF specification, "COFF Relocations", x64
/// processors. `IMAGE_REL_AMD64_SECREL` was once `0x000E`, which is
/// `IMAGE_REL_AMD64_SREL32`, and a test suite that spelled it symbolically agreed.
const IMAGE_REL_AMD64_ADDR64: u16 = 1;
const IMAGE_REL_AMD64_ADDR32: u16 = 2;
const IMAGE_REL_AMD64_ADDR32NB: u16 = 3;
const IMAGE_REL_AMD64_REL32: u16 = 4;
const IMAGE_REL_AMD64_SECREL: u16 = 0x000B;
const IMAGE_REL_AMD64_SECTION: u16 = 0x000A;

/// Type indicators from the Microsoft PE/COFF specification, "COFF Relocations", ARM64
/// processors. `IMAGE_REL_ARM64_ADDR64` is genuinely `0x000E` on this machine type.
const IMAGE_REL_ARM64_ADDR32: u16 = 0x0001;
const IMAGE_REL_ARM64_ADDR32NB: u16 = 0x0002;
const IMAGE_REL_ARM64_BRANCH26: u16 = 0x0003;
const IMAGE_REL_ARM64_PAGEBASE_REL21: u16 = 0x0004;
const IMAGE_REL_ARM64_PAGEOFFSET_12A: u16 = 0x0006;
const IMAGE_REL_ARM64_PAGEOFFSET_12L: u16 = 0x0007;
const IMAGE_REL_ARM64_ADDR64: u16 = 0x000E;
const IMAGE_REL_ARM64_BRANCH19: u16 = 0x000F;
const IMAGE_REL_ARM64_BRANCH14: u16 = 0x0010;
const IMAGE_REL_ARM64_SECREL: u16 = 0x0008;
const IMAGE_REL_ARM64_SECTION: u16 = 0x000D;

const COFF_SECTION_HEADER_SIZE: usize = 40;
const COFF_SYMBOL_SIZE: usize = 18;
const COFF_RELOC_SIZE: usize = 10;
const COFF_RAW_ALIGN: usize = 16;

fn coff_is_uninitialized(section: &Section) -> bool {
    section.name == ".bss"
}

fn coff_align_bits(align: usize) -> Result<u32, RasError> {
    if !align.is_power_of_two() || align > 8192 {
        return Err(RasError::ObjectError(format!(
            "COFF section alignment {align} is not a power of two up to 8192"
        )));
    }
    Ok((align.trailing_zeros() + 1) << 20)
}

fn coff_section_characteristics(section: &Section) -> Result<u32, RasError> {
    let mut flags = coff_align_bits(section.alignment.max(1))?;
    if section.flags.exec {
        flags |= IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ;
    } else if coff_is_uninitialized(section) {
        flags |= IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE;
    } else {
        flags |= IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ;
        if section.flags.write {
            flags |= IMAGE_SCN_MEM_WRITE;
        }
    }
    if !section.flags.alloc {
        flags |= IMAGE_SCN_MEM_DISCARDABLE;
    }
    Ok(flags)
}

fn coff_reloc_type(kind: u32, symbol: &str) -> Result<(u16, usize), RasError> {
    match kind {
        1 => Ok((IMAGE_REL_AMD64_ADDR64, 8)),
        10 | 11 => Ok((IMAGE_REL_AMD64_ADDR32, 4)),
        X86_64_SECREL32 | COFF_SECREL32 => Ok((IMAGE_REL_AMD64_SECREL, 4)),
        COFF_SECTION => Ok((IMAGE_REL_AMD64_SECTION, 2)),
        COFF_IMGREL32 => Ok((IMAGE_REL_AMD64_ADDR32NB, 4)),
        12 | 14 => Err(RasError::ObjectError(format!(
            "x86_64 relocation kind {kind} for {symbol} has no COFF equivalent"
        ))),
        _ => Ok((IMAGE_REL_AMD64_REL32, 4)),
    }
}

/// Where a COFF relocation's addend lives: in a data field of this many bytes,
/// or inside the instruction word, which the writer must not rewrite.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CoffAddend {
    Field(usize),
    Instruction,
}

fn aarch64_reloc_name(kind: u32) -> &'static str {
    match kind {
        257 => "R_AARCH64_ABS64",
        258 => "R_AARCH64_ABS32",
        259 => "R_AARCH64_ABS16",
        260 => "R_AARCH64_PREL64",
        261 => "R_AARCH64_PREL32",
        275 => "R_AARCH64_ADR_PREL_PG_HI21",
        277 => "R_AARCH64_ADD_ABS_LO12_NC",
        279 => "R_AARCH64_TSTBR14",
        280 => "R_AARCH64_CONDBR19",
        282 => "R_AARCH64_JUMP26",
        283 => "R_AARCH64_CALL26",
        286 => "R_AARCH64_LDST64_ABS_LO12_NC",
        311 => "R_AARCH64_ADR_GOT_PAGE",
        312 => "R_AARCH64_LD64_GOT_LO12_NC",
        313 | 314 => "Mach-O TLVP",
        541 => "R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21",
        542 => "R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC",
        549 => "R_AARCH64_TLSLE_ADD_TPREL_HI12",
        550 => "R_AARCH64_TLSLE_ADD_TPREL_LO12",
        551 => "R_AARCH64_TLSLE_ADD_TPREL_LO12_NC",
        _ => "unknown AArch64 relocation",
    }
}

fn coff_arm64_reloc_type(
    kind: u32,
    symbol: &str,
    addend: i64,
) -> Result<(u16, CoffAddend), RasError> {
    let (ty, addend_place) = match kind {
        COFF_SECREL32 => (IMAGE_REL_ARM64_SECREL, CoffAddend::Field(4)),
        COFF_SECTION => (IMAGE_REL_ARM64_SECTION, CoffAddend::Field(2)),
        COFF_IMGREL32 => (IMAGE_REL_ARM64_ADDR32NB, CoffAddend::Field(4)),
        257 => (IMAGE_REL_ARM64_ADDR64, CoffAddend::Field(8)),
        258 => (IMAGE_REL_ARM64_ADDR32, CoffAddend::Field(4)),
        275 => (IMAGE_REL_ARM64_PAGEBASE_REL21, CoffAddend::Instruction),
        277 => (IMAGE_REL_ARM64_PAGEOFFSET_12A, CoffAddend::Instruction),
        286 => (IMAGE_REL_ARM64_PAGEOFFSET_12L, CoffAddend::Instruction),
        279 => (IMAGE_REL_ARM64_BRANCH14, CoffAddend::Instruction),
        280 => (IMAGE_REL_ARM64_BRANCH19, CoffAddend::Instruction),
        282 | 283 => (IMAGE_REL_ARM64_BRANCH26, CoffAddend::Instruction),
        _ => {
            return Err(RasError::ObjectError(format!(
                "{} against {symbol} has no ARM64 COFF relocation",
                aarch64_reloc_name(kind)
            )));
        }
    };
    if addend_place == CoffAddend::Instruction && addend != 0 {
        return Err(RasError::ObjectError(format!(
            "{} against {symbol} carries addend {addend}; the ARM64 COFF writer only encodes a zero addend in an instruction field",
            aarch64_reloc_name(kind)
        )));
    }
    Ok((ty, addend_place))
}

fn coff_symbol_entry(
    strtab: &mut Vec<u8>,
    name: &str,
    value: u32,
    section: i16,
    ty: u16,
    class: u8,
    aux: u8,
) -> [u8; COFF_SYMBOL_SIZE] {
    let mut entry = [0u8; COFF_SYMBOL_SIZE];
    let nb = name.as_bytes();
    if nb.len() <= 8 {
        entry[..nb.len()].copy_from_slice(nb);
    } else {
        let off = strtab.len() as u32;
        entry[4..8].copy_from_slice(&off.to_le_bytes());
        strtab.extend_from_slice(nb);
        strtab.push(0);
    }
    entry[8..12].copy_from_slice(&value.to_le_bytes());
    entry[12..14].copy_from_slice(&section.to_le_bytes());
    entry[14..16].copy_from_slice(&ty.to_le_bytes());
    entry[16] = class;
    entry[17] = aux;
    entry
}

/// The auxiliary record that must follow every `IMAGE_SYM_CLASS_WEAK_EXTERNAL`
/// symbol: the symbol-table index of the default definition, then the search
/// characteristics. A weak external without this record is malformed.
fn coff_weak_aux(default_index: u32) -> [u8; COFF_SYMBOL_SIZE] {
    let mut aux = [0u8; COFF_SYMBOL_SIZE];
    aux[0..4].copy_from_slice(&default_index.to_le_bytes());
    aux[4..8].copy_from_slice(&IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY.to_le_bytes());
    aux
}

/// The name of the default definition a weak external falls back to. COFF holds
/// the definition under this name and leaves the weak name undefined, so a
/// strong definition anywhere else in the link wins. link.exe takes only an
/// external default, so the name carries the first strong external of the object,
/// as LLVM does, to keep it apart from the default another object gives `name`.
fn coff_weak_default_name(name: &str, unique: Option<&str>) -> String {
    match unique {
        Some(unique) => format!(".weak.{name}.default.{unique}"),
        None => format!(".weak.{name}.default"),
    }
}

/// COFF has no symbol visibility. The only thing it can say about a symbol that
/// is not meant to be exported is that it is file-local, which is the storage
/// class `IMAGE_SYM_CLASS_STATIC`.
fn coff_is_file_local(attributes: &crate::parser::SymbolAttributes) -> bool {
    attributes.hidden || attributes.internal_visibility || attributes.private_extern
}

fn coff_section_name(name: &str, strtab: &mut Vec<u8>) -> Result<[u8; 8], RasError> {
    let mut out = [0u8; 8];
    if name.len() <= 8 {
        out[..name.len()].copy_from_slice(name.as_bytes());
        return Ok(out);
    }
    let off = strtab.len();
    strtab.extend_from_slice(name.as_bytes());
    strtab.push(0);
    let text = format!("/{off}");
    if text.len() > 8 {
        return Err(RasError::ObjectError(format!(
            "COFF string table offset {off} does not fit a section name"
        )));
    }
    out[..text.len()].copy_from_slice(text.as_bytes());
    Ok(out)
}

pub struct CoffWriter;

impl Default for CoffWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl CoffWriter {
    pub fn new() -> Self {
        Self
    }
}

impl ObjectWriter for CoffWriter {
    fn object_bytes(&mut self, req: &ObjectWriteRequest<'_>) -> Result<Vec<u8>, RasError> {
        let relocations = req.relocations;
        let machine = match req.target_arch {
            TargetArchitecture::X86_64 => IMAGE_FILE_MACHINE_AMD64,
            TargetArchitecture::Aarch64 => IMAGE_FILE_MACHINE_ARM64,
            other => {
                return Err(RasError::ObjectError(format!(
                    "COFF writer: unsupported architecture {other:?}"
                )));
            }
        };
        let x86 = machine == IMAGE_FILE_MACHINE_AMD64;
        if x86 && let Some(r) = relocations.iter().find(|r| matches!(r.kind, 18 | 22 | 23)) {
            return Err(RasError::ObjectError(format!(
                "x86_64 TLS relocation kind {} for {} is only supported in ELF objects",
                r.kind, r.symbol
            )));
        }
        if let Some(s) = req.sections.iter().find(|s| s.macho.is_some()) {
            return Err(RasError::ObjectError(format!(
                "Mach-O section {} cannot be written to a COFF object",
                s.name
            )));
        }
        if let Some(sym) = req.symbols.iter().find(|s| s.attributes.protected) {
            return Err(RasError::ObjectError(format!(
                "protected visibility is not supported for COFF: {}",
                sym.name
            )));
        }
        if let Some(sym) = req.symbols.iter().find(|s| {
            s.attributes.weak && (coff_is_file_local(&s.attributes) || s.attributes.local)
        }) {
            return Err(RasError::ObjectError(format!(
                "COFF cannot make {} both weak and file-local: a weak external is resolved across objects, so it has to be an external symbol",
                sym.name
            )));
        }
        if let Some(sym) = req
            .symbols
            .iter()
            .find(|s| s.attributes.weak && s.section == "*COM*")
        {
            return Err(RasError::ObjectError(format!(
                "COFF has no weak common symbol: {}",
                sym.name
            )));
        }
        for r in relocations
            .iter()
            .filter(|r| x86 && r.kind == X86_64_SECREL32)
        {
            if let Some(sym) = req.symbols.iter().find(|s| s.name == r.symbol)
                && !sym.section.is_empty()
                && !sym.section.starts_with(".tls")
            {
                return Err(RasError::ObjectError(format!(
                    "@SECREL32 target {} is in {}, not a .tls section",
                    r.symbol, sym.section
                )));
            }
        }

        let default_sections;
        let sections: &[Section] = if req.sections.is_empty() {
            default_sections = [Section {
                group: None,
                name: ".text".into(),
                flags: crate::parser::SectionFlags {
                    alloc: true,
                    exec: true,
                    write: false,
                    tls: false,
                },
                data: req.code.to_vec(),
                alignment: 16,
                macho: None,
            }];
            &default_sections
        } else {
            req.sections
        };
        if sections.len() > u16::MAX as usize {
            return Err(RasError::ObjectError(
                "COFF object has too many sections".into(),
            ));
        }
        if let Some(s) = sections.iter().find(|s| s.group.is_some()) {
            return Err(RasError::ObjectError(format!(
                "COFF writer has no COMDAT groups; {} is in a group",
                s.name
            )));
        }

        let section_index: HashMap<&str, usize> = sections
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.as_str(), i))
            .collect();
        let mut reloc_counts = vec![0usize; sections.len()];
        for r in relocations {
            let si = *section_index.get(r.section.as_str()).ok_or_else(|| {
                RasError::ObjectError(format!(
                    "relocation in section {} which has no COFF section",
                    r.section
                ))
            })?;
            reloc_counts[si] += 1;
            if reloc_counts[si] > u16::MAX as usize {
                return Err(RasError::ObjectError(format!(
                    "section {} has too many relocations for COFF",
                    r.section
                )));
            }
        }

        let mut strtab: Vec<u8> = vec![0u8; 4];
        let mut sym_bytes: Vec<u8> = Vec::new();
        let mut nsyms: u32 = 0;
        let mut section_symbol: HashMap<&str, u32> = HashMap::new();
        for (i, s) in sections.iter().enumerate() {
            section_symbol.insert(s.name.as_str(), nsyms);
            sym_bytes.extend_from_slice(&coff_symbol_entry(
                &mut strtab,
                &s.name,
                0,
                (i + 1) as i16,
                0,
                IMAGE_SYM_CLASS_STATIC,
                1,
            ));
            let mut aux = [0u8; COFF_SYMBOL_SIZE];
            aux[0..4].copy_from_slice(&(s.data.len() as u32).to_le_bytes());
            aux[4..6].copy_from_slice(&(reloc_counts[i] as u16).to_le_bytes());
            aux[12..14].copy_from_slice(&((i + 1) as u16).to_le_bytes());
            sym_bytes.extend_from_slice(&aux);
            nsyms += 2;
        }

        let referenced: std::collections::HashSet<&str> = relocations
            .iter()
            .filter(|r| !r.section_symbol)
            .map(|r| r.symbol.as_str())
            .collect();
        let weak_unique = req
            .symbols
            .iter()
            .find(|s| {
                s.global
                    && !s.attributes.weak
                    && !coff_is_file_local(&s.attributes)
                    && !matches!(s.section.as_str(), "" | "*COM*")
            })
            .map(|s| s.name.as_str());
        let mut symbol_index: HashMap<&str, u32> = HashMap::new();
        for sym in req.symbols {
            let wanted = referenced.contains(sym.name.as_str())
                || (!sym.attributes.internal && (sym.global || !sym.name.starts_with(".L")));
            if !wanted || symbol_index.contains_key(sym.name.as_str()) {
                continue;
            }
            let (section, value, ty) = match sym.section.as_str() {
                "" if sym.attributes.weak && referenced.contains(sym.name.as_str()) => {
                    let ty = if sym.attributes.symbol_type == 2 {
                        IMAGE_SYM_DTYPE_FUNCTION
                    } else {
                        0
                    };
                    (IMAGE_SYM_ABSOLUTE, 0, ty)
                }
                "" => continue,
                "*ABS*" => (IMAGE_SYM_ABSOLUTE, sym.value as u32, 0),
                "*COM*" => (
                    IMAGE_SYM_UNDEFINED,
                    sym.attributes.common.map_or(0, |c| c.0) as u32,
                    0,
                ),
                name => {
                    let si = *section_index.get(name).ok_or_else(|| {
                        RasError::ObjectError(format!(
                            "symbol {} is in unknown section {name}",
                            sym.name
                        ))
                    })?;
                    let ty = if sections[si].flags.exec {
                        IMAGE_SYM_DTYPE_FUNCTION
                    } else {
                        0
                    };
                    ((si + 1) as i16, sym.value as u32, ty)
                }
            };
            if sym.attributes.weak {
                let default_index = nsyms;
                sym_bytes.extend_from_slice(&coff_symbol_entry(
                    &mut strtab,
                    &coff_weak_default_name(&sym.name, weak_unique),
                    value,
                    section,
                    ty,
                    IMAGE_SYM_CLASS_EXTERNAL,
                    0,
                ));
                symbol_index.insert(sym.name.as_str(), nsyms + 1);
                sym_bytes.extend_from_slice(&coff_symbol_entry(
                    &mut strtab,
                    &sym.name,
                    0,
                    IMAGE_SYM_UNDEFINED,
                    ty,
                    IMAGE_SYM_CLASS_WEAK_EXTERNAL,
                    1,
                ));
                sym_bytes.extend_from_slice(&coff_weak_aux(default_index));
                nsyms += 3;
                continue;
            }
            let class = if coff_is_file_local(&sym.attributes) {
                IMAGE_SYM_CLASS_STATIC
            } else if sym.global || sym.section == "*COM*" {
                IMAGE_SYM_CLASS_EXTERNAL
            } else {
                IMAGE_SYM_CLASS_STATIC
            };
            symbol_index.insert(sym.name.as_str(), nsyms);
            sym_bytes.extend_from_slice(&coff_symbol_entry(
                &mut strtab,
                &sym.name,
                value,
                section,
                ty,
                class,
                0,
            ));
            nsyms += 1;
        }
        let declared_function: std::collections::HashSet<&str> = req
            .symbols
            .iter()
            .filter(|s| s.attributes.symbol_type == 2)
            .map(|s| s.name.as_str())
            .collect();
        for r in relocations.iter().filter(|r| !r.section_symbol) {
            if symbol_index.contains_key(r.symbol.as_str()) {
                continue;
            }
            let ty = if declared_function.contains(r.symbol.as_str()) {
                IMAGE_SYM_DTYPE_FUNCTION
            } else {
                0
            };
            symbol_index.insert(r.symbol.as_str(), nsyms);
            sym_bytes.extend_from_slice(&coff_symbol_entry(
                &mut strtab,
                &r.symbol,
                0,
                IMAGE_SYM_UNDEFINED,
                ty,
                IMAGE_SYM_CLASS_EXTERNAL,
                0,
            ));
            nsyms += 1;
        }

        let mut data: Vec<Vec<u8>> = sections.iter().map(|s| s.data.clone()).collect();
        let mut reloc_bytes: Vec<Vec<u8>> = vec![Vec::new(); sections.len()];
        for r in relocations {
            let si = section_index[r.section.as_str()];
            let section = &sections[si];
            let (ty, addend_place) = if x86 {
                let (ty, width) = coff_reloc_type(r.kind, &r.symbol)?;
                (ty, CoffAddend::Field(width))
            } else {
                coff_arm64_reloc_type(r.kind, &r.symbol, r.addend)?
            };
            let target = if r.section_symbol {
                *section_symbol.get(r.symbol.as_str()).ok_or_else(|| {
                    RasError::ObjectError(format!("Unknown relocation section: {}", r.symbol))
                })?
            } else {
                symbol_index[r.symbol.as_str()]
            };
            if coff_is_uninitialized(section) {
                return Err(RasError::ObjectError(format!(
                    "relocation inside uninitialized section {}",
                    section.name
                )));
            }
            let width = match addend_place {
                CoffAddend::Field(width) => width,
                CoffAddend::Instruction => 4,
            };
            let bytes = data[si]
                .get_mut(r.offset..r.offset + width)
                .ok_or_else(|| {
                    RasError::ObjectError(format!(
                        "relocation at offset {} exceeds section {}",
                        r.offset, section.name
                    ))
                })?;
            if let CoffAddend::Field(width) = addend_place {
                let implicit = match width {
                    2 => i64::from(i16::from_le_bytes(bytes.try_into().unwrap())),
                    4 => i64::from(i32::from_le_bytes(bytes.try_into().unwrap())),
                    _ => i64::from_le_bytes(bytes.try_into().unwrap()),
                };
                let adjust = if x86 && ty == IMAGE_REL_AMD64_REL32 {
                    4
                } else {
                    0
                };
                let value = implicit + r.addend + adjust;
                bytes.copy_from_slice(&value.to_le_bytes()[..width]);
            }
            reloc_bytes[si].extend_from_slice(&(r.offset as u32).to_le_bytes());
            reloc_bytes[si].extend_from_slice(&target.to_le_bytes());
            reloc_bytes[si].extend_from_slice(&ty.to_le_bytes());
        }

        let mut raw_ptr = vec![0u32; sections.len()];
        let mut raw_size = vec![0u32; sections.len()];
        let mut raw_end = vec![0u32; sections.len()];
        let mut reloc_ptr = vec![0u32; sections.len()];
        let mut cursor = 20 + COFF_SECTION_HEADER_SIZE * sections.len();
        for (i, s) in sections.iter().enumerate() {
            if coff_is_uninitialized(s) {
                raw_size[i] = s.data.len() as u32;
            } else if !s.data.is_empty() {
                let padded = s.data.len().div_ceil(COFF_RAW_ALIGN) * COFF_RAW_ALIGN;
                raw_ptr[i] = cursor as u32;
                raw_size[i] = s.data.len() as u32;
                raw_end[i] = (cursor + padded) as u32;
                cursor += padded;
            }
        }
        for i in 0..sections.len() {
            if reloc_counts[i] > 0 {
                reloc_ptr[i] = cursor as u32;
                cursor += COFF_RELOC_SIZE * reloc_counts[i];
            }
        }
        let sym_ptr = cursor as u32;

        let mut out: Vec<u8> = Vec::with_capacity(cursor + sym_bytes.len() + strtab.len());
        out.extend_from_slice(&machine.to_le_bytes());
        out.extend_from_slice(&(sections.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&sym_ptr.to_le_bytes());
        out.extend_from_slice(&nsyms.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());

        for (i, s) in sections.iter().enumerate() {
            out.extend_from_slice(&coff_section_name(&s.name, &mut strtab)?);
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&raw_size[i].to_le_bytes());
            out.extend_from_slice(&raw_ptr[i].to_le_bytes());
            out.extend_from_slice(&reloc_ptr[i].to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&(reloc_counts[i] as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&coff_section_characteristics(s)?.to_le_bytes());
        }

        for (i, _) in sections.iter().enumerate() {
            if raw_ptr[i] != 0 {
                out.extend_from_slice(&data[i]);
                out.resize(raw_end[i] as usize, 0);
            }
        }
        for bytes in &reloc_bytes {
            out.extend_from_slice(bytes);
        }
        out.extend_from_slice(&sym_bytes);
        let strtab_size = strtab.len() as u32;
        strtab[0..4].copy_from_slice(&strtab_size.to_le_bytes());
        out.extend_from_slice(&strtab);

        Ok(out)
    }
}

/// Returns the object writer for `target_os` (ELF, Mach-O, or COFF).
pub fn object_writer_for_os(
    target_os: TargetOperatingSystem,
) -> Result<Box<dyn ObjectWriter>, RasError> {
    match target_os {
        TargetOperatingSystem::Linux
        | TargetOperatingSystem::FreeBSD
        | TargetOperatingSystem::OpenBSD
        | TargetOperatingSystem::NetBSD
        | TargetOperatingSystem::DragonFly
        | TargetOperatingSystem::Redox
        | TargetOperatingSystem::Orbis
        | TargetOperatingSystem::Prospero => Ok(Box::new(ElfWriter::new())),
        TargetOperatingSystem::MacOS => Ok(Box::new(MachOWriter::new())),
        TargetOperatingSystem::Windows => Ok(Box::new(CoffWriter::new())),
        _ => Err(RasError::UnsupportedTarget(format!(
            "No object file format is mapped for operating system {:?}",
            target_os
        ))),
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::TempDir;

    use super::*;

    #[test]
    fn test_amd64_relocation_types_are_the_values_the_pe_spec_gives_them() {
        assert_eq!(IMAGE_REL_AMD64_ADDR64, 0x0001);
        assert_eq!(IMAGE_REL_AMD64_ADDR32, 0x0002);
        assert_eq!(IMAGE_REL_AMD64_REL32, 0x0004);
        assert_eq!(IMAGE_REL_AMD64_SECREL, 0x000B);
    }

    #[test]
    fn test_arm64_relocation_types_are_the_values_the_pe_spec_gives_them() {
        assert_eq!(IMAGE_REL_ARM64_ADDR32, 0x0001);
        assert_eq!(IMAGE_REL_ARM64_BRANCH26, 0x0003);
        assert_eq!(IMAGE_REL_ARM64_PAGEBASE_REL21, 0x0004);
        assert_eq!(IMAGE_REL_ARM64_PAGEOFFSET_12A, 0x0006);
        assert_eq!(IMAGE_REL_ARM64_PAGEOFFSET_12L, 0x0007);
        assert_eq!(IMAGE_REL_ARM64_ADDR64, 0x000E);
        assert_eq!(IMAGE_REL_ARM64_BRANCH19, 0x000F);
        assert_eq!(IMAGE_REL_ARM64_BRANCH14, 0x0010);
    }

    #[test]
    fn test_i386_relocation_types_are_the_values_the_psabi_gives_them() {
        assert_eq!(R_386_32, 1);
        assert_eq!(R_386_PC32, 2);
        assert_eq!(R_386_PLT32, 4);
        assert_eq!(R_386_16, 20);
        assert_eq!(R_386_8, 22);
    }

    #[test]
    fn test_arm_relocation_types_are_the_values_the_psabi_gives_them() {
        assert_eq!(R_ARM_ABS32, 2);
        assert_eq!(R_ARM_REL32, 3);
        assert_eq!(R_ARM_CALL, 28);
        assert_eq!(R_ARM_JUMP24, 29);
        assert_eq!(R_ARM_MOVW_ABS_NC, 43);
        assert_eq!(R_ARM_MOVT_ABS, 44);
    }

    #[test]
    fn test_elf64_x86_64_write() {
        let mut w = ElfWriter::new();
        let code = [0xc3u8];
        let sections = vec![Section {
            group: None,
            name: ".text".to_string(),
            data: code.to_vec(),
            alignment: 1,
            macho: None,
            flags: SectionFlags {
                alloc: true,
                exec: true,
                write: false,
                tls: false,
            },
        }];
        let symbols: Vec<ObjectSymbol> = vec![];
        let tmp = TempDir::new("elf_test_x86_64");
        let path = tmp.path().join("elf_test_x86_64.o");
        let opts = ObjectWriteOptions::default();
        let result = w.write_object_file(
            &path,
            &ObjectWriteRequest {
                code: &code,
                sections: &sections,
                symbols: &symbols,
                relocations: &[],
                target_arch: TargetArchitecture::X86_64,
                target_os: TargetOperatingSystem::Linux,
                opts: &opts,
            },
        );
        result.expect("ELF write should succeed");
    }

    #[test]
    fn test_elf64_aarch64_write() {
        let mut w = ElfWriter::new();
        let code = [0xc0, 0x03, 0x5f, 0xd6];
        let sections = vec![Section {
            group: None,
            name: ".text".to_string(),
            data: code.to_vec(),
            alignment: 1,
            macho: None,
            flags: SectionFlags {
                alloc: true,
                exec: true,
                write: false,
                tls: false,
            },
        }];
        let symbols: Vec<ObjectSymbol> = vec![];
        let tmp = TempDir::new("elf_test_aarch64");
        let path = tmp.path().join("elf_test_aarch64.o");
        let opts = ObjectWriteOptions::default();
        let result = w.write_object_file(
            &path,
            &ObjectWriteRequest {
                code: &code,
                sections: &sections,
                symbols: &symbols,
                relocations: &[],
                target_arch: TargetArchitecture::Aarch64,
                target_os: TargetOperatingSystem::Linux,
                opts: &opts,
            },
        );
        result.expect("ELF write should succeed");
    }

    #[test]
    fn object_writer_for_os_elf_and_coff_paths() {
        assert!(object_writer_for_os(TargetOperatingSystem::Linux).is_ok());
        assert!(object_writer_for_os(TargetOperatingSystem::MacOS).is_ok());
        assert!(object_writer_for_os(TargetOperatingSystem::Windows).is_ok());
    }

    #[test]
    fn test_elf64_has_valid_magic() {
        let mut w = ElfWriter::new();
        let code = [0xc3u8];
        let tmp = TempDir::new("elf_magic_test");
        let path = tmp.path().join("elf_magic_test.o");
        let opts = ObjectWriteOptions::default();
        w.write_object_file(
            &path,
            &ObjectWriteRequest {
                code: &code,
                sections: &[],
                symbols: &[],
                relocations: &[],
                target_arch: TargetArchitecture::X86_64,
                target_os: TargetOperatingSystem::Linux,
                opts: &opts,
            },
        )
        .expect("write");
        let buf = std::fs::read(&path).expect("read");
        assert!(buf.len() >= 64);
        assert_eq!(&buf[0..4], &[0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn tls_section_symbols_are_stt_tls() {
        let tls_section = |name: &str, nobits: bool| Section {
            group: None,
            name: name.into(),
            flags: SectionFlags {
                alloc: true,
                exec: false,
                write: true,
                tls: true,
            },
            data: if nobits { vec![0; 8] } else { vec![1; 8] },
            alignment: 8,
            macho: None,
        };
        let symbol = |name: &str, global: bool, section: &str| ObjectSymbol {
            name: name.into(),
            global,
            section: section.into(),
            value: 0,
            attributes: Default::default(),
        };
        let opts = ObjectWriteOptions::default();
        let buf = ElfWriter::new()
            .object_bytes(&ObjectWriteRequest {
                code: &[],
                sections: &[tls_section(".tdata", false), tls_section(".tbss", true)],
                symbols: &[
                    symbol("tls_init", true, ".tdata"),
                    symbol("tls_zero", false, ".tbss"),
                ],
                relocations: &[],
                target_arch: TargetArchitecture::X86_64,
                target_os: TargetOperatingSystem::Linux,
                opts: &opts,
            })
            .expect("write");
        let u64_at = |at: usize| u64::from_le_bytes(buf[at..at + 8].try_into().unwrap());
        let u32_at = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
        let shoff = u64_at(0x28) as usize;
        let shnum = u16::from_le_bytes([buf[0x3c], buf[0x3d]]) as usize;
        let header = |index: usize| shoff + index * ELF64_SECTION_HEADER_SIZE;
        let symtab = (0..shnum)
            .find(|&i| u32_at(header(i) + 4) == SHT_SYMTAB)
            .expect(".symtab section");
        let strtab = header(u32_at(header(symtab) + 0x28) as usize);
        let strings = u64_at(strtab + 0x18) as usize;
        let sym_base = u64_at(header(symtab) + 0x18) as usize;
        let sym_count = u64_at(header(symtab) + 0x20) as usize / ELF64_SYM_SIZE as usize;
        let type_of = |wanted: &str| {
            (0..sym_count)
                .map(|i| sym_base + i * ELF64_SYM_SIZE as usize)
                .find(|&entry| {
                    let start = strings + u32_at(entry) as usize;
                    let end = start + buf[start..].iter().position(|b| *b == 0).unwrap();
                    &buf[start..end] == wanted.as_bytes()
                })
                .map(|entry| buf[entry + 4] & 0xf)
                .unwrap_or_else(|| panic!("symbol {wanted} missing"))
        };
        assert_eq!(type_of("tls_init"), 6, "STT_TLS in .tdata");
        assert_eq!(type_of("tls_zero"), 6, "STT_TLS in .tbss");
    }

    #[test]
    fn section_symbol_relocations_emit_rela_for_debug_sections() {
        let text = Section {
            group: None,
            name: ".text".into(),
            flags: SectionFlags {
                alloc: true,
                exec: true,
                write: false,
                tls: false,
            },
            data: vec![0x90; 4],
            alignment: 1,
            macho: None,
        };
        let debug_line = Section {
            group: None,
            name: ".debug_line".into(),
            flags: SectionFlags {
                alloc: false,
                exec: false,
                write: false,
                tls: false,
            },
            data: vec![0; 8],
            alignment: 1,
            macho: None,
        };
        let relocations = [ExternalReloc {
            offset: 0,
            symbol: ".text".into(),
            section: ".debug_line".into(),
            kind: 1,
            addend: 2,
            trailing: 0,
            branch: false,
            section_symbol: true,
        }];
        let code = vec![0x90u8; 4];
        let opts = ObjectWriteOptions::default();
        let buf = ElfWriter::new()
            .object_bytes(&ObjectWriteRequest {
                code: &code,
                sections: &[text, debug_line],
                symbols: &[],
                relocations: &relocations,
                target_arch: TargetArchitecture::X86_64,
                target_os: TargetOperatingSystem::Linux,
                opts: &opts,
            })
            .expect("write");
        let u64_at = |at: usize| u64::from_le_bytes(buf[at..at + 8].try_into().unwrap());
        let u32_at = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
        let shoff = u64_at(0x28) as usize;
        let shnum = u16::from_le_bytes([buf[0x3c], buf[0x3d]]) as usize;
        let header = |index: usize| shoff + index * ELF64_SECTION_HEADER_SIZE;
        let shstrtab = header(usize::from(u16::from_le_bytes([buf[0x3e], buf[0x3f]])));
        let names = u64_at(shstrtab + 0x18) as usize;
        let name_of = |index: usize| {
            let start = names + u32_at(header(index)) as usize;
            let end = start + buf[start..].iter().position(|b| *b == 0).unwrap();
            std::str::from_utf8(&buf[start..end]).unwrap().to_string()
        };
        let rela = (0..shnum)
            .find(|&i| name_of(i) == ".rela.debug_line")
            .expect(".rela.debug_line section");
        assert_eq!(u32_at(header(rela) + 4), SHT_RELA);
        let entry = u64_at(header(rela) + 0x18) as usize;
        assert_eq!(u64_at(entry + 8) & 0xffff_ffff, 1, "R_X86_64_64");
        assert_eq!(u64_at(entry + 8) >> 32, 1, "section symbol for .text");
        assert_eq!(u64_at(entry + 16), 2, "addend");
    }

    fn macho_text_section(data: Vec<u8>) -> Section {
        Section {
            group: None,
            name: ".text".into(),
            flags: SectionFlags {
                alloc: true,
                exec: true,
                write: false,
                tls: false,
            },
            data,
            alignment: 16,
            macho: None,
        }
    }

    fn macho_object(
        sections: &[Section],
        symbols: &[ObjectSymbol],
        relocations: &[ExternalReloc],
    ) -> Result<Vec<u8>, RasError> {
        let opts = ObjectWriteOptions::default();
        macho_object_bytes(&ObjectWriteRequest {
            code: &[],
            sections,
            symbols,
            relocations,
            target_arch: TargetArchitecture::X86_64,
            target_os: TargetOperatingSystem::MacOS,
            opts: &opts,
        })
    }

    fn le32(buf: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(buf[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn macho_x86_64_header_declares_relocatable_object() {
        let sections = [macho_text_section(vec![0x31, 0xc0, 0xc3])];
        let symbols = [ObjectSymbol {
            attributes: Default::default(),
            name: "_leaf".into(),
            global: true,
            section: ".text".into(),
            value: 0,
        }];
        let buf = macho_object(&sections, &symbols, &[]).expect("write");
        assert_eq!(le32(&buf, 0), 0xfeed_facf);
        assert_eq!(le32(&buf, 4), 0x0100_0007);
        assert_eq!(le32(&buf, 12), 1, "MH_OBJECT");
        assert_eq!(
            le32(&buf, 16),
            4,
            "segment, build version, symtab, dysymtab"
        );
    }

    #[test]
    fn macho_call_to_undefined_symbol_is_extern_branch() {
        let sections = [macho_text_section(vec![0xe8, 0, 0, 0, 0, 0xc3])];
        let symbols = [ObjectSymbol {
            attributes: Default::default(),
            name: "_caller".into(),
            global: true,
            section: ".text".into(),
            value: 0,
        }];
        let relocations = [ExternalReloc {
            offset: 1,
            symbol: "_puts".into(),
            section: ".text".into(),
            kind: 4,
            addend: -4,
            trailing: 0,
            branch: true,
            section_symbol: false,
        }];
        let buf = macho_object(&sections, &symbols, &relocations).expect("write");
        let text_reloc = le32(&buf, 32 + 72 + 56) as usize;
        let info = le32(&buf, text_reloc + 4);
        assert_eq!(le32(&buf, text_reloc), 1, "r_address is the field offset");
        assert_eq!(info & 0x00ff_ffff, 1, "undefined _puts follows _caller");
        assert_eq!(info >> 24 & 1, 1, "pc-relative");
        assert_eq!(info >> 27 & 1, 1, "extern");
        assert_eq!(info >> 28, X86_64_RELOC_BRANCH);
        assert!(
            buf.windows(5).any(|w| w == b"_puts"),
            "name written verbatim"
        );
    }

    #[test]
    fn macho_rejects_sections_and_relocations_it_cannot_express() {
        let debug = Section {
            name: ".debug_info".into(),
            ..macho_text_section(vec![0])
        };
        let err = macho_object(&[debug], &[], &[]).expect_err("unmapped section");
        assert!(
            err.to_string().contains("no section for .debug_info"),
            "{err}"
        );

        let bss = Section {
            name: ".bss".into(),
            flags: SectionFlags {
                alloc: true,
                exec: false,
                write: true,
                tls: false,
            },
            ..macho_text_section(vec![1, 0])
        };
        let err = macho_object(&[macho_text_section(vec![]), bss], &[], &[]).expect_err("bss");
        assert!(err.to_string().contains("zero-fill"), "{err}");

        let pc64 = ExternalReloc {
            offset: 0,
            symbol: "_f".into(),
            section: ".text".into(),
            kind: 24,
            addend: 0,
            trailing: 0,
            branch: false,
            section_symbol: false,
        };
        let err = macho_object(&[macho_text_section(vec![0; 8])], &[], &[pc64]).expect_err("pc64");
        assert!(err.to_string().contains("kind 24"), "{err}");
    }

    fn macho_arm64_object(
        sections: &[Section],
        symbols: &[ObjectSymbol],
        relocations: &[ExternalReloc],
    ) -> Result<Vec<u8>, RasError> {
        let opts = ObjectWriteOptions::default();
        macho_object_bytes(&ObjectWriteRequest {
            code: &[],
            sections,
            symbols,
            relocations,
            target_arch: TargetArchitecture::Aarch64,
            target_os: TargetOperatingSystem::MacOS,
            opts: &opts,
        })
    }

    fn arm64_section(name: &str, exec: bool, data: Vec<u8>) -> Section {
        Section {
            group: None,
            name: name.into(),
            flags: SectionFlags {
                alloc: true,
                exec,
                write: false,
                tls: false,
            },
            data,
            alignment: 4,
            macho: None,
        }
    }

    fn arm64_symbol(name: &str, global: bool, section: &str, value: u64) -> ObjectSymbol {
        ObjectSymbol {
            attributes: Default::default(),
            name: name.into(),
            global,
            section: section.into(),
            value,
        }
    }

    fn macho_symbol_names(buf: &[u8]) -> Vec<String> {
        macho_nlists(buf)
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    fn macho_nlists(buf: &[u8]) -> Vec<(String, u8)> {
        let symtab = (0..le32(buf, 16))
            .scan(32usize, |off, _| {
                let at = *off;
                *off += le32(buf, at + 4) as usize;
                Some(at)
            })
            .find(|&at| le32(buf, at) == LC_SYMTAB)
            .expect("LC_SYMTAB");
        let symoff = le32(buf, symtab + 8) as usize;
        let nsyms = le32(buf, symtab + 12) as usize;
        let stroff = le32(buf, symtab + 16) as usize;
        (0..nsyms)
            .map(|i| {
                let start = stroff + le32(buf, symoff + 16 * i) as usize;
                let len = buf[start..].iter().position(|&b| b == 0).expect("nul");
                let name = String::from_utf8_lossy(&buf[start..start + len]).into_owned();
                (name, buf[symoff + 16 * i + 4])
            })
            .collect()
    }

    fn macho_section_header(section: usize) -> usize {
        32 + 72 + 80 * section
    }

    fn macho_section_bytes(buf: &[u8], section: usize) -> &[u8] {
        let base = macho_section_header(section);
        let offset = le32(buf, base + 48) as usize;
        let size = le32(buf, base + 40) as usize;
        &buf[offset..offset + size]
    }

    fn macho_reloc_entries(buf: &[u8], section: usize) -> Vec<(u32, u32)> {
        let base = macho_section_header(section);
        let reloff = le32(buf, base + 56) as usize;
        let nreloc = le32(buf, base + 60) as usize;
        (0..nreloc)
            .map(|i| (le32(buf, reloff + 8 * i), le32(buf, reloff + 8 * i + 4)))
            .collect()
    }

    #[test]
    fn macho_arm64_page_symbol_operands_relocate_adrp_and_add() {
        let source = ".text\n.globl _f\n_f:\nadrp x0, _fmt@PAGE\nadd x0, x0, _fmt@PAGEOFF\nret\n";
        let mut ras = crate::Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS)
            .expect("ras");
        let buf = ras.assemble_to_bytes(source).expect("assemble");

        let text = macho_section_bytes(&buf, 0);
        assert_eq!(
            &text[0..4],
            &0x9000_0000u32.to_le_bytes(),
            "adrp keeps a zero immediate"
        );
        assert_eq!(
            &text[4..8],
            &0x9100_0000u32.to_le_bytes(),
            "add keeps a zero immediate"
        );

        let relocs: Vec<(u32, u32)> = macho_reloc_entries(&buf, 0)
            .into_iter()
            .map(|(addr, info)| {
                assert_eq!(info >> 27 & 1, 1, "undefined _fmt is extern");
                (addr, info >> 28)
            })
            .collect();
        assert_eq!(
            relocs,
            [(0, ARM64_RELOC_PAGE21), (4, ARM64_RELOC_PAGEOFF12)]
        );
    }

    #[test]
    fn macho_weak_directives_set_the_weak_definition_bit() {
        for directive in [".weak", ".weak_definition"] {
            let source = format!(".text\n.globl _f\n{directive} _f\n_f:\nret\n");
            let mut ras =
                crate::Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS)
                    .expect("ras");
            let buf = ras.assemble_to_bytes(&source).expect("assemble");
            let symtab = (0..le32(&buf, 16))
                .scan(32usize, |off, _| {
                    let at = *off;
                    *off += le32(&buf, at + 4) as usize;
                    Some(at)
                })
                .find(|&at| le32(&buf, at) == LC_SYMTAB)
                .expect("LC_SYMTAB");
            let symoff = le32(&buf, symtab + 8) as usize;
            let nsyms = le32(&buf, symtab + 12) as usize;
            let stroff = le32(&buf, symtab + 16) as usize;
            let weak = (0..nsyms)
                .find(|i| {
                    let start = stroff + le32(&buf, symoff + 16 * i) as usize;
                    let end = buf[start..].iter().position(|&b| b == 0).unwrap() + start;
                    &buf[start..end] == b"_f"
                })
                .map(|i| u16::from_le_bytes([buf[symoff + 16 * i + 6], buf[symoff + 16 * i + 7]]))
                .expect("_f in symbol table");
            assert_eq!(weak & N_WEAK_DEF, N_WEAK_DEF, "{directive}");
        }
    }

    #[test]
    fn macho_private_extern_sets_private_external_nlist_bit() {
        let source = ".text\n.globl _f\n.private_extern _f\n_f:\nret\n";
        let mut ras = crate::Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS)
            .expect("ras");
        let buf = ras.assemble_to_bytes(source).expect("assemble");
        let (_, n_type) = macho_nlists(&buf)
            .into_iter()
            .find(|(name, _)| name == "_f")
            .expect("_f in symbol table");
        assert_eq!(n_type & (N_EXT | N_PEXT), N_EXT | N_PEXT);
    }

    #[test]
    fn macho_arm64_header_declares_arm64_object() {
        let sections = [arm64_section(
            ".text",
            true,
            vec![0x40, 0x05, 0x80, 0x52, 0xc0, 0x03, 0x5f, 0xd6],
        )];
        let symbols = [arm64_symbol("_leaf", true, ".text", 0)];
        let buf = macho_arm64_object(&sections, &symbols, &[]).expect("write");
        assert_eq!(le32(&buf, 4), 0x0100_000c, "CPU_TYPE_ARM64");
        assert_eq!(le32(&buf, 8), 0, "CPU_SUBTYPE_ARM64_ALL");
        assert_eq!(le32(&buf, 12), 1, "MH_OBJECT");
    }

    #[test]
    fn macho_arm64_section_label_binds_to_ltmp_with_addend() {
        let sections = [
            arm64_section(
                ".text",
                true,
                vec![0, 0, 0, 0x90, 0, 0, 0, 0x91, 0xc0, 0x03, 0x5f, 0xd6],
            ),
            arm64_section(".rodata", false, b"x\0yz\0".to_vec()),
        ];
        let symbols = [
            arm64_symbol("_f", true, ".text", 0),
            arm64_symbol("L.str", false, ".rodata", 2),
        ];
        let relocations = [
            ExternalReloc {
                offset: 0,
                symbol: "L.str".into(),
                section: ".text".into(),
                kind: 275,
                addend: 0,
                trailing: 0,
                branch: false,
                section_symbol: false,
            },
            ExternalReloc {
                offset: 4,
                symbol: "L.str".into(),
                section: ".text".into(),
                kind: 277,
                addend: 0,
                trailing: 0,
                branch: false,
                section_symbol: false,
            },
        ];
        let buf = macho_arm64_object(&sections, &symbols, &relocations).expect("write");
        let names = macho_symbol_names(&buf);
        assert_eq!(
            names[..2],
            ["ltmp0", "ltmp1"],
            "one section symbol per section"
        );
        assert!(
            !names.iter().any(|n| n == "L.str"),
            "private label stays out"
        );

        let text = macho_section_bytes(&buf, 0);
        assert_eq!(
            &text[0..4],
            &0x9000_0000u32.to_le_bytes(),
            "adrp keeps a zero immediate"
        );
        assert_eq!(
            &text[4..8],
            &0x9100_0000u32.to_le_bytes(),
            "add keeps a zero immediate"
        );

        let entries = macho_reloc_entries(&buf, 0);
        assert_eq!(entries.len(), 4, "each page reference adds an ADDEND");
        let (addr, info) = entries[0];
        assert_eq!((addr, info >> 28), (0, ARM64_RELOC_ADDEND));
        assert_eq!(
            info & 0x00ff_ffff,
            2,
            "addend is the offset of L.str in .rodata"
        );
        let (addr, info) = entries[1];
        assert_eq!((addr, info >> 28), (0, ARM64_RELOC_PAGE21));
        assert_eq!(
            info & 0x00ff_ffff,
            1,
            "bound to ltmp1, the .rodata section symbol"
        );
        assert_eq!(info >> 24 & 1, 1, "pc-relative");
        assert_eq!(info >> 27 & 1, 1, "extern");
        let (addr, info) = entries[3];
        assert_eq!((addr, info >> 28), (4, ARM64_RELOC_PAGEOFF12));
        assert_eq!(info & 0x00ff_ffff, 1);
    }

    #[test]
    fn macho_arm64_call_to_undefined_symbol_is_extern_branch26() {
        let sections = [arm64_section(
            ".text",
            true,
            vec![0, 0, 0, 0x94, 0xc0, 0x03, 0x5f, 0xd6],
        )];
        let symbols = [arm64_symbol("_caller", true, ".text", 0)];
        let relocations = [ExternalReloc {
            offset: 0,
            symbol: "_puts".into(),
            section: ".text".into(),
            kind: 283,
            addend: 0,
            trailing: 0,
            branch: true,
            section_symbol: false,
        }];
        let buf = macho_arm64_object(&sections, &symbols, &relocations).expect("write");
        assert_eq!(
            &macho_section_bytes(&buf, 0)[0..4],
            &[0, 0, 0, 0x94],
            "bl with zero immediate"
        );
        let entries = macho_reloc_entries(&buf, 0);
        assert_eq!(entries.len(), 1, "no ADDEND for a bare reference");
        let (addr, info) = entries[0];
        assert_eq!(addr, 0);
        assert_eq!(info >> 28, ARM64_RELOC_BRANCH26);
        assert_eq!(info >> 27 & 1, 1, "extern");
        assert_eq!(info >> 25 & 3, 2, "four-byte field");
        let names = macho_symbol_names(&buf);
        assert_eq!(names[(info & 0x00ff_ffff) as usize], "_puts");
    }

    #[test]
    fn macho_arm64_rejects_relocation_kinds_without_equivalent() {
        for kind in [280, 260, 261] {
            let reloc = ExternalReloc {
                offset: 0,
                symbol: "_f".into(),
                section: ".text".into(),
                kind,
                addend: 0,
                trailing: 0,
                branch: false,
                section_symbol: false,
            };
            let sections = [arm64_section(".text", true, vec![0; 8])];
            let symbols = [arm64_symbol("_f", true, ".text", 0)];
            let err = macho_arm64_object(&sections, &symbols, &[reloc])
                .expect_err("kind without a Mach-O equivalent");
            assert!(err.to_string().contains(&format!("kind {kind}")), "{err}");
        }
    }

    #[test]
    fn test_coff_amd64_write() {
        let mut w = CoffWriter::new();
        let code = [0xc3u8];
        let tmp = TempDir::new("coff_test");
        let path = tmp.path().join("coff_test.obj");
        let opts = ObjectWriteOptions::default();
        w.write_object_file(
            &path,
            &ObjectWriteRequest {
                code: &code,
                sections: &[],
                symbols: &[],
                relocations: &[],
                target_arch: TargetArchitecture::X86_64,
                target_os: TargetOperatingSystem::Windows,
                opts: &opts,
            },
        )
        .expect("coff write");
        let buf = std::fs::read(&path).expect("read");
        assert_eq!(&buf[0..2], &IMAGE_FILE_MACHINE_AMD64.to_le_bytes());
    }

    struct CoffSection {
        name: String,
        raw_ptr: u32,
        raw_size: u32,
        characteristics: u32,
        data: Vec<u8>,
        relocs: Vec<(u32, u32, u16)>,
    }

    struct CoffSymbol {
        name: String,
        value: u32,
        section: i16,
        typ: u16,
        class: u8,
    }

    struct CoffFile {
        sections: Vec<CoffSection>,
        symbols: Vec<CoffSymbol>,
    }

    fn coff_parse(buf: &[u8]) -> CoffFile {
        let u16_at = |at: usize| u16::from_le_bytes(buf[at..at + 2].try_into().unwrap());
        let u32_at = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
        let nsections = u16_at(2) as usize;
        let sym_ptr = u32_at(8) as usize;
        let nsyms = u32_at(12) as usize;
        let strtab = sym_ptr + nsyms * COFF_SYMBOL_SIZE;
        let name_at = |raw: &[u8]| {
            if raw[..4] == [0, 0, 0, 0] {
                let off = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
                let end = buf[strtab + off..].iter().position(|&b| b == 0).unwrap();
                String::from_utf8(buf[strtab + off..strtab + off + end].to_vec()).unwrap()
            } else {
                String::from_utf8(raw.iter().copied().take_while(|&b| b != 0).collect()).unwrap()
            }
        };
        let sections = (0..nsections)
            .map(|i| {
                let h = 20 + i * COFF_SECTION_HEADER_SIZE;
                let raw_size = u32_at(h + 16);
                let raw_ptr = u32_at(h + 20);
                let reloc_ptr = u32_at(h + 24) as usize;
                let nrelocs = u16_at(h + 32) as usize;
                let relocs = (0..nrelocs)
                    .map(|r| {
                        let e = reloc_ptr + r * COFF_RELOC_SIZE;
                        (u32_at(e), u32_at(e + 4), u16_at(e + 8))
                    })
                    .collect();
                let data = if raw_ptr == 0 {
                    Vec::new()
                } else {
                    buf[raw_ptr as usize..(raw_ptr + raw_size) as usize].to_vec()
                };
                CoffSection {
                    name: if buf[h] == b'/' {
                        let digits: String = buf[h + 1..h + 8]
                            .iter()
                            .take_while(|b| b.is_ascii_digit())
                            .map(|&b| b as char)
                            .collect();
                        let off = strtab + digits.parse::<usize>().unwrap();
                        let end = buf[off..].iter().position(|&b| b == 0).unwrap();
                        String::from_utf8(buf[off..off + end].to_vec()).unwrap()
                    } else {
                        name_at(&buf[h..h + 8])
                    },
                    raw_ptr,
                    raw_size,
                    characteristics: u32_at(h + 36),
                    data,
                    relocs,
                }
            })
            .collect();
        let mut symbols = Vec::new();
        let mut i = 0;
        while i < nsyms {
            let e = sym_ptr + i * COFF_SYMBOL_SIZE;
            let aux = buf[e + 17] as usize;
            symbols.push(CoffSymbol {
                name: name_at(&buf[e..e + 8]),
                value: u32_at(e + 8),
                section: u16_at(e + 12) as i16,
                typ: u16_at(e + 14),
                class: buf[e + 16],
            });
            for _ in 0..aux {
                symbols.push(CoffSymbol {
                    name: String::new(),
                    value: 0,
                    section: 0,
                    typ: 0,
                    class: 0,
                });
            }
            i += 1 + aux;
        }
        CoffFile { sections, symbols }
    }

    fn coff_section(name: &str, exec: bool, write: bool, data: Vec<u8>) -> Section {
        Section {
            group: None,
            name: name.into(),
            flags: SectionFlags {
                alloc: true,
                exec,
                write,
                tls: false,
            },
            data,
            alignment: 16,
            macho: None,
        }
    }

    fn coff_symbol(name: &str, global: bool, section: &str, value: u64) -> ObjectSymbol {
        ObjectSymbol {
            attributes: Default::default(),
            name: name.into(),
            global,
            section: section.into(),
            value,
        }
    }

    fn coff_reloc(
        section: &str,
        offset: usize,
        symbol: &str,
        kind: u32,
        addend: i64,
    ) -> ExternalReloc {
        ExternalReloc {
            offset,
            symbol: symbol.into(),
            section: section.into(),
            kind,
            addend,
            trailing: 0,
            branch: false,
            section_symbol: false,
        }
    }

    fn coff_object(
        sections: &[Section],
        symbols: &[ObjectSymbol],
        relocations: &[ExternalReloc],
    ) -> Vec<u8> {
        let opts = ObjectWriteOptions::default();
        CoffWriter::new()
            .object_bytes(&ObjectWriteRequest {
                code: &sections[0].data,
                sections,
                symbols,
                relocations,
                target_arch: TargetArchitecture::X86_64,
                target_os: TargetOperatingSystem::Windows,
                opts: &opts,
            })
            .expect("coff write")
    }

    #[test]
    fn coff_non_text_sections_hold_their_symbols_and_relocations() {
        let lea = [0x48, 0x8d, 0x05, 0, 0, 0, 0];
        let mut text = Vec::new();
        text.extend_from_slice(&lea);
        text.extend_from_slice(&lea);
        text.push(0xc3);
        let sections = [
            coff_section(".text", true, false, text),
            coff_section(".data", false, true, b"hi\0".to_vec()),
            coff_section(".rodata.str1", false, false, b"ro\0".to_vec()),
            coff_section(".bss", false, true, vec![0; 16]),
        ];
        let symbols = [
            coff_symbol("main", true, ".text", 0),
            coff_symbol("msg", false, ".data", 0),
            coff_symbol("ro", false, ".rodata.str1", 0),
            coff_symbol("buf", false, ".bss", 8),
        ];
        let relocations = [
            coff_reloc(".text", 3, "msg", 2, -4),
            coff_reloc(".text", 10, "ro", 2, -4),
        ];
        let buf = coff_object(&sections, &symbols, &relocations);
        let obj = coff_parse(&buf);

        let names: Vec<&str> = obj.sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, [".text", ".data", ".rodata.str1", ".bss"]);
        let index_of =
            |name: &str| obj.sections.iter().position(|s| s.name == name).unwrap() as i16 + 1;
        let symbol = |name: &str| {
            obj.symbols
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("symbol {name} missing"))
        };

        let data = &obj.sections[1];
        assert!(data.characteristics & IMAGE_SCN_CNT_INITIALIZED_DATA != 0);
        assert!(data.characteristics & IMAGE_SCN_MEM_WRITE != 0);
        assert_eq!(&data.data[..3], b"hi\0");
        let msg = symbol("msg");
        assert_eq!(msg.section, index_of(".data"));
        assert_eq!(msg.class, IMAGE_SYM_CLASS_STATIC);
        assert_eq!(msg.value, 0);

        let rodata = &obj.sections[2];
        assert!(rodata.characteristics & IMAGE_SCN_CNT_INITIALIZED_DATA != 0);
        assert!(rodata.characteristics & IMAGE_SCN_MEM_WRITE == 0);
        assert_eq!(symbol("ro").section, index_of(".rodata.str1"));

        let bss = &obj.sections[3];
        assert!(bss.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA != 0);
        assert_eq!(bss.raw_ptr, 0);
        assert_eq!(bss.raw_size, 16);
        let buf_sym = symbol("buf");
        assert_eq!(buf_sym.section, index_of(".bss"));
        assert_eq!(buf_sym.value, 8);

        let main = symbol("main");
        assert_eq!(main.section, index_of(".text"));
        assert_eq!(main.class, IMAGE_SYM_CLASS_EXTERNAL);

        let text = &obj.sections[0];
        assert_eq!(text.relocs.len(), 2);
        let (offset, symidx, ty) = text.relocs[0];
        assert_eq!((offset, ty), (3, IMAGE_REL_AMD64_REL32));
        assert_eq!(obj.symbols[symidx as usize].name, "msg");
        let (offset, symidx, ty) = text.relocs[1];
        assert_eq!((offset, ty), (10, IMAGE_REL_AMD64_REL32));
        assert_eq!(obj.symbols[symidx as usize].name, "ro");
        assert_eq!(&text.data[3..7], &[0, 0, 0, 0]);
        assert_eq!(&text.data[10..14], &[0, 0, 0, 0]);
    }

    #[test]
    fn coff_undefined_function_symbols_carry_the_function_type() {
        let sections = [coff_section(".text", true, false, vec![0; 16])];
        let mut declared = coff_symbol("printf", true, "", 0);
        declared.attributes.symbol_type = 2;
        let symbols = [coff_symbol("main", true, ".text", 0), declared];
        let relocations = [
            coff_reloc(".text", 3, "printf", 4, 0),
            coff_reloc(".text", 10, "datasym", 4, 0),
        ];
        let buf = coff_object(&sections, &symbols, &relocations);
        let obj = coff_parse(&buf);

        let kind = |want: &str| {
            obj.symbols
                .iter()
                .find(|s| s.name == want)
                .unwrap_or_else(|| panic!("{want} missing"))
                .typ
        };
        assert_eq!(kind("printf") & 0xF0, IMAGE_SYM_DTYPE_FUNCTION);
        assert_eq!(kind("datasym") & 0xF0, 0);
    }

    #[test]
    fn coff_absolute_data_pointer_relocation_is_addr64() {
        let sections = [
            coff_section(".text", true, false, vec![0xc3]),
            coff_section(".data", false, true, vec![0; 8]),
        ];
        let symbols = [coff_symbol("main", true, ".text", 0)];
        let relocations = [coff_reloc(".data", 0, "main", 1, 8)];
        let buf = coff_object(&sections, &symbols, &relocations);
        let obj = coff_parse(&buf);

        let data = &obj.sections[1];
        assert_eq!(data.relocs.len(), 1);
        let (offset, symidx, ty) = data.relocs[0];
        assert_eq!((offset, ty), (0, IMAGE_REL_AMD64_ADDR64));
        assert_eq!(obj.symbols[symidx as usize].name, "main");
        assert_eq!(obj.symbols[symidx as usize].section, 1);
        assert_eq!(&data.data[..8], &8u64.to_le_bytes());
    }

    #[test]
    fn coff_secrel32_relocates_against_tls_symbol_with_type_0b() {
        let text = vec![0x8b, 0x80, 0, 0, 0, 0, 0xc3];
        let sections = [
            coff_section(".text", true, false, text),
            coff_section(".tls$x", false, true, vec![0; 8]),
        ];
        let symbols = [
            coff_symbol("main", true, ".text", 0),
            coff_symbol("_x", true, ".tls$x", 4),
        ];
        let relocations = [coff_reloc(".text", 2, "_x", X86_64_SECREL32, 8)];
        let buf = coff_object(&sections, &symbols, &relocations);
        let obj = coff_parse(&buf);

        let text = &obj.sections[0];
        assert_eq!(text.relocs.len(), 1);
        let (offset, symbol, ty) = text.relocs[0];
        assert_eq!(offset, 2);
        assert_eq!(ty, IMAGE_REL_AMD64_SECREL);
        assert_eq!(ty, 0x000b);
        let target = obj
            .symbols
            .iter()
            .position(|s| s.name == "_x")
            .expect("_x symbol") as u32;
        assert_eq!(symbol, target);
        let tls_index = obj
            .sections
            .iter()
            .position(|s| s.name == ".tls$x")
            .expect(".tls$x section") as i16
            + 1;
        assert_eq!(obj.symbols[target as usize].section, tls_index);
        assert_eq!(obj.symbols[target as usize].value, 4);
        assert_eq!(&text.data[2..6], &8i32.to_le_bytes());
    }

    #[test]
    fn coff_secrel32_refuses_targets_outside_tls_sections() {
        let sections = [
            coff_section(".text", true, false, vec![0x8b, 0x80, 0, 0, 0, 0]),
            coff_section(".data", false, true, vec![0; 4]),
        ];
        let symbols = [coff_symbol("_y", true, ".data", 0)];
        let relocations = [coff_reloc(".text", 2, "_y", X86_64_SECREL32, 0)];
        let opts = ObjectWriteOptions::default();
        let err = CoffWriter::new()
            .object_bytes(&ObjectWriteRequest {
                code: &sections[0].data,
                sections: &sections,
                symbols: &symbols,
                relocations: &relocations,
                target_arch: TargetArchitecture::X86_64,
                target_os: TargetOperatingSystem::Windows,
                opts: &opts,
            })
            .expect_err("non-TLS SECREL target");
        assert!(err.to_string().contains("not a .tls section"), "{err}");
    }

    #[test]
    fn elf_rejects_secrel32_relocations() {
        let sections = [coff_section(
            ".text",
            true,
            false,
            vec![0x8b, 0x80, 0, 0, 0, 0],
        )];
        let relocations = [coff_reloc(".text", 2, "_x", X86_64_SECREL32, 0)];
        let opts = ObjectWriteOptions::default();
        let err = ElfWriter::new()
            .object_bytes(&ObjectWriteRequest {
                code: &sections[0].data,
                sections: &sections,
                symbols: &[],
                relocations: &relocations,
                target_arch: TargetArchitecture::X86_64,
                target_os: TargetOperatingSystem::Linux,
                opts: &opts,
            })
            .expect_err("SECREL in ELF");
        assert!(err.to_string().contains("only supported in COFF"), "{err}");
    }

    fn coff_try_object(
        sections: &[Section],
        symbols: &[ObjectSymbol],
        relocations: &[ExternalReloc],
    ) -> Result<Vec<u8>, RasError> {
        let opts = ObjectWriteOptions::default();
        CoffWriter::new().object_bytes(&ObjectWriteRequest {
            code: &sections[0].data,
            sections,
            symbols,
            relocations,
            target_arch: TargetArchitecture::X86_64,
            target_os: TargetOperatingSystem::Windows,
            opts: &opts,
        })
    }

    /// The raw 18-byte records of the COFF symbol table, auxiliary ones included.
    fn coff_raw_symbols(buf: &[u8]) -> Vec<[u8; COFF_SYMBOL_SIZE]> {
        let sym_ptr = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let nsyms = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        (0..nsyms)
            .map(|i| {
                let e = sym_ptr + i * COFF_SYMBOL_SIZE;
                buf[e..e + COFF_SYMBOL_SIZE].try_into().unwrap()
            })
            .collect()
    }

    #[test]
    fn coff_weak_definition_becomes_a_weak_external_with_its_auxiliary_record() {
        let sections = [coff_section(".text", true, false, vec![0x90, 0xc3])];
        let mut weak = coff_symbol("wf", true, ".text", 1);
        weak.attributes.weak = true;
        weak.attributes.symbol_type = 2;
        let symbols = [coff_symbol("main", true, ".text", 0), weak];
        let buf = coff_object(&sections, &symbols, &[]);
        let obj = coff_parse(&buf);

        let default = obj
            .symbols
            .iter()
            .position(|s| s.name == ".weak.wf.default.main")
            .expect("default definition");
        assert_eq!(obj.symbols[default].class, IMAGE_SYM_CLASS_EXTERNAL);
        assert_eq!(obj.symbols[default].section, 1);
        assert_eq!(obj.symbols[default].value, 1);

        let weak_index = obj
            .symbols
            .iter()
            .position(|s| s.name == "wf")
            .expect("weak external");
        assert_eq!(obj.symbols[weak_index].class, IMAGE_SYM_CLASS_WEAK_EXTERNAL);
        assert_eq!(obj.symbols[weak_index].section, IMAGE_SYM_UNDEFINED);
        assert_eq!(obj.symbols[weak_index].value, 0);
        assert_eq!(obj.symbols[weak_index].typ, IMAGE_SYM_DTYPE_FUNCTION);

        let raw = coff_raw_symbols(&buf);
        assert_eq!(raw.len(), obj.symbols.len(), "NumberOfSymbols counts aux");
        assert_eq!(raw[weak_index][17], 1, "one auxiliary record");
        let aux = raw[weak_index + 1];
        assert_eq!(
            u32::from_le_bytes(aux[0..4].try_into().unwrap()),
            default as u32,
            "aux names the default definition"
        );
        assert_eq!(
            u32::from_le_bytes(aux[4..8].try_into().unwrap()),
            IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY
        );
        assert!(aux[8..].iter().all(|&b| b == 0), "rest of aux is unused");
    }

    #[test]
    fn coff_relocation_against_a_weak_symbol_targets_the_weak_external() {
        let sections = [coff_section(
            ".text",
            true,
            false,
            vec![0xe8, 0, 0, 0, 0, 0xc3],
        )];
        let mut weak = coff_symbol("wf", true, ".text", 5);
        weak.attributes.weak = true;
        let symbols = [coff_symbol("main", true, ".text", 0), weak];
        let relocations = [coff_reloc(".text", 1, "wf", 4, -4)];
        let buf = coff_object(&sections, &symbols, &relocations);
        let obj = coff_parse(&buf);
        let target = obj.sections[0].relocs[0].1 as usize;
        assert_eq!(obj.symbols[target].name, "wf");
        assert_eq!(obj.symbols[target].class, IMAGE_SYM_CLASS_WEAK_EXTERNAL);
    }

    #[test]
    fn coff_weak_reference_without_a_definition_falls_back_to_absolute_zero() {
        let sections = [coff_section(
            ".text",
            true,
            false,
            vec![0xe8, 0, 0, 0, 0, 0xc3],
        )];
        let mut weak = coff_symbol("maybe", true, "", 0);
        weak.attributes.weak = true;
        let symbols = [coff_symbol("main", true, ".text", 0), weak];
        let relocations = [coff_reloc(".text", 1, "maybe", 4, -4)];
        let buf = coff_object(&sections, &symbols, &relocations);
        let obj = coff_parse(&buf);
        let default = obj
            .symbols
            .iter()
            .position(|s| s.name == ".weak.maybe.default.main")
            .expect("default definition");
        assert_eq!(obj.symbols[default].section, IMAGE_SYM_ABSOLUTE);
        assert_eq!(obj.symbols[default].value, 0);
        let weak_index = obj.symbols.iter().position(|s| s.name == "maybe").unwrap();
        assert_eq!(obj.symbols[weak_index].class, IMAGE_SYM_CLASS_WEAK_EXTERNAL);
        let raw = coff_raw_symbols(&buf);
        assert_eq!(
            u32::from_le_bytes(raw[weak_index + 1][0..4].try_into().unwrap()),
            default as u32
        );
    }

    #[test]
    fn coff_weak_default_is_named_after_the_first_strong_external() {
        let sections = [coff_section(".text", true, false, vec![0x90, 0xc3])];
        let mut weak = coff_symbol("wf", true, ".text", 1);
        weak.attributes.weak = true;
        let mut other = coff_symbol("other", true, ".text", 0);
        other.attributes.weak = true;
        let mut quiet = coff_symbol("quiet", true, ".text", 0);
        quiet.attributes.hidden = true;
        let names = |symbols: &[ObjectSymbol]| -> Vec<String> {
            coff_parse(&coff_object(&sections, symbols, &[]))
                .symbols
                .into_iter()
                .filter(|s| s.name.starts_with(".weak.wf"))
                .map(|s| s.name)
                .collect()
        };
        // Weak, file-local, undefined and common symbols do not name the default.
        assert_eq!(
            names(&[
                other.clone(),
                quiet.clone(),
                coff_symbol("ext", true, "", 0),
                coff_symbol("c", true, "*COM*", 8),
                coff_symbol("first", true, ".text", 0),
                coff_symbol("second", true, ".text", 1),
                weak.clone(),
            ]),
            [".weak.wf.default.first"]
        );
        assert_eq!(
            names(&[coff_symbol("second", true, ".text", 1), weak.clone()]),
            [".weak.wf.default.second"]
        );
        assert_eq!(names(&[other, quiet, weak]), [".weak.wf.default"]);
    }

    #[test]
    fn coff_visibility_that_is_not_exported_becomes_a_static_symbol() {
        for set in [
            |a: &mut crate::parser::SymbolAttributes| a.hidden = true,
            |a: &mut crate::parser::SymbolAttributes| a.internal_visibility = true,
            |a: &mut crate::parser::SymbolAttributes| a.private_extern = true,
        ] {
            let sections = [coff_section(".text", true, false, vec![0xc3])];
            let mut sym = coff_symbol("quiet", true, ".text", 0);
            set(&mut sym.attributes);
            let symbols = [coff_symbol("main", true, ".text", 0), sym];
            let buf = coff_object(&sections, &symbols, &[]);
            let obj = coff_parse(&buf);
            let quiet = obj.symbols.iter().find(|s| s.name == "quiet").unwrap();
            assert_eq!(quiet.class, IMAGE_SYM_CLASS_STATIC);
            let main = obj.symbols.iter().find(|s| s.name == "main").unwrap();
            assert_eq!(main.class, IMAGE_SYM_CLASS_EXTERNAL);
        }
    }

    #[test]
    fn coff_refuses_protected_visibility() {
        let sections = [coff_section(".text", true, false, vec![0xc3])];
        let mut sym = coff_symbol("p", true, ".text", 0);
        sym.attributes.protected = true;
        let err = coff_try_object(&sections, &[sym], &[]).expect_err("protected on COFF");
        assert!(err.to_string().contains("protected visibility"), "{err}");
    }

    #[test]
    fn coff_refuses_a_weak_symbol_that_is_also_file_local() {
        for set in [
            |a: &mut crate::parser::SymbolAttributes| a.hidden = true,
            |a: &mut crate::parser::SymbolAttributes| a.local = true,
        ] {
            let sections = [coff_section(".text", true, false, vec![0xc3])];
            let mut sym = coff_symbol("w", true, ".text", 0);
            sym.attributes.weak = true;
            set(&mut sym.attributes);
            let err = coff_try_object(&sections, &[sym], &[]).expect_err("weak and file-local");
            assert!(
                err.to_string().contains("both weak and file-local"),
                "{err}"
            );
        }
    }

    #[test]
    fn coff_refuses_a_weak_common_symbol() {
        let sections = [coff_section(".text", true, false, vec![0xc3])];
        let mut sym = coff_symbol("c", true, "*COM*", 0);
        sym.attributes.weak = true;
        sym.attributes.common = Some((8, 8));
        let err = coff_try_object(&sections, &[sym], &[]).expect_err("weak common");
        assert!(err.to_string().contains("no weak common symbol"), "{err}");
    }

    fn macho_custom_section(name: &str, kind: u32, attributes: u32, data: Vec<u8>) -> Section {
        Section {
            name: name.into(),
            macho: Some(crate::parser::MachOSectionType { kind, attributes }),
            ..macho_text_section(data)
        }
    }

    fn macho_section_name(buf: &[u8], section: usize) -> String {
        let base = macho_section_header(section);
        let name = &buf[base..base + 16];
        String::from_utf8(name.iter().copied().take_while(|&b| b != 0).collect()).unwrap()
    }

    #[test]
    fn macho_arbitrary_segment_sections_keep_name_and_type() {
        let sections = [
            macho_text_section(vec![0x90]),
            macho_custom_section("__TEXT,__custom", 0, 0x8000_0000, vec![0xc3]),
            macho_custom_section("__DATA,__x", 0, 0, vec![1, 2, 3, 4]),
            macho_custom_section("__DATA,__thread_vars", 0x13, 0, vec![0; 24]),
            macho_custom_section("__DATA,__thread_data", 0x11, 0, vec![7; 8]),
            macho_custom_section("__DATA,__thread_bss", 0x12, 0, vec![0; 8]),
        ];
        let buf = macho_object(&sections, &[], &[]).expect("write");
        let names: Vec<String> = (0..sections.len())
            .map(|i| macho_section_name(&buf, i))
            .collect();
        assert_eq!(names[1], "__custom");
        assert_eq!(names[2], "__x");
        assert_eq!(names[3], "__thread_vars");
        assert_eq!(names[4], "__thread_data");
        assert_eq!(names[5], "__thread_bss");
        let segment = |i: usize| {
            let base = macho_section_header(i);
            String::from_utf8(
                buf[base + 16..base + 32]
                    .iter()
                    .copied()
                    .take_while(|&b| b != 0)
                    .collect(),
            )
            .unwrap()
        };
        assert_eq!(segment(1), "__TEXT");
        assert_eq!(segment(3), "__DATA");
        let flags = |i: usize| le32(&buf, macho_section_header(i) + 64);
        assert_eq!(flags(1), 0x8000_0000);
        assert_eq!(flags(3), 0x13);
        assert_eq!(flags(4), 0x11);
        assert_eq!(flags(5), 0x12);
        assert_eq!(macho_section_bytes(&buf, 2), &[1, 2, 3, 4]);
        assert_eq!(macho_section_bytes(&buf, 4), &[7; 8]);
    }

    #[test]
    fn macho_custom_sections_refuse_bad_names_and_nonzero_zerofill() {
        let long = macho_custom_section("__DATA,__this_name_is_too_long", 0, 0, vec![1]);
        let err = macho_object(&[macho_text_section(vec![]), long], &[], &[]).expect_err("long");
        assert!(err.to_string().contains("16 bytes"), "{err}");

        let bare = macho_custom_section("__DATA", 0, 0, vec![1]);
        let err = macho_object(&[macho_text_section(vec![]), bare], &[], &[]).expect_err("bare");
        assert!(
            err.to_string().contains("segment,section") || err.to_string().contains("no segment"),
            "{err}"
        );

        let tls_bss = macho_custom_section("__DATA,__thread_bss", 0x12, 0, vec![0, 1]);
        let err = macho_object(&[macho_text_section(vec![]), tls_bss], &[], &[]).expect_err("tlv");
        assert!(err.to_string().contains("zero-fill"), "{err}");
    }

    fn macho_reloc_fields(buf: &[u8], section: usize) -> Vec<(u32, u32, u32, u32, u32, u32)> {
        macho_reloc_entries(buf, section)
            .into_iter()
            .map(|(addr, info)| {
                (
                    addr,
                    info & 0x00ff_ffff,
                    info >> 24 & 1,
                    info >> 25 & 3,
                    info >> 27 & 1,
                    info >> 28,
                )
            })
            .collect()
    }

    #[test]
    fn macho_arm64_tlvp_operands_emit_tlvp_load_relocations() {
        let source = ".text\n.globl _f\n_f:\nadrp x0, _x@TLVPPAGE\nldr x1, [x0, _x@TLVPPAGEOFF]\nadd x2, x0, _x@TLVPPAGEOFF\nret\n";
        let mut ras = crate::Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::MacOS)
            .expect("ras");
        let buf = ras.assemble_to_bytes(source).expect("assemble");

        let text = macho_section_bytes(&buf, 0);
        assert_eq!(
            &text[0..4],
            &0x9000_0000u32.to_le_bytes(),
            "adrp x0, zero page"
        );
        assert_eq!(&text[4..8], &0xf940_0001u32.to_le_bytes(), "ldr x1, [x0]");
        assert_eq!(
            &text[8..12],
            &0x9100_0002u32.to_le_bytes(),
            "add x2, x0, #0"
        );
        let relocs = macho_reloc_fields(&buf, 0);
        assert_eq!(relocs.len(), 3);
        assert_eq!(
            relocs[0],
            (0, relocs[0].1, 1, 2, 1, ARM64_RELOC_TLVP_LOAD_PAGE21)
        );
        assert_eq!(
            relocs[1],
            (4, relocs[0].1, 0, 2, 1, ARM64_RELOC_TLVP_LOAD_PAGEOFF12)
        );
        assert_eq!(
            relocs[2],
            (8, relocs[0].1, 0, 2, 1, ARM64_RELOC_TLVP_LOAD_PAGEOFF12)
        );
        assert_eq!(ARM64_RELOC_TLVP_LOAD_PAGE21, 8);
        assert_eq!(ARM64_RELOC_TLVP_LOAD_PAGEOFF12, 9);
    }

    #[test]
    fn macho_arm64_tlvp_is_refused_outside_mach_o() {
        let source = ".text\nadrp x0, _x@TLVPPAGE\nret\n";
        let mut ras = crate::Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .expect("ras");
        let err = ras.assemble_to_bytes(source).expect_err("ELF TLVP");
        assert!(
            err.to_string().contains("only supported in Mach-O"),
            "{err}"
        );
    }

    #[test]
    fn macho_thread_sections_carry_tlv_types_and_descriptor_relocations() {
        let source = ".section __DATA,__thread_vars,thread_local_variables\n_x:\n.quad __tlv_bootstrap\n.quad 0\n.quad _store\n.section __DATA,__thread_data,thread_local_regular\n_store:\n.quad 42\n.section __DATA,__thread_bss,thread_local_zerofill\n_b:\n.zero 8\n";
        let mut ras =
            crate::Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::MacOS).expect("ras");
        let buf = ras.assemble_to_bytes(source).expect("assemble");

        let index_of = |name: &str| {
            (0..le32(&buf, 32 + 64) as usize)
                .find(|&i| macho_section_name(&buf, i) == name)
                .unwrap_or_else(|| panic!("section {name} missing"))
        };
        let vars = index_of("__thread_vars");
        let data = index_of("__thread_data");
        let bss = index_of("__thread_bss");
        let flags = |i: usize| le32(&buf, macho_section_header(i) + 64);
        assert_eq!(flags(vars), 0x13, "S_THREAD_LOCAL_VARIABLES");
        assert_eq!(flags(data), 0x11, "S_THREAD_LOCAL_REGULAR");
        assert_eq!(flags(bss), 0x12, "S_THREAD_LOCAL_ZEROFILL");
        assert_eq!(macho_section_bytes(&buf, data), &42u64.to_le_bytes());
        assert_eq!(
            le32(&buf, macho_section_header(bss) + 40),
            8,
            "zero-fill size"
        );

        let relocs = macho_reloc_fields(&buf, vars);
        assert_eq!(relocs.len(), 2);
        assert_eq!(relocs[0].0, 0, "descriptor thunk at +0");
        assert_eq!(relocs[0].3, 3, "8-byte length");
        assert_eq!(relocs[0].2, 0, "not pc-relative");
        assert_eq!(relocs[0].4, 1, "extern to __tlv_bootstrap");
        assert_eq!(relocs[0].5, X86_64_RELOC_UNSIGNED);
        assert_eq!(relocs[1].0, 16, "storage pointer at +16");
        assert_eq!(relocs[1].3, 3);
        assert_eq!(relocs[1].2, 0);
        assert_eq!(relocs[1].5, X86_64_RELOC_UNSIGNED);
    }

    /// `__LD,__compact_unwind` is a regular section the linker reads and does not copy, so
    /// it carries S_ATTR_DEBUG and a pointer relocation on each entry's function address.
    #[test]
    fn macho_compact_unwind_is_a_debug_section_with_pointer_relocations() {
        let source = ".text\n_f:\nret\nLend:\n.section __LD,__compact_unwind,regular,debug\n.p2align 3\n.quad _f\n.long Lend-_f\n.long 16777216\n.quad 0\n.quad 0\n";
        let mut ras =
            crate::Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::MacOS).expect("ras");
        let buf = ras.assemble_to_bytes(source).expect("assemble");

        let index = (0..le32(&buf, 32 + 64) as usize)
            .find(|&i| macho_section_name(&buf, i) == "__compact_unwind")
            .expect("compact unwind section");
        let header = macho_section_header(index);
        assert_eq!(le32(&buf, header + 64), 0x0200_0000, "S_ATTR_DEBUG");
        assert_eq!(le32(&buf, header + 52), 3, "eight-byte alignment");
        let bytes = macho_section_bytes(&buf, index);
        assert_eq!(bytes.len(), 32, "one entry");
        assert_eq!(le32(bytes, 8), 1, "the assembler measured the length");
        assert_eq!(le32(bytes, 12), 0x0100_0000, "the encoding is stored");
        assert_eq!(&bytes[16..32], &[0u8; 16], "no personality and no LSDA");

        let relocs = macho_reloc_fields(&buf, index);
        assert_eq!(relocs.len(), 1, "only the function address relocates");
        assert_eq!(relocs[0].0, 0);
        assert_eq!(relocs[0].3, 3, "8-byte length");
        assert_eq!(relocs[0].2, 0, "not pc-relative");
        assert_eq!(relocs[0].5, X86_64_RELOC_UNSIGNED);
    }

    /// These compare ras against GNU binutils and clang on ELF objects, so they
    /// only run where both the reference assembler and the format are available.
    #[cfg(target_os = "linux")]
    mod elf_sections {
        use crate::Ras;
        use crate::test_support::TempDir;
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

        fn u64_at(bytes: &[u8], offset: usize) -> u64 {
            u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
        }

        #[test]
        fn sections_keep_contents_symbols_and_relocations() {
            let tmp = TempDir::new("sections");
            let path = tmp.path().join("sections.o");
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(
                    ".section .rodata\n.globl message\nmessage: .asciz \"hi\"\n.data\nvalue: .ascii \"ab\"\n.text\n.globl entry\nentry: leaq message(%rip), %rax\nret\n.section .rodata\n.ascii \"!\"\n",
                    &path,
                )
                .unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
            let strings = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;
            let headers = u64_at(&bytes, 40) as usize;
            let string_header = headers + strings * 64;
            let names = &bytes[u64_at(&bytes, string_header + 24) as usize..];
            let section = |name: &str| {
                (1..count)
                    .find_map(|i| {
                        let h = headers + i * 64;
                        let n = u32::from_le_bytes(bytes[h..h + 4].try_into().unwrap()) as usize;
                        (names[n..].split(|b| *b == 0).next().unwrap() == name.as_bytes())
                            .then_some((i, h))
                    })
                    .unwrap()
            };
            let contents = |h: usize| {
                let start = u64_at(&bytes, h + 24) as usize;
                &bytes[start..start + u64_at(&bytes, h + 32) as usize]
            };
            let (rodata_index, rodata) = section(".rodata");
            assert_eq!(u64_at(&bytes, rodata + 8), 2);
            assert_eq!(contents(rodata), b"hi\0!");
            assert_eq!(contents(section(".data").1), b"ab");
            assert_eq!(
                contents(section(".text").1),
                &[0x48, 0x8d, 5, 0, 0, 0, 0, 0xc3]
            );
            let symbols = contents(section(".symtab").1);
            assert!(symbols.as_chunks::<24>().0.iter().any(|s| {
                u16::from_le_bytes(s[6..8].try_into().unwrap()) as usize == rodata_index
                    && s[4] >> 4 == 1
                    && u64_at(s, 8) == 0
            }));
            let reloc = contents(section(".rela.text").1);
            assert_eq!(u64_at(reloc, 0), 3);
            assert_eq!(u64_at(reloc, 8) as u32, 2);
            assert_eq!(u64_at(reloc, 16) as i64, -4);
        }

        #[test]
        fn numeric_data_alignment_and_bss_match_gas() {
            let tmp = TempDir::new("data_sections");
            let dir = tmp.path();
            let source = dir.join("input.s");
            let reference = dir.join("reference.o");
            let output = dir.join("ras.o");
            let asm = ".data\n.byte 1, 255\n.short 0x1234\n.balign 8\n.quad 0x1122334455667788\n.long -1\n.space 3, 42\n.bss\n.balign 16\n.zero 32\n.section .rodata\n.ascii \"a\"\n.p2align 3\n.asciz \"b\"\n";
            std::fs::write(&source, asm).unwrap();
            let result = std::process::Command::new("as")
                .arg("--64")
                .arg(&source)
                .arg("-o")
                .arg(&reference)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(asm, &output)
                .unwrap();
            let inspect = |path: &std::path::Path| {
                let bytes = std::fs::read(path).unwrap();
                let headers = u64_at(&bytes, 40) as usize;
                let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
                let strings = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;
                let names = &bytes[u64_at(&bytes, headers + strings * 64 + 24) as usize..];
                let mut sections = std::collections::BTreeMap::new();
                for i in 1..count {
                    let h = headers + i * 64;
                    let name = u32::from_le_bytes(bytes[h..h + 4].try_into().unwrap()) as usize;
                    let name = names[name..].split(|b| *b == 0).next().unwrap();
                    if ![b".data".as_slice(), b".rodata", b".bss"].contains(&name) {
                        continue;
                    }
                    let kind = u32::from_le_bytes(bytes[h + 4..h + 8].try_into().unwrap());
                    let offset = u64_at(&bytes, h + 24) as usize;
                    let size = u64_at(&bytes, h + 32) as usize;
                    let contents = if kind == 8 {
                        vec![0; size]
                    } else {
                        bytes[offset..offset + size].to_vec()
                    };
                    sections.insert(
                        name.to_vec(),
                        (
                            kind,
                            u64_at(&bytes, h + 8),
                            u64_at(&bytes, h + 48),
                            contents,
                        ),
                    );
                }
                sections
            };
            assert_eq!(inspect(&output), inspect(&reference));
        }

        #[test]
        fn numeric_labels_and_deferred_data_match_gas() {
            let tmp = TempDir::new("label_expressions");
            let dir = tmp.path();
            let source = dir.join("input.s");
            let reference = dir.join("gas.o");
            let output = dir.join("ras.o");
            let asm = ".text\njmp 1f\n0: ret\n1: jmp 0b\n0: jmp 1f\n1: ret\n.data\n0: .byte 7\n.byte 1f-0b\n.short 1f-0b\n.long 1f-0b\n.quad 1f-0b\n.balign 16\n1: .long 0b-1b\n.byte ext+3\n.short ext-2\n.long ext\n.quad ext+8\n.section .note.gnu.property,\"a\"\n.align 8\n.long 1f-0f\n.long 4f-1f\n.long 5\n0: .string \"GNU\"\n1: .long 0xc0000002\n.long 3f-2f\n2: .long 3\n3: .align 8\n4:\n";
            std::fs::write(&source, asm).unwrap();
            let result = std::process::Command::new("as")
                .args(["--64"])
                .arg(&source)
                .arg("-o")
                .arg(&reference)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(asm, &output)
                .unwrap();
            let inspect = |path: &std::path::Path| {
                let bytes = std::fs::read(path).unwrap();
                let headers = u64_at(&bytes, 40) as usize;
                let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
                let strings = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;
                let names = &bytes[u64_at(&bytes, headers + strings * 64 + 24) as usize..];
                let mut sections = std::collections::BTreeMap::new();
                for i in 1..count {
                    let h = headers + i * 64;
                    let n = u32::from_le_bytes(bytes[h..h + 4].try_into().unwrap()) as usize;
                    let name = names[n..].split(|b| *b == 0).next().unwrap();
                    if [
                        b".text".as_slice(),
                        b".data",
                        b".note.gnu.property",
                        b".rela.data",
                    ]
                    .contains(&name)
                    {
                        let start = u64_at(&bytes, h + 24) as usize;
                        let mut data =
                            bytes[start..start + u64_at(&bytes, h + 32) as usize].to_vec();
                        if name == b".rela.data" {
                            for reloc in data.as_chunks_mut::<24>().0 {
                                reloc[12..16].fill(0);
                            }
                        }
                        sections.insert(
                            name.to_vec(),
                            (bytes[h + 4..h + 16].to_vec(), u64_at(&bytes, h + 48), data),
                        );
                    }
                    if name == b".symtab" {
                        let start = u64_at(&bytes, h + 24) as usize;
                        let symbols = &bytes[start..start + u64_at(&bytes, h + 32) as usize];
                        let globals: Vec<_> = symbols
                            .as_chunks::<24>()
                            .0
                            .iter()
                            .filter(|sym| sym[4] >> 4 == 1)
                            .collect();
                        assert_eq!(globals.len(), 1);
                        assert_eq!(&globals[0][6..8], &[0, 0]);
                    }
                }
                sections
            };
            assert_eq!(inspect(&output), inspect(&reference));
        }

        #[test]
        fn undefined_numeric_labels_and_cross_section_differences_fail() {
            let tmp = TempDir::new("invalid_labels");
            let path = tmp.path().join("invalid_labels.o");
            for (asm, error) in [
                (".long 1f", "Undefined numeric label: 1f"),
                ("jmp 1b\n1: ret", "Undefined numeric label: 1b"),
                (
                    ".data\nx: .long x-y\n.text\ny: ret",
                    "Undefined or cross-section label in difference: y",
                ),
            ] {
                let result = Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                    .unwrap()
                    .assemble(asm, &path);
                assert!(result.unwrap_err().to_string().contains(error));
            }
            assert!(!path.exists());
        }

        #[test]
        fn section_name_may_be_quoted() {
            let tmp = TempDir::new("quoted");
            let dir = tmp.path();
            let body = ".text\n.globl f\nf: ret\n";
            let assemble = |source: &str, name: &str| {
                let path = dir.join(name);
                Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                    .unwrap()
                    .assemble(source, &path)
                    .unwrap();
                std::fs::read(&path).unwrap()
            };
            let quoted = assemble(
                &format!("{body}.section \".note.GNU-stack\",\"\",@progbits\n"),
                "quoted.o",
            );
            let bare = assemble(
                &format!("{body}.section .note.GNU-stack,\"\",@progbits\n"),
                "bare.o",
            );
            assert_eq!(quoted, bare);
        }
    }

    /// COFF section flags.
    mod coff_sections {
        use crate::Ras;
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

        const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
        const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
        const IMAGE_SCN_MEM_DISCARDABLE: u32 = 0x0200_0000;
        const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
        const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

        fn u16_at(bytes: &[u8], at: usize) -> usize {
            u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()) as usize
        }

        fn u32_at(bytes: &[u8], at: usize) -> u32 {
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
        }

        /// Characteristics of the named COFF section. Names longer than eight bytes are
        /// held in the string table and the header keeps `/offset`.
        fn characteristics(bytes: &[u8], name: &str) -> u32 {
            let sections = u16_at(bytes, 2);
            let strtab = u32_at(bytes, 8) as usize + u32_at(bytes, 12) as usize * 18;
            let header = (0..sections)
                .map(|i| 20 + 40 * i)
                .find(|&h| {
                    let raw = bytes[h..h + 8].split(|b| *b == 0).next().unwrap();
                    let raw = match raw.strip_prefix(b"/") {
                        Some(offset) => {
                            let at = strtab
                                + std::str::from_utf8(offset)
                                    .unwrap()
                                    .parse::<usize>()
                                    .unwrap();
                            bytes[at..].split(|b| *b == 0).next().unwrap()
                        }
                        None => raw,
                    };
                    raw == name.as_bytes()
                })
                .expect("section present");
            u32_at(bytes, header + 36)
        }

        fn assemble_windows(source: &str) -> Vec<u8> {
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Windows)
                .unwrap()
                .assemble_to_bytes(source)
                .unwrap()
        }

        #[test]
        fn custom_code_section_is_kept_code() {
            let obj = assemble_windows(
                ".text\n\
                 .globl main\n\
                 main:\n\
                     call custom\n\
                     ret\n\
                 .section .text.custom,\"xr\"\n\
                 .globl custom\n\
                 custom:\n\
                     ret\n",
            );
            let flags = characteristics(&obj, ".text.custom");
            assert_eq!(
                flags & (IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE),
                IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE
            );
            assert_eq!(flags & (IMAGE_SCN_MEM_WRITE | IMAGE_SCN_MEM_DISCARDABLE), 0);
        }

        #[test]
        fn data_section_flags_set_write_and_discard() {
            let obj = assemble_windows(
                ".section .tls$,\"dw\"\n\
                     .quad 0\n\
                 .section .rdata$x,\"dr\"\n\
                     .quad 0\n\
                 .section .drop,\"drD\"\n\
                     .quad 0\n\
                 .section .debug$x,\"dr\"\n\
                     .quad 0\n\
                 .text\n\
                 .globl main\n\
                 main:\n\
                     ret\n",
            );
            let tls = characteristics(&obj, ".tls$");
            assert_ne!(tls & IMAGE_SCN_CNT_INITIALIZED_DATA, 0);
            assert_ne!(tls & IMAGE_SCN_MEM_WRITE, 0);
            assert_eq!(tls & IMAGE_SCN_MEM_DISCARDABLE, 0);
            let rdata = characteristics(&obj, ".rdata$x");
            assert_eq!(rdata & (IMAGE_SCN_MEM_WRITE | IMAGE_SCN_MEM_DISCARDABLE), 0);
            for discarded in [".drop", ".debug$x"] {
                let flags = characteristics(&obj, discarded);
                assert_ne!(flags & IMAGE_SCN_MEM_DISCARDABLE, 0, "{discarded}");
                assert_eq!(flags & IMAGE_SCN_MEM_WRITE, 0, "{discarded}");
            }
        }
    }

    /// Section relative relocations.
    mod coff_secrel {
        use crate::Ras;
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

        const IMAGE_REL_AMD64_ADDR32NB: u16 = 0x0003;
        const IMAGE_REL_AMD64_SECREL: u16 = 0x000b;
        const IMAGE_REL_AMD64_SREL32: u16 = 0x000e;

        fn u16_at(bytes: &[u8], at: usize) -> usize {
            u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()) as usize
        }

        fn u32_at(bytes: &[u8], at: usize) -> u32 {
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
        }

        /// `(offset, type, field)` for each relocation of the named COFF section, where
        /// `field` is the 32-bit value stored at the relocated position.
        fn section_relocations(bytes: &[u8], name: &str) -> Vec<(u32, u16, u32)> {
            let sections = u16_at(bytes, 2);
            let header = (0..sections)
                .map(|i| 20 + 40 * i)
                .find(|&h| bytes[h..h + 8].split(|b| *b == 0).next() == Some(name.as_bytes()))
                .expect("section present");
            let raw = u32_at(bytes, header + 20) as usize;
            let relocs = u32_at(bytes, header + 24) as usize;
            let count = u16_at(bytes, header + 32);
            (0..count)
                .map(|i| {
                    let at = relocs + 10 * i;
                    let offset = u32_at(bytes, at);
                    let ty = u16_at(bytes, at + 8) as u16;
                    let field = u32_at(bytes, raw + offset as usize);
                    (offset, ty, field)
                })
                .collect()
        }

        fn assemble_windows(source: &str) -> Vec<u8> {
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Windows)
                .unwrap()
                .assemble_to_bytes(source)
                .unwrap()
        }

        #[test]
        fn secrel32_emits_section_relative_type_not_srel32() {
            let obj = assemble_windows(
                ".text\n\
                 .globl f\n\
                 f:\n\
                     leaq _x@SECREL32(%rax), %rax\n\
                     movl _x@SECREL32+8(%rsp), %ecx\n\
                     ret\n\
                 .section .tls$,\"aw\"\n\
                 .p2align 3\n\
                 _x:\n\
                     .zero 16\n",
            );
            let relocs = section_relocations(&obj, ".text");
            assert_eq!(relocs.len(), 2, "{relocs:?}");
            for (_, ty, _) in &relocs {
                assert_eq!(*ty, IMAGE_REL_AMD64_SECREL);
                assert_ne!(*ty, IMAGE_REL_AMD64_SREL32);
            }
            assert_eq!(relocs[0].0, 3);
            assert_eq!(relocs[1].0, 10);
            assert_eq!(relocs[1].2, 8);
        }

        #[test]
        fn rva_and_imgrel_emit_image_relative_addr32nb() {
            let obj = assemble_windows(
                ".text\n\
                 .globl f\n\
                 f:\n\
                     ret\n\
                 .Lend:\n\
                 .section .xdata,\"dr\"\n\
                 .Linfo:\n\
                     .long 0\n\
                 .section .pdata,\"dr\"\n\
                     .rva f, .Lend\n\
                     .long .Linfo@IMGREL\n",
            );
            let relocs = section_relocations(&obj, ".pdata");
            assert_eq!(relocs.len(), 3, "{relocs:?}");
            for (index, (offset, ty, _)) in relocs.iter().enumerate() {
                assert_eq!(*ty, IMAGE_REL_AMD64_ADDR32NB, "{relocs:?}");
                assert_eq!(*offset, 4 * index as u32, "{relocs:?}");
            }
        }

        #[test]
        fn rva_is_refused_outside_coff() {
            let error = Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble_to_bytes(".text\n.globl f\nf:\n    ret\n.data\n    .rva f\n")
                .expect_err("ELF has no image base");
            assert!(
                error.to_string().contains("only supported in COFF objects"),
                "{error}"
            );
        }
    }

    /// The ARM64 COFF writer: which relocations it maps, and what it refuses.
    mod aarch64_coff {
        use crate::Ras;
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

        const IMAGE_FILE_MACHINE_ARM64: u16 = 0xAA64;
        const SECTION_HEADER_SIZE: usize = 40;
        const SYMBOL_SIZE: usize = 18;
        const RELOC_SIZE: usize = 10;

        fn assemble(asm: &str) -> Result<Vec<u8>, String> {
            Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Windows)
                .map_err(|e| e.to_string())?
                .assemble_to_bytes(asm)
                .map_err(|e| e.to_string())
        }

        fn object(asm: &str) -> Vec<u8> {
            assemble(asm).expect("aarch64 COFF object")
        }

        fn u16_at(bytes: &[u8], at: usize) -> u16 {
            u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
        }

        fn u32_at(bytes: &[u8], at: usize) -> u32 {
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
        }

        fn symbol_names(bytes: &[u8]) -> Vec<String> {
            let sym_ptr = u32_at(bytes, 8) as usize;
            let nsyms = u32_at(bytes, 12) as usize;
            let strtab = sym_ptr + nsyms * SYMBOL_SIZE;
            let mut names = Vec::new();
            let mut i = 0;
            while i < nsyms {
                let e = sym_ptr + i * SYMBOL_SIZE;
                let raw = &bytes[e..e + 8];
                let name = if raw[..4] == [0, 0, 0, 0] {
                    let off = strtab + u32_at(raw, 4) as usize;
                    let end = bytes[off..].iter().position(|&b| b == 0).unwrap();
                    String::from_utf8(bytes[off..off + end].to_vec()).unwrap()
                } else {
                    String::from_utf8(raw.iter().copied().take_while(|&b| b != 0).collect())
                        .unwrap()
                };
                names.push(name);
                i += 1 + bytes[e + 17] as usize;
            }
            names
        }

        /// `(section name, offset, relocation type, target symbol name)` for every relocation.
        fn relocations(bytes: &[u8]) -> Vec<(String, u32, u16, String)> {
            let names = symbol_names(bytes);
            let nsections = u16_at(bytes, 2) as usize;
            let sym_ptr = u32_at(bytes, 8) as usize;
            let nsyms = u32_at(bytes, 12) as usize;
            let strtab = sym_ptr + nsyms * SYMBOL_SIZE;
            // Symbol indices count auxiliary records, so build an index-to-name map.
            let mut by_index = vec![String::new(); nsyms];
            let mut i = 0;
            let mut n = 0;
            while i < nsyms {
                by_index[i] = names[n].clone();
                i += 1 + bytes[sym_ptr + i * SYMBOL_SIZE + 17] as usize;
                n += 1;
            }
            let mut out = Vec::new();
            for s in 0..nsections {
                let h = 20 + s * SECTION_HEADER_SIZE;
                let raw = &bytes[h..h + 8];
                let section = if raw[0] == b'/' {
                    let off: usize = std::str::from_utf8(&raw[1..])
                        .unwrap()
                        .trim_end_matches('\0')
                        .parse()
                        .unwrap();
                    let at = strtab + off;
                    let end = bytes[at..].iter().position(|&b| b == 0).unwrap();
                    String::from_utf8(bytes[at..at + end].to_vec()).unwrap()
                } else {
                    String::from_utf8(raw.iter().copied().take_while(|&b| b != 0).collect())
                        .unwrap()
                };
                let reloc_ptr = u32_at(bytes, h + 24) as usize;
                for r in 0..u16_at(bytes, h + 32) as usize {
                    let e = reloc_ptr + r * RELOC_SIZE;
                    out.push((
                        section.clone(),
                        u32_at(bytes, e),
                        u16_at(bytes, e + 8),
                        by_index[u32_at(bytes, e + 4) as usize].clone(),
                    ));
                }
            }
            out
        }

        #[test]
        fn object_header_names_the_arm64_machine() {
            let obj = object(".text\n.globl main\nmain:\n    ret\n");
            assert_eq!(u16_at(&obj, 0), IMAGE_FILE_MACHINE_ARM64);
        }

        #[test]
        fn page_and_page_offset_pairs_become_pagebase_rel21_and_pageoffset_12a() {
            let obj = object(
                ".data\nmsg:\n    .byte 0x68, 0x00\n\
                 .text\n.globl main\nmain:\n    adrp x0, msg\n    add x0, x0, :lo12:msg\n    ret\n",
            );
            assert_eq!(
                relocations(&obj),
                vec![
                    (".text".into(), 0, 0x0004, "msg".into()),
                    (".text".into(), 4, 0x0006, "msg".into()),
                ]
            );
        }

        #[test]
        fn a_relocated_load_becomes_pageoffset_12l() {
            let obj = object(
                ".data\n.p2align 3\nslot:\n    .quad 0\n\
                 .text\n.globl main\nmain:\n    adrp x0, slot\n    ldr x1, [x0, #:lo12:slot]\n    ret\n",
            );
            let kinds: Vec<u16> = relocations(&obj).iter().map(|r| r.2).collect();
            assert_eq!(kinds, vec![0x0004, 0x0007]);
        }

        #[test]
        fn a_call_to_an_external_function_becomes_branch26() {
            let obj = object(".text\n.globl main\nmain:\n    bl other\n    ret\n");
            assert_eq!(
                relocations(&obj),
                vec![(".text".into(), 0, 0x0003, "other".into())]
            );
        }

        #[test]
        fn branches_to_a_global_keep_their_field_width() {
            let obj = object(
                ".text\n.globl main\n.globl far\nmain:\n\
                     b.ne far\n    tbz x0, #3, far\n    b far\n    ret\nfar:\n    ret\n",
            );
            let kinds: Vec<u16> = relocations(&obj).iter().map(|r| r.2).collect();
            assert_eq!(kinds, vec![0x000F, 0x0010, 0x0003]);
        }

        #[test]
        fn an_absolute_pointer_becomes_addr64_with_the_addend_in_the_field() {
            let obj =
                object(".data\nslot:\n    .quad target + 24\n.text\n.globl main\nmain:\n    ret\n");
            let relocs = relocations(&obj);
            assert_eq!(relocs.len(), 1);
            assert_eq!((relocs[0].2, relocs[0].3.as_str()), (0x000E, "target"));

            let data_header = (0..u16_at(&obj, 2) as usize)
                .map(|s| 20 + s * SECTION_HEADER_SIZE)
                .find(|&h| obj[h..h + 5] == *b".data")
                .expect(".data header");
            let at = u32_at(&obj, data_header + 20) as usize + relocs[0].1 as usize;
            assert_eq!(u64::from_le_bytes(obj[at..at + 8].try_into().unwrap()), 24);
        }

        #[test]
        fn an_image_relative_address_becomes_addr32nb() {
            let obj = object(
                ".text\n.globl main\nmain:\n    ret\n.section .pdata,\"dr\"\n    .rva main\n",
            );
            let relocs = relocations(&obj);
            assert_eq!(relocs.len(), 1);
            assert_eq!(
                (relocs[0].0.as_str(), relocs[0].2, relocs[0].3.as_str()),
                (".pdata", 0x0002, "main")
            );
        }

        #[test]
        fn instruction_relocations_leave_the_instruction_word_untouched() {
            let obj =
                object(".data\nmsg:\n    .byte 0\n.text\n.globl main\nmain:\n    adrp x0, msg\n");
            let text_header = (0..u16_at(&obj, 2) as usize)
                .map(|s| 20 + s * SECTION_HEADER_SIZE)
                .find(|&h| obj[h..h + 5] == *b".text")
                .expect(".text header");
            let at = u32_at(&obj, text_header + 20) as usize;
            assert_eq!(u32_at(&obj, at), 0x9000_0000);
        }

        #[test]
        fn elf_mapping_symbols_stay_out_of_coff_objects() {
            let obj = object(".text\n.globl main\nmain:\n    ret\n.data\nd:\n    .byte 1\n");
            let names = symbol_names(&obj);
            assert!(
                !names
                    .iter()
                    .any(|n| n.starts_with("$x") || n.starts_with("$d")),
                "{names:?}"
            );
        }

        #[test]
        fn a_got_page_relocation_is_refused_by_name() {
            let error = assemble(".text\n.globl main\nmain:\n    adrp x0, :got:sym\n    ret\n")
                .expect_err("refused");
            assert!(error.contains("R_AARCH64_ADR_GOT_PAGE"), "{error}");
            assert!(error.contains("sym"), "{error}");
        }

        #[test]
        fn a_got_low12_relocation_is_refused_by_name() {
            let error =
                assemble(".text\n.globl main\nmain:\n    ldr x0, [x0, #:got_lo12:sym]\n    ret\n")
                    .expect_err("refused");
            assert!(error.contains("R_AARCH64_LD64_GOT_LO12_NC"), "{error}");
        }

        #[test]
        fn an_addend_on_an_instruction_relocation_is_refused() {
            let error = assemble(
                ".data\nmsg:\n    .byte 0\n.text\n.globl main\nmain:\n    adrp x0, msg+8\n    ret\n",
            )
            .expect_err("refused");
            assert!(error.contains("R_AARCH64_ADR_PREL_PG_HI21"), "{error}");
            assert!(error.contains("addend 8"), "{error}");
        }

        #[test]
        fn a_pc_relative_data_difference_is_refused_by_name() {
            let error = assemble(
                ".text\n.globl main\nmain:\n    ret\n.data\nhere:\n    .word there - here\n",
            )
            .expect_err("refused");
            assert!(error.contains("R_AARCH64_PREL32"), "{error}");
        }
    }

    /// The same DWARF, carried in a Mach-O object.
    ///
    /// Mach-O spells a debug section `__DWARF,__debug_*` and marks it `S_ATTR_DEBUG`, and
    /// its linker drops those sections rather than relocating them, so the only relocation
    /// left is the one that names code. Every assertion below is a structural one, and
    /// nothing in this file claims a debugger ran.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    mod macho_debug_info {
        use crate::test_support::TempDir;
        use crate::{
            BaseType, DebugInfoRequest, DebugType, DeclSite, FormalParameter, ObjectWriteOptions,
            Ras, Subprogram,
        };
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
        use std::path::Path;
        use std::process::Command;

        const RELOC_UNSIGNED: u32 = 0;

        const SOURCE: &str = ".file 1 \"shapes.lamina\"\n\
        .text\n\
        .globl _twice\n\
        _twice:\n\
        .loc 1 1 1\n\
        addl %edi, %edi\n\
        movl %edi, %eax\n\
        ret\n\
        _half:\n\
        .loc 1 6 1\n\
        movl %edi, %eax\n\
        ret\n";

        /// The DWARF name is the source name and the object label carries Mach-O's underscore,
        /// so the two differ for every function here.
        fn request() -> DebugInfoRequest {
            let describe = |name: &str, line: u32| Subprogram {
                name: name.into(),
                symbol: Some(format!("_{name}")),
                external: name == "twice",
                decl: Some(DeclSite { file: 1, line }),
                ret: Some(DebugType::Base(BaseType::I32)),
                params: vec![FormalParameter {
                    name: Some("n".into()),
                    ty: Some(DebugType::Base(BaseType::I32)),
                }],
                variadic: false,
            };
            DebugInfoRequest {
                producer: Some("ras test".into()),
                subprograms: vec![describe("twice", 1), describe("half", 6)],
            }
        }

        fn assemble(arch: TargetArchitecture, path: &Path) {
            let options = ObjectWriteOptions {
                function_symbols: vec!["_twice".into(), "_half".into()],
                debug_info: Some(request()),
                ..ObjectWriteOptions::default()
            };
            let source = match arch {
                TargetArchitecture::Aarch64 => SOURCE
                    .replace("addl %edi, %edi", "add w0, w0, w0")
                    .replace("movl %edi, %eax", "mov w0, w0"),
                _ => SOURCE.to_string(),
            };
            Ras::with_object_write_options(arch, TargetOperatingSystem::MacOS, options)
                .expect("assembler")
                .assemble(&source, path)
                .expect("assemble");
        }

        fn which(tool: &str) -> Option<String> {
            for name in [tool.to_string(), format!("{tool}-18"), format!("{tool}-19")] {
                let found = Command::new(&name)
                    .arg("--version")
                    .output()
                    .is_ok_and(|output| output.status.success());
                if found {
                    return Some(name);
                }
            }
            // Xcode keeps its LLVM tools out of PATH.
            let output = Command::new("xcrun")
                .args(["--find", tool])
                .output()
                .ok()
                .filter(|output| output.status.success())?;
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        }

        #[test]
        fn the_debug_sections_are_in_the_dwarf_segment_and_marked_debug() {
            if which("llvm-readobj").is_none() {
                return;
            }
            let dir = TempDir::new("macho_debug_sections");
            let path = dir.path().join("out.o");
            assemble(TargetArchitecture::X86_64, &path);
            let output = Command::new(which("llvm-readobj").expect("llvm-readobj"))
                .arg("--sections")
                .arg(&path)
                .output()
                .expect("llvm-readobj");
            let dump = String::from_utf8_lossy(&output.stdout).into_owned();

            for name in ["__debug_line", "__debug_abbrev", "__debug_info"] {
                let section = dump
                    .split("    Index:")
                    .find(|part| part.contains(&format!("Name: {name} ")))
                    .unwrap_or_else(|| panic!("no section {name} in {dump}"));
                assert!(section.contains("Segment: __DWARF "), "{section}");
                assert!(section.contains("Debug ("), "{section}");
            }
        }

        /// `ld` discards the `__DWARF` sections instead of rebasing them, so an offset from one
        /// into another is already final. Only the addresses that name code may be relocated.
        #[test]
        fn the_only_relocations_in_the_debug_sections_name_the_text_section() {
            let dir = TempDir::new("macho_debug_relocs");
            let path = dir.path().join("out.o");
            assemble(TargetArchitecture::X86_64, &path);
            let object = std::fs::read(&path).expect("read object");
            let macho = Macho::parse(&object);

            let text = macho
                .sections
                .iter()
                .position(|s| s.name == "__text")
                .expect("a text section");
            let mut relocated = 0;
            for section in macho.sections.iter().filter(|s| s.segment == "__DWARF") {
                for &(_, info) in &section.relocs {
                    assert_eq!(info >> 28, RELOC_UNSIGNED, "{} type", section.name);
                    assert_eq!((info >> 25) & 3, 3, "{} width", section.name);
                    assert_eq!((info >> 24) & 1, 0, "{} pcrel", section.name);
                    assert_eq!((info >> 27) & 1, 0, "{} extern", section.name);
                    assert_eq!(info & 0x00ff_ffff, text as u32 + 1, "{}", section.name);
                    relocated += 1;
                }
            }
            assert_eq!(
                relocated, 4,
                "the unit's and both subprograms' low_pc, and the line table's first address"
            );
            assert!(
                macho
                    .sections
                    .iter()
                    .any(|s| s.name == "__debug_abbrev" && s.relocs.is_empty())
            );
        }

        #[test]
        fn a_subprogram_is_named_without_the_object_underscore() {
            let Some(dwarfdump) = which("llvm-dwarfdump") else {
                return;
            };
            let dir = TempDir::new("macho_debug_names");
            let path = dir.path().join("out.o");
            assemble(TargetArchitecture::X86_64, &path);
            let output = Command::new(dwarfdump)
                .arg("--debug-info")
                .arg(&path)
                .output()
                .expect("llvm-dwarfdump");
            let dump = String::from_utf8_lossy(&output.stdout).into_owned();

            assert!(dump.contains("DW_AT_name\t(\"twice\")"), "{dump}");
            assert!(dump.contains("DW_AT_name\t(\"half\")"), "{dump}");
            assert!(!dump.contains("\"_twice\""), "{dump}");
            assert_eq!(dump.matches("DW_TAG_subprogram").count(), 2, "{dump}");
        }

        /// The property the ELF path holds, held here too: nothing knows where the allocator
        /// put a parameter, so the answer must be silence rather than a guess.
        #[test]
        fn a_parameter_without_a_known_location_is_given_none() {
            let Some(dwarfdump) = which("llvm-dwarfdump") else {
                return;
            };
            let dir = TempDir::new("macho_debug_no_location");
            let path = dir.path().join("out.o");
            assemble(TargetArchitecture::X86_64, &path);
            let output = Command::new(dwarfdump)
                .arg("--debug-info")
                .arg(&path)
                .output()
                .expect("llvm-dwarfdump");
            let dump = String::from_utf8_lossy(&output.stdout).into_owned();

            assert!(!dump.contains("DW_AT_location"), "{dump}");
        }

        #[test]
        fn both_mach_o_architectures_pass_the_dwarf_verifier() {
            let Some(dwarfdump) = which("llvm-dwarfdump") else {
                return;
            };
            for arch in [TargetArchitecture::X86_64, TargetArchitecture::Aarch64] {
                let dir = TempDir::new("macho_debug_verify");
                let path = dir.path().join("out.o");
                assemble(arch, &path);
                let output = Command::new(&dwarfdump)
                    .arg("--verify")
                    .arg(&path)
                    .output()
                    .expect("llvm-dwarfdump");
                let text = String::from_utf8_lossy(&output.stdout);
                assert!(output.status.success(), "{arch:?}: {text}");
                assert!(text.contains("No errors."), "{arch:?}: {text}");
            }
        }

        /// `dsymutil` reads the objects themselves, so feeding it a debug map that names this
        /// object is the whole of the dSYM path short of a linker that writes one. It relocates
        /// `low_pc` and the line table to the addresses the map gives and verifies its output.
        #[test]
        fn dsymutil_links_the_debug_info_to_the_addresses_a_debug_map_gives() {
            let Some(dsymutil) = which("dsymutil") else {
                return;
            };
            let dir = TempDir::new("macho_debug_dsym");
            let path = dir.path().join("out.o");
            assemble(TargetArchitecture::X86_64, &path);
            let base = 0x1_0000_1000u64;
            let map = dir.path().join("map.yaml");
            let entries: String = [("_twice", 0u64, 5u64), ("_half", 5, 3)]
                .iter()
                .map(|(name, at, size)| {
                    format!(
                        "      - {{ sym: {name}, objAddr: 0x{at:X}, binAddr: 0x{:X}, size: 0x{size:X} }}\n",
                        base + at
                    )
                })
                .collect();
            std::fs::write(
                &map,
                format!(
                    "---\ntriple:          'x86_64-apple-darwin'\nbinary-path:     '{0}'\nobjects:\n  - filename:        '{0}'\n    timestamp:       0\n    symbols:\n{entries}...\n",
                    path.display()
                ),
            )
            .expect("write debug map");

            let out = dir.path().join("out.dwarf");
            let run = Command::new(dsymutil)
                .args(["-y", "-f", "--verify-dwarf=all", "-o"])
                .arg(&out)
                .arg(&map)
                .output()
                .expect("dsymutil");
            assert!(
                run.status.success(),
                "{}{}",
                String::from_utf8_lossy(&run.stdout),
                String::from_utf8_lossy(&run.stderr)
            );

            let dwarfdump = which("llvm-dwarfdump").expect("llvm-dwarfdump beside dsymutil");
            let dump = Command::new(dwarfdump)
                .arg("--debug-info")
                .arg(&out)
                .output()
                .expect("llvm-dwarfdump");
            let dump = String::from_utf8_lossy(&dump.stdout).into_owned();
            assert!(
                dump.contains(&format!("DW_AT_low_pc\t(0x{base:016x})")),
                "{dump}"
            );
            assert!(
                dump.contains(&format!("DW_AT_low_pc\t(0x{:016x})", base + 5)),
                "half moved with the debug map: {dump}"
            );
        }

        struct MachoSection {
            segment: String,
            name: String,
            relocs: Vec<(u32, u32)>,
        }

        struct Macho {
            sections: Vec<MachoSection>,
        }

        impl Macho {
            fn parse(object: &[u8]) -> Self {
                let u32_at =
                    |at: usize| u32::from_le_bytes(object[at..at + 4].try_into().expect("four"));
                let name16 = |at: usize| {
                    let field = &object[at..at + 16];
                    let end = field.iter().position(|&b| b == 0).unwrap_or(16);
                    String::from_utf8_lossy(&field[..end]).into_owned()
                };
                assert_eq!(u32_at(0), 0xfeed_facf, "Mach-O 64 magic");
                let mut sections = Vec::new();
                let mut command = 32;
                for _ in 0..u32_at(16) {
                    if u32_at(command) == 0x19 {
                        let count = u32_at(command + 64) as usize;
                        for index in 0..count {
                            let header = command + 72 + 80 * index;
                            let relocation_offset = u32_at(header + 56) as usize;
                            let relocation_count = u32_at(header + 60) as usize;
                            sections.push(MachoSection {
                                name: name16(header),
                                segment: name16(header + 16),
                                relocs: (0..relocation_count)
                                    .map(|entry| {
                                        let at = relocation_offset + 8 * entry;
                                        (u32_at(at), u32_at(at + 4))
                                    })
                                    .collect(),
                            });
                        }
                    }
                    command += u32_at(command + 4) as usize;
                }
                Self { sections }
            }
        }
    }
}
