//! Object file generation: ELF (Linux/BSD), Mach-O (macOS), COFF/PE (Windows).

#[cfg(test)]
use crate::parser::SectionFlags;
use crate::{error::RasError, parser::Section};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::io::Write;

/// Extra sections and metadata for [`ObjectWriter`] backends.
#[derive(Debug, Clone)]
pub struct ObjectWriteOptions {
    /// When true, [`ElfWriter`] also emits minimal `.debug_line`, `.debug_abbrev`, and `.debug_info`.
    pub emit_minimal_dwarf_sections: bool,
    /// File name stored in DWARF line/info (used when `emit_minimal_dwarf_sections` is set).
    pub dwarf_decl_file_name: String,
}

impl Default for ObjectWriteOptions {
    fn default() -> Self {
        Self {
            emit_minimal_dwarf_sections: false,
            dwarf_decl_file_name: "lamina.s".to_string(),
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
}

/// All data needed to write a single relocatable object file.
///
/// Passed to [`ObjectWriter::write_object_file`] so the trait method signature
/// stays below clippy's argument-count threshold without losing any information.
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
    /// Write a relocatable object file to `path`.
    fn write_object_file(
        &mut self,
        path: &std::path::Path,
        req: &ObjectWriteRequest<'_>,
    ) -> Result<(), RasError>;
}

const ELF64_HEADER_SIZE: usize = 64;
const ELF64_SECTION_HEADER_SIZE: usize = 64;
const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ET_REL: u16 = 1;
const EM_X86_64: u16 = 62;
const EM_AARCH64: u16 = 183;
const EM_RISCV: u16 = 243;
const EM_ARX64: u16 = 0xa064;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHF_INFO_LINK: u64 = 0x40;
const ELF64_SYM_SIZE: u64 = 24;
const ELF64_RELA_SIZE: u64 = 24;
const STB_GLOBAL: u8 = 1;

fn elf64_e_machine(arch: TargetArchitecture) -> Option<u16> {
    match arch {
        TargetArchitecture::X86_64 => Some(EM_X86_64),
        TargetArchitecture::Aarch64 => Some(EM_AARCH64),
        TargetArchitecture::Arx64 => Some(EM_ARX64),
        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => Some(EM_RISCV),
        _ => None,
    }
}

fn push_uleb128(buf: &mut Vec<u8>, mut n: u32) {
    loop {
        let mut b = (n & 0x7f) as u8;
        n >>= 7;
        if n != 0 {
            b |= 0x80;
        }
        buf.push(b);
        if n == 0 {
            break;
        }
    }
}

fn dwarf_debug_line_bytes(decl_file: &str) -> Vec<u8> {
    let mut prog = Vec::new();
    prog.push(0);
    push_uleb128(&mut prog, 9);
    prog.push(2);
    prog.extend_from_slice(&0u64.to_le_bytes());
    prog.push(1);
    prog.push(0);
    push_uleb128(&mut prog, 1);
    prog.push(1);

    let std_lens = [0u8, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1];
    let mut body_tail = vec![1u8, 1, 1, -5i8 as u8, 14, 13];
    body_tail.extend_from_slice(&std_lens);
    body_tail.push(0);
    push_uleb128(&mut body_tail, 0);
    push_uleb128(&mut body_tail, 0);
    push_uleb128(&mut body_tail, 0);
    for b in decl_file.as_bytes() {
        if *b == 0 {
            break;
        }
        body_tail.push(*b);
    }
    body_tail.push(0);
    body_tail.push(0);
    body_tail.extend_from_slice(&prog);

    let header_length = body_tail.len() as u32;
    let mut unit_inner = Vec::new();
    unit_inner.extend_from_slice(&4u16.to_le_bytes());
    unit_inner.extend_from_slice(&header_length.to_le_bytes());
    unit_inner.extend_from_slice(&body_tail);
    let unit_length = unit_inner.len() as u32;
    let mut out = Vec::new();
    out.extend_from_slice(&unit_length.to_le_bytes());
    out.extend_from_slice(&unit_inner);
    out
}

fn dwarf_debug_abbrev_bytes() -> Vec<u8> {
    vec![
        1, 0x11, 0x00, 0x16, 0x17, 0x03, 0x08, 0x00, 0x00, 0x00, 0x00,
    ]
}

fn dwarf_debug_info_bytes(decl_file: &str) -> Vec<u8> {
    let mut die = Vec::new();
    die.push(1);
    die.extend_from_slice(&0u32.to_le_bytes());
    for b in decl_file.as_bytes() {
        if *b == 0 {
            break;
        }
        die.push(*b);
    }
    die.push(0);

    let mut inner = Vec::new();
    inner.extend_from_slice(&4u16.to_le_bytes());
    inner.extend_from_slice(&0u32.to_le_bytes());
    inner.push(8);
    inner.extend_from_slice(&die);
    let unit_length = inner.len() as u32;
    let mut out = Vec::new();
    out.extend_from_slice(&unit_length.to_le_bytes());
    out.extend_from_slice(&inner);
    out
}

fn write_elf64_header_with_shnum<W: Write>(
    w: &mut W,
    e_machine: u16,
    e_shnum: u16,
    e_shstrndx: u16,
) -> Result<(), RasError> {
    let mut e_ident = [0u8; 16];
    e_ident[0..4].copy_from_slice(&ELF_MAGIC);
    e_ident[4] = 2;
    e_ident[5] = 1;
    e_ident[6] = 1;
    w.write_all(&e_ident)
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&ET_REL.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&e_machine.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&1u32.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&0u64.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&0u64.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&(ELF64_HEADER_SIZE as u64).to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&0u32.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&64u16.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&0u16.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&0u16.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&64u16.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&e_shnum.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&e_shstrndx.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
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

fn write_elf64_section_header_full<W: Write>(
    w: &mut W,
    header: &Elf64SectionHeader,
) -> Result<(), RasError> {
    let map = |e: std::io::Error| RasError::ObjectError(format!("ELF write: {}", e));
    w.write_all(&header.name.to_le_bytes()).map_err(map)?;
    w.write_all(&header.section_type.to_le_bytes())
        .map_err(map)?;
    w.write_all(&header.flags.to_le_bytes()).map_err(map)?;
    w.write_all(&0u64.to_le_bytes()).map_err(map)?; // sh_addr
    w.write_all(&header.offset.to_le_bytes()).map_err(map)?;
    w.write_all(&header.size.to_le_bytes()).map_err(map)?;
    w.write_all(&header.link.to_le_bytes()).map_err(map)?;
    w.write_all(&header.info.to_le_bytes()).map_err(map)?;
    w.write_all(&header.addralign.to_le_bytes()).map_err(map)?;
    w.write_all(&header.entsize.to_le_bytes()).map_err(map)?;
    Ok(())
}

fn write_elf64_symbol<W: Write>(
    w: &mut W,
    name_offset: u32,
    bind: u8,
    sym_type: u8,
    hidden: bool,
    section_index: u16,
    value: u64,
) -> Result<(), RasError> {
    let map = |e: std::io::Error| RasError::ObjectError(format!("ELF write: {}", e));
    w.write_all(&name_offset.to_le_bytes()).map_err(map)?;
    w.write_all(&[(bind << 4) | (sym_type & 0xf)])
        .map_err(map)?;
    w.write_all(&[if hidden { 2 } else { 0 }]).map_err(map)?;
    w.write_all(&section_index.to_le_bytes()).map_err(map)?;
    w.write_all(&value.to_le_bytes()).map_err(map)?;
    w.write_all(&0u64.to_le_bytes()).map_err(map)?; // st_size
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

impl ObjectWriter for ElfWriter {
    fn write_object_file(
        &mut self,
        path: &std::path::Path,
        req: &ObjectWriteRequest<'_>,
    ) -> Result<(), RasError> {
        let machine = elf64_e_machine(req.target_arch)
            .ok_or_else(|| RasError::ObjectError("Unsupported ELF architecture".into()))?;
        let mut sections = req.sections.to_vec();
        if sections.is_empty() {
            sections.push(Section {
                group: None,
                name: ".text".into(),
                flags: crate::parser::SectionFlags {
                    alloc: true,
                    exec: true,
                    write: false,
                },
                data: req.code.to_vec(),
                alignment: 1,
            });
        }
        if req.target_arch == TargetArchitecture::X86_64 {
            for name in [".data", ".bss"] {
                if !sections.iter().any(|s| s.name == name) {
                    sections.push(Section {
                        group: None,
                        name: name.into(),
                        flags: crate::parser::SectionFlags {
                            alloc: true,
                            exec: false,
                            write: true,
                        },
                        data: Vec::new(),
                        alignment: 1,
                    });
                }
            }
        }
        let mut names: Vec<String> = sections.iter().map(|s| s.name.clone()).collect();
        let mut data: Vec<Vec<u8>> = sections.iter().map(|s| s.data.clone()).collect();
        let mut headers: Vec<Elf64SectionHeader> = sections
            .iter()
            .map(|s| Elf64SectionHeader {
                section_type: if s.name == ".bss" {
                    8
                } else if s.name.starts_with(".note.") && s.name != ".note.GNU-stack" {
                    7
                } else {
                    SHT_PROGBITS
                },
                flags: u64::from(s.flags.write)
                    | (u64::from(s.flags.alloc) << 1)
                    | (u64::from(s.flags.exec) << 2)
                    | if s.group.is_some() { 0x200 } else { 0 },
                addralign: s.alignment as u64,
                ..Default::default()
            })
            .collect();
        let mut strtab = vec![0];
        let mut symtab = vec![0; ELF64_SYM_SIZE as usize];
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
        let mut defined: Vec<_> = object_symbols.iter().collect();
        defined.sort_by_key(|s| s.global || s.attributes.weak);
        let first_global = 1 + defined
            .iter()
            .filter(|s| !s.global && !s.attributes.weak)
            .count() as u32;
        for sym in defined {
            let index = symtab.len() / ELF64_SYM_SIZE as usize;
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
            write_elf64_symbol(
                &mut symtab,
                name,
                if sym.attributes.weak {
                    2
                } else {
                    u8::from(sym.global)
                },
                sym.attributes.symbol_type,
                sym.attributes.hidden,
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
                symtab[end - 8..].copy_from_slice(&size.to_le_bytes());
            }
        }
        for reloc in req.relocations {
            if !sym_indices.contains_key(reloc.symbol.as_str()) {
                sym_indices.insert(
                    reloc.symbol.as_str(),
                    (symtab.len() / ELF64_SYM_SIZE as usize) as u32,
                );
                let name = strtab.len() as u32;
                strtab.extend_from_slice(reloc.symbol.as_bytes());
                strtab.push(0);
                write_elf64_symbol(&mut symtab, name, STB_GLOBAL, 0, false, 0, 0)?;
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
                bytes.extend_from_slice(&(r.offset as u64).to_le_bytes());
                let info = (u64::from(sym_indices[r.symbol.as_str()]) << 32) | u64::from(r.kind);
                bytes.extend_from_slice(&info.to_le_bytes());
                bytes.extend_from_slice(&r.addend.to_le_bytes());
            }
            if !bytes.is_empty() {
                if let Some(signature) = &section.group {
                    groups
                        .get_mut(signature.as_str())
                        .unwrap()
                        .push((headers.len() + 1) as u32);
                }
                rela_sections.push(headers.len());
                names.push(format!(".rela{}", section.name));
                data.push(bytes);
                headers.push(Elf64SectionHeader {
                    section_type: SHT_RELA,
                    flags: SHF_INFO_LINK | if section.group.is_some() { 0x200 } else { 0 },
                    info: (i + 1) as u32,
                    addralign: 8,
                    entsize: ELF64_RELA_SIZE,
                    ..Default::default()
                });
            }
        }
        let mut group_sections = Vec::new();
        for (signature, members) in groups {
            if !req.symbols.iter().any(|s| s.name == signature) {
                let offset = sym_indices[signature] as usize * ELF64_SYM_SIZE as usize + 6;
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
            addralign: 8,
            entsize: ELF64_SYM_SIZE,
            ..Default::default()
        });
        names.push(".strtab".into());
        data.push(strtab);
        headers.push(Elf64SectionHeader {
            section_type: SHT_STRTAB,
            addralign: 1,
            ..Default::default()
        });
        if req.opts.emit_minimal_dwarf_sections {
            for (name, bytes) in [
                (
                    ".debug_line",
                    dwarf_debug_line_bytes(&req.opts.dwarf_decl_file_name),
                ),
                (".debug_abbrev", dwarf_debug_abbrev_bytes()),
                (
                    ".debug_info",
                    dwarf_debug_info_bytes(&req.opts.dwarf_decl_file_name),
                ),
            ] {
                names.push(name.into());
                data.push(bytes);
                headers.push(Elf64SectionHeader {
                    section_type: SHT_PROGBITS,
                    addralign: 1,
                    ..Default::default()
                });
            }
        }
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
        let mut offset = ELF64_HEADER_SIZE + (headers.len() + 1) * ELF64_SECTION_HEADER_SIZE;
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
        let mut f =
            std::fs::File::create(path).map_err(|e| RasError::ObjectError(e.to_string()))?;
        write_elf64_header_with_shnum(
            &mut f,
            machine,
            (headers.len() + 1) as u16,
            headers.len() as u16,
        )?;
        write_elf64_section_header_full(&mut f, &Elf64SectionHeader::default())?;
        for header in headers {
            write_elf64_section_header_full(&mut f, &header)?;
        }
        f.write_all(&payload)
            .map_err(|e| RasError::ObjectError(e.to_string()))?;
        Ok(())
    }
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
    fn write_object_file(
        &mut self,
        _path: &std::path::Path,
        _req: &ObjectWriteRequest<'_>,
    ) -> Result<(), RasError> {
        Err(RasError::ObjectError(
            "Mach-O object file generation not yet implemented".to_string(),
        ))
    }
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

const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_ALIGN_16BYTES: u32 = 0x0050_0000;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SYM_CLASS_STATIC: u8 = 3;

const IMAGE_REL_AMD64_REL32: u16 = 4;

impl ObjectWriter for CoffWriter {
    fn write_object_file(
        &mut self,
        path: &std::path::Path,
        req: &ObjectWriteRequest<'_>,
    ) -> Result<(), RasError> {
        let code = req.code;
        let symbols = req.symbols;
        let relocations = req.relocations;
        if req.target_arch != TargetArchitecture::X86_64 {
            return Err(RasError::ObjectError(format!(
                "COFF writer: unsupported architecture {:?}",
                req.target_arch
            )));
        }

        // Collect unique external symbols referenced by relocations.
        let mut extern_names: Vec<String> = Vec::new();
        for r in relocations {
            if !extern_names.contains(&r.symbol) {
                extern_names.push(r.symbol.clone());
            }
        }

        // Collect global exported functions (defined in this object).
        let exported: Vec<&ObjectSymbol> = symbols
            .iter()
            .filter(|s| s.global && !s.name.starts_with(".L"))
            .collect();

        // Build symbol table and string table.
        // Symbol index layout: 0 = .text section, 1..=exported, then extern symbols.
        let mut strtab: Vec<u8> = vec![0u8; 4]; // size placeholder
        let mut sym_bytes: Vec<u8> = Vec::new();

        let push_sym = |sym_bytes: &mut Vec<u8>,
                        strtab: &mut Vec<u8>,
                        name: &str,
                        value: u32,
                        section: i16,
                        ty: u16,
                        storage: u8| {
            let mut entry = [0u8; 18];
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
            entry[16] = storage;
            sym_bytes.extend_from_slice(&entry);
        };

        // Index 0: .text section symbol
        push_sym(
            &mut sym_bytes,
            &mut strtab,
            ".text",
            0,
            1,
            0,
            IMAGE_SYM_CLASS_STATIC,
        );
        // Indices 1..=exported: defined global functions
        for sym in &exported {
            push_sym(
                &mut sym_bytes,
                &mut strtab,
                &sym.name,
                sym.value as u32,
                1,
                0x20,
                2, /* EXTERNAL */
            );
        }
        // Indices after exported: undefined external symbols (printf, etc.)
        let extern_base_idx = 1 + exported.len(); // 0-based index of first extern sym
        for name in &extern_names {
            push_sym(&mut sym_bytes, &mut strtab, name, 0, 0, 0x20, 2);
        }

        let strtab_size = strtab.len() as u32;
        strtab[0..4].copy_from_slice(&strtab_size.to_le_bytes());
        let total_syms = (1 + exported.len() + extern_names.len()) as u32;

        // Build COFF relocation entries (10 bytes each).
        let mut reloc_bytes: Vec<u8> = Vec::new();
        for r in relocations {
            let sym_idx = extern_base_idx
                + extern_names
                    .iter()
                    .position(|n| *n == r.symbol)
                    .unwrap_or(0);
            reloc_bytes.extend_from_slice(&(r.offset as u32).to_le_bytes()); // VirtualAddress
            reloc_bytes.extend_from_slice(&(sym_idx as u32).to_le_bytes()); // SymbolTableIndex
            reloc_bytes.extend_from_slice(&IMAGE_REL_AMD64_REL32.to_le_bytes()); // Type
        }
        let n_relocs = relocations.len() as u16;

        let hdr_sz = 20usize;
        let sec_hdr_sz = 40usize;
        let align = 16usize;
        let padded_len = (code.len().div_ceil(align) * align) as u32;
        let raw_data_off = hdr_sz + sec_hdr_sz;
        let reloc_off = raw_data_off + padded_len as usize;
        let sym_off = reloc_off + reloc_bytes.len();

        let text_name = b".text\0\0\0";
        let sec_flags = IMAGE_SCN_CNT_CODE
            | IMAGE_SCN_ALIGN_16BYTES
            | IMAGE_SCN_MEM_EXECUTE
            | IMAGE_SCN_MEM_READ;

        let map = |e: std::io::Error| RasError::ObjectError(format!("COFF write: {}", e));
        let mut f = std::fs::File::create(path)
            .map_err(|e| RasError::ObjectError(format!("Failed to create COFF file: {}", e)))?;

        // COFF file header (20 bytes)
        f.write_all(&IMAGE_FILE_MACHINE_AMD64.to_le_bytes())
            .map_err(map)?;
        f.write_all(&1u16.to_le_bytes()).map_err(map)?;
        f.write_all(&0u32.to_le_bytes()).map_err(map)?;
        f.write_all(&(sym_off as u32).to_le_bytes()).map_err(map)?;
        f.write_all(&total_syms.to_le_bytes()).map_err(map)?;
        f.write_all(&0u16.to_le_bytes()).map_err(map)?;
        f.write_all(&0u16.to_le_bytes()).map_err(map)?;

        // .text section header (40 bytes)
        f.write_all(text_name).map_err(map)?;
        f.write_all(&0u32.to_le_bytes()).map_err(map)?; // VirtualSize
        f.write_all(&0u32.to_le_bytes()).map_err(map)?; // VirtualAddress
        f.write_all(&padded_len.to_le_bytes()).map_err(map)?; // SizeOfRawData
        f.write_all(&(raw_data_off as u32).to_le_bytes())
            .map_err(map)?; // PointerToRawData
        if n_relocs > 0 {
            f.write_all(&(reloc_off as u32).to_le_bytes())
                .map_err(map)?; // PointerToRelocations
        } else {
            f.write_all(&0u32.to_le_bytes()).map_err(map)?;
        }
        f.write_all(&0u32.to_le_bytes()).map_err(map)?; // PointerToLinenumbers
        f.write_all(&n_relocs.to_le_bytes()).map_err(map)?; // NumberOfRelocations
        f.write_all(&0u16.to_le_bytes()).map_err(map)?; // NumberOfLinenumbers
        f.write_all(&sec_flags.to_le_bytes()).map_err(map)?; // Characteristics

        // Raw data
        f.write_all(code).map_err(map)?;
        let pad = padded_len as usize - code.len();
        if pad > 0 {
            f.write_all(&vec![0u8; pad]).map_err(map)?;
        }

        // Relocation table
        if !reloc_bytes.is_empty() {
            f.write_all(&reloc_bytes).map_err(map)?;
        }

        // Symbol table + string table
        f.write_all(&sym_bytes).map_err(map)?;
        f.write_all(&strtab).map_err(map)?;

        Ok(())
    }
}

/// Returns the object writer for `target_os` (ELF, Mach-O, or COFF).
///
/// This is the single dispatch point used by [`crate::assembler::RasAssembler`];
/// add new OS families here instead of duplicating `match` arms.
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
    use super::*;

    #[test]
    fn test_elf64_x86_64_write() {
        let mut w = ElfWriter::new();
        let code = [0xc3u8];
        let sections = vec![Section {
            group: None,
            name: ".text".to_string(),
            data: code.to_vec(),
            alignment: 1,
            flags: SectionFlags {
                alloc: true,
                exec: true,
                write: false,
            },
        }];
        let symbols: Vec<ObjectSymbol> = vec![];
        let path = std::env::temp_dir().join("ras_elf_test_x86_64.o");
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
        let _ = std::fs::remove_file(&path);
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
            flags: SectionFlags {
                alloc: true,
                exec: true,
                write: false,
            },
        }];
        let symbols: Vec<ObjectSymbol> = vec![];
        let path = std::env::temp_dir().join("ras_elf_test_aarch64.o");
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
        let _ = std::fs::remove_file(&path);
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
        let path = std::env::temp_dir().join("ras_elf_magic_test.o");
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
        let _ = std::fs::remove_file(&path);
        assert!(buf.len() >= 64);
        assert_eq!(&buf[0..4], &[0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn test_elf64_emits_debug_sections_when_requested() {
        let mut w = ElfWriter::new();
        let code = [0xc3u8];
        let path = std::env::temp_dir().join("ras_elf_dwarf_test.o");
        let opts = ObjectWriteOptions {
            emit_minimal_dwarf_sections: true,
            dwarf_decl_file_name: "t.s".to_string(),
        };
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
        let _ = std::fs::remove_file(&path);
        assert!(
            buf.windows(11).any(|w| w == b".debug_line"),
            "expected .debug_line in shstrtab"
        );
    }

    #[test]
    fn test_coff_amd64_write() {
        let mut w = CoffWriter::new();
        let code = [0xc3u8];
        let path = std::env::temp_dir().join("ras_coff_test.obj");
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
        let _ = std::fs::remove_file(&path);
        assert_eq!(&buf[0..2], &IMAGE_FILE_MACHINE_AMD64.to_le_bytes());
    }
}
