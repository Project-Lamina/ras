//! Object file generation: ELF (Linux/BSD), Mach-O (macOS), COFF/PE (Windows).

#[cfg(test)]
use crate::parser::SectionFlags;
use crate::{error::RasError, parser::Section};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::collections::HashMap;
use std::io::Write;

/// Extra sections and metadata for [`ObjectWriter`] backends.
#[derive(Debug, Clone)]
pub struct ObjectWriteOptions {
    /// When true, [`ElfWriter`] also emits minimal `.debug_line`, `.debug_abbrev`, and `.debug_info`.
    pub emit_minimal_dwarf_sections: bool,
    /// Retain temporary local symbols in ELF output.
    pub keep_locals: bool,
    /// File name stored in DWARF line/info (used when `emit_minimal_dwarf_sections` is set).
    pub dwarf_decl_file_name: String,
}

impl Default for ObjectWriteOptions {
    fn default() -> Self {
        Self {
            emit_minimal_dwarf_sections: false,
            keep_locals: false,
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
    /// x86_64: bytes that follow the 4-byte displacement in the instruction (an immediate).
    pub trailing: usize,
    /// x86_64: the reference is a call or jump target, encoded as a Mach-O branch.
    pub branch: bool,
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
        let merge_relocs: std::collections::HashSet<_> = req
            .relocations
            .iter()
            .filter(|r| {
                !matches!(
                    (req.target_arch, r.kind),
                    (TargetArchitecture::X86_64, 1 | 10 | 11 | 12 | 14)
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
            let index = symtab.len() / ELF64_SYM_SIZE as usize;
            section_symbols.insert(symbol.section.as_str(), index as u32);
            write_elf64_symbol(&mut symtab, 0, 0, 3, false, section as u16, 0)?;
        }
        let mut defined: Vec<_> = object_symbols
            .iter()
            .filter(|s| !omitted.contains_key(s.name.as_str()))
            .collect();
        defined.sort_by_key(|s| s.global || s.attributes.weak);
        let first_global = (symtab.len() / ELF64_SYM_SIZE as usize) as u32
            + defined
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
            if !sym_indices.contains_key(reloc.symbol.as_str())
                && !omitted.contains_key(reloc.symbol.as_str())
            {
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
                let (index, addend) = if let Some(symbol) = omitted.get(r.symbol.as_str()) {
                    (
                        section_symbols[symbol.section.as_str()],
                        r.addend.wrapping_add(symbol.value as i64),
                    )
                } else {
                    (sym_indices[r.symbol.as_str()], r.addend)
                };
                let info = (u64::from(index) << 32) | u64::from(r.kind);
                bytes.extend_from_slice(&info.to_le_bytes());
                bytes.extend_from_slice(&addend.to_le_bytes());
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
const S_ZEROFILL: u32 = 0x1;
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

const NLIST_GROUP_LOCAL: u8 = 0;
const NLIST_GROUP_EXTDEF: u8 = 1;
const NLIST_GROUP_UNDEF: u8 = 2;

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
    segment: &'static str,
    name: &'static str,
    flags: u32,
    align: u64,
    addr: u64,
    data: Vec<u8>,
    relocs: Vec<(u32, u32)>,
}

impl MachOSection {
    fn is_zerofill(&self) -> bool {
        self.flags & 0xff == S_ZEROFILL
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
    let (cputype, cpusubtype) = match req.target_arch {
        TargetArchitecture::X86_64 => (CPU_TYPE_X86_64, CPU_SUBTYPE_X86_64_ALL),
        other => {
            return Err(RasError::ObjectError(format!(
                "Mach-O objects are written for x86_64 only, not {other:?}"
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
            },
            data: req.code.to_vec(),
            alignment: 1,
        });
    }

    let mut sections = Vec::with_capacity(source.len());
    let mut section_ord: HashMap<&str, usize> = HashMap::new();
    let mut addr = 0u64;
    for s in &source {
        let (segment, name, flags) = macho_section_kind(&s.name).ok_or_else(|| {
            RasError::ObjectError(format!(
                "Mach-O has no section for {}; supported sections are .text, .rodata, .data and .bss",
                s.name
            ))
        })?;
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
        let zerofill = flags & 0xff == S_ZEROFILL;
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
                    if sym.attributes.hidden {
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
        let spec = x86_64_reloc_spec(reloc.kind, reloc.addend, reloc.trailing, reloc.branch)?;
        // clang stores addend minus the immediate width, so the content bias stays 4.
        let bias = if spec.pcrel { 4 } else { 0 };
        let field = reloc.offset as u64;
        let field_addr = sections[ord].addr + field;
        let width = 1usize << spec.width_log2;
        let addend = reloc.addend + bias;

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
        let info = (symbol_num & 0x00ff_ffff)
            | (u32::from(spec.pcrel) << 24)
            | (spec.width_log2 << 25)
            | (u32::from(is_extern) << 27)
            | (spec.r_type << 28);
        let r_address = u32::try_from(reloc.offset)
            .map_err(|_| RasError::ObjectError("relocation offset too large".into()))?;
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
        put_name16(&mut out, sec.name);
        put_name16(&mut out, sec.segment);
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
    fn write_object_file(
        &mut self,
        path: &std::path::Path,
        req: &ObjectWriteRequest<'_>,
    ) -> Result<(), RasError> {
        let bytes = macho_object_bytes(req)?;
        std::fs::write(path, bytes)
            .map_err(|e| RasError::ObjectError(format!("Mach-O write: {}", e)))
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

        let mut extern_names: Vec<String> = Vec::new();
        for r in relocations {
            if !extern_names.contains(&r.symbol) {
                extern_names.push(r.symbol.clone());
            }
        }

        let exported: Vec<&ObjectSymbol> = symbols
            .iter()
            .filter(|s| s.global && !s.name.starts_with(".L"))
            .collect();

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

        push_sym(
            &mut sym_bytes,
            &mut strtab,
            ".text",
            0,
            1,
            0,
            IMAGE_SYM_CLASS_STATIC,
        );
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
        let extern_base_idx = 1 + exported.len();
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

        f.write_all(code).map_err(map)?;
        let pad = padded_len as usize - code.len();
        if pad > 0 {
            f.write_all(&vec![0u8; pad]).map_err(map)?;
        }

        if !reloc_bytes.is_empty() {
            f.write_all(&reloc_bytes).map_err(map)?;
        }

        f.write_all(&sym_bytes).map_err(map)?;
        f.write_all(&strtab).map_err(map)?;

        Ok(())
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
            ..Default::default()
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

    fn macho_text_section(data: Vec<u8>) -> Section {
        Section {
            group: None,
            name: ".text".into(),
            flags: SectionFlags {
                alloc: true,
                exec: true,
                write: false,
            },
            data,
            alignment: 16,
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
        };
        let err = macho_object(&[macho_text_section(vec![0; 8])], &[], &[pc64]).expect_err("pc64");
        assert!(err.to_string().contains("kind 24"), "{err}");
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
