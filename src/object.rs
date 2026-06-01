//! Object file generation: ELF (Linux/BSD), Mach-O (macOS), COFF/PE (Windows).

use crate::error::RasError;
#[cfg(test)]
use crate::parser::SectionFlags;
use crate::parser::Section;
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
const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;
const SHF_INFO_LINK: u64 = 0x40;
const ELF64_SYM_SIZE: u64 = 24;
const ELF64_RELA_SIZE: u64 = 24;
const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STT_FUNC: u8 = 2;
const TEXT_SECTION_INDEX: u16 = 1;
const R_X86_64_PLT32: u64 = 4;
const R_AARCH64_CALL26: u64 = 283;

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

fn write_elf64_section_header<W: Write>(
    w: &mut W,
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_addralign: u64,
) -> Result<(), RasError> {
    w.write_all(&sh_name.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&sh_type.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&sh_flags.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&0u64.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&sh_offset.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&sh_size.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&0u32.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&0u32.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&sh_addralign.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&0u64.to_le_bytes())
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
    w.write_all(&header.section_type.to_le_bytes()).map_err(map)?;
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
    section_index: u16,
    value: u64,
) -> Result<(), RasError> {
    let map = |e: std::io::Error| RasError::ObjectError(format!("ELF write: {}", e));
    w.write_all(&name_offset.to_le_bytes()).map_err(map)?;
    w.write_all(&[(bind << 4) | (sym_type & 0xf)]).map_err(map)?;
    w.write_all(&[0u8]).map_err(map)?; // st_other
    w.write_all(&section_index.to_le_bytes()).map_err(map)?;
    w.write_all(&value.to_le_bytes()).map_err(map)?;
    w.write_all(&0u64.to_le_bytes()).map_err(map)?; // st_size
    Ok(())
}

/// Symbols defined in `.text` that should appear in the object symbol table.
/// Assembler-internal labels (`.L*`) are dropped, matching gas behavior.
/// Locals are returned before globals so the symbol table satisfies the ELF
/// requirement that `sh_info` marks the first global symbol.
fn linkable_text_symbols(symbols: &[ObjectSymbol]) -> (Vec<&ObjectSymbol>, Vec<&ObjectSymbol>) {
    let mut locals = Vec::new();
    let mut globals = Vec::new();
    for symbol in symbols {
        if symbol.section != ".text" || symbol.name.starts_with(".L") {
            continue;
        }
        if symbol.global {
            globals.push(symbol);
        } else {
            locals.push(symbol);
        }
    }
    (locals, globals)
}

/// Builds `.symtab` bytes (null entry + locals + globals) and `.strtab`,
/// also appending undefined external symbols for cross-object relocations.
///
/// Returns `(symtab_bytes, strtab_bytes, first_global_index, extern_base_index)`.
///
/// - `first_global_index`: symtab index of the first `STB_GLOBAL` entry;
///   used as `sh_info` on the symtab section header.
/// - `extern_base_index`: symtab index of the first `SHN_UNDEF` external entry;
///   used when building `.rela.text` `r_sym` fields.
fn build_symtab_with_externals(
    symbols: &[ObjectSymbol],
    extern_names: &[&str],
) -> (Vec<u8>, Vec<u8>, u32, u32) {
    let (locals, globals) = linkable_text_symbols(symbols);

    let mut strtab = vec![0u8];
    let mut symtab = vec![0u8; ELF64_SYM_SIZE as usize]; // null symbol

    // Dedup extern names while preserving order
    let mut seen = std::collections::HashSet::new();
    let unique_externs: Vec<&str> = extern_names
        .iter()
        .copied()
        .filter(|&n| seen.insert(n))
        .collect();

    // Local defined symbols
    for sym in &locals {
        let name_off = strtab.len() as u32;
        strtab.extend_from_slice(sym.name.as_bytes());
        strtab.push(0);
        let mut e = Vec::new();
        let _ = write_elf64_symbol(&mut e, name_off, STB_LOCAL, STT_FUNC, TEXT_SECTION_INDEX, sym.value);
        symtab.extend_from_slice(&e);
    }

    let first_global_index = 1 + locals.len() as u32;

    // Global defined symbols
    for sym in &globals {
        let name_off = strtab.len() as u32;
        strtab.extend_from_slice(sym.name.as_bytes());
        strtab.push(0);
        let mut e = Vec::new();
        let _ = write_elf64_symbol(&mut e, name_off, STB_GLOBAL, STT_FUNC, TEXT_SECTION_INDEX, sym.value);
        symtab.extend_from_slice(&e);
    }

    let extern_base_index = first_global_index + globals.len() as u32;

    // Undefined external symbols (STB_GLOBAL, SHN_UNDEF=0, value=0)
    for name in &unique_externs {
        let name_off = strtab.len() as u32;
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
        let mut e = Vec::new();
        let _ = write_elf64_symbol(&mut e, name_off, STB_GLOBAL, 0, 0, 0); // SHN_UNDEF = 0
        symtab.extend_from_slice(&e);
    }

    (symtab, strtab, first_global_index, extern_base_index)
}

/// Build the raw bytes for a `.rela.text` section.
fn build_rela_text(
    relocations: &[ExternalReloc],
    extern_names: &[&str],
    extern_base_index: u32,
    e_machine: u16,
) -> Vec<u8> {
    // Deduplicated extern name → symbol table index
    let mut seen: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut next_idx = extern_base_index;
    for name in extern_names {
        seen.entry(name).or_insert_with(|| {
            let idx = next_idx;
            next_idx += 1;
            idx
        });
    }

    let rel_type: u64 = match e_machine {
        183 => R_AARCH64_CALL26, // EM_AARCH64
        _ => R_X86_64_PLT32,
    };

    let mut out = Vec::with_capacity(relocations.len() * 24);
    for reloc in relocations {
        let sym_idx = *seen.get(reloc.symbol.as_str()).unwrap_or(&0) as u64;
        let r_info = (sym_idx << 32) | rel_type;
        out.extend_from_slice(&(reloc.offset as u64).to_le_bytes()); // r_offset
        out.extend_from_slice(&r_info.to_le_bytes());                // r_info
        out.extend_from_slice(&(-4i64).to_le_bytes());               // r_addend = -4
    }
    out
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
        let code = req.code;
        let symbols = req.symbols;
        let relocations = req.relocations;
        let opts = req.opts;
        let e_machine = elf64_e_machine(req.target_arch).ok_or_else(|| {
            RasError::ObjectError(format!("ELF does not support arch: {:?}", req.target_arch))
        })?;

        let mut f = std::fs::File::create(path)
            .map_err(|e| RasError::ObjectError(format!("Failed to create ELF file: {}", e)))?;

        if opts.emit_minimal_dwarf_sections {
            const SHSTRTAB_DWARF: &[u8] =
                b"\0.text\0.debug_line\0.debug_abbrev\0.debug_info\0.shstrtab\0";
            let line_b = dwarf_debug_line_bytes(opts.dwarf_decl_file_name.as_str());
            let abbrev_b = dwarf_debug_abbrev_bytes();
            let info_b = dwarf_debug_info_bytes(opts.dwarf_decl_file_name.as_str());

            let n_sec = 6u16;
            let shoff = ELF64_HEADER_SIZE as u64;
            let first_data = shoff + u64::from(n_sec) * ELF64_SECTION_HEADER_SIZE as u64;
            let text_align = 16u64;
            let text_size_aligned = (code.len() as u64 + text_align - 1) & !(text_align - 1);
            let mut cur = first_data + text_size_aligned;
            let line_off = cur;
            cur += line_b.len() as u64;
            let abbrev_off = cur;
            cur += abbrev_b.len() as u64;
            let info_off = cur;
            cur += info_b.len() as u64;
            let shstrtab_off = cur;
            let shstrtab_size = SHSTRTAB_DWARF.len() as u64;

            write_elf64_header_with_shnum(&mut f, e_machine, n_sec, 5)?;
            write_elf64_section_header(&mut f, 0, SHT_NULL, 0, 0, 0, 0)?;
            write_elf64_section_header(
                &mut f,
                1,
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                first_data,
                code.len() as u64,
                text_align,
            )?;
            write_elf64_section_header(
                &mut f,
                7,
                SHT_PROGBITS,
                0,
                line_off,
                line_b.len() as u64,
                1,
            )?;
            write_elf64_section_header(
                &mut f,
                19,
                SHT_PROGBITS,
                0,
                abbrev_off,
                abbrev_b.len() as u64,
                1,
            )?;
            write_elf64_section_header(
                &mut f,
                33,
                SHT_PROGBITS,
                0,
                info_off,
                info_b.len() as u64,
                1,
            )?;
            write_elf64_section_header(&mut f, 45, SHT_STRTAB, 0, shstrtab_off, shstrtab_size, 1)?;

            let pad = text_size_aligned as usize - code.len();
            f.write_all(code)
                .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
            if pad > 0 {
                f.write_all(&vec![0u8; pad])
                    .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
            }
            f.write_all(&line_b)
                .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
            f.write_all(&abbrev_b)
                .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
            f.write_all(&info_b)
                .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
            f.write_all(SHSTRTAB_DWARF)
                .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
            return Ok(());
        }

        // Build symtab including undefined external symbols (so weld/ld can resolve them).
        let extern_names: Vec<&str> = relocations.iter().map(|r| r.symbol.as_str()).collect();
        let (symtab_bytes, strtab_bytes, first_global_index, extern_sym_base) =
            build_symtab_with_externals(symbols, &extern_names);

        // Build .rela.text entries for each external relocation.
        // R_X86_64_PLT32 / R_AARCH64_CALL26 with addend = -4.
        let rela_bytes = if !relocations.is_empty() {
            build_rela_text(relocations, &extern_names, extern_sym_base, e_machine)
        } else {
            Vec::new()
        };
        let has_rela = !rela_bytes.is_empty();

        // Section layout:
        //   Without rela: [0] NULL  [1] .text  [2] .symtab  [3] .strtab  [4] .shstrtab
        //   With rela:    [0] NULL  [1] .text  [2] .rela.text  [3] .symtab  [4] .strtab  [5] .shstrtab
        const SHSTRTAB_NO_RELA: &[u8] = b"\0.text\0.symtab\0.strtab\0.shstrtab\0";
        const SHSTRTAB_RELA: &[u8] =
            b"\0.text\0.rela.text\0.symtab\0.strtab\0.shstrtab\0";

        let (shstrtab, name_text, name_rela_text, name_symtab, name_strtab, name_shstrtab,
             section_count, shstrtab_index, symtab_section_index, strtab_section_index) =
        if has_rela {
            (SHSTRTAB_RELA,
             1u32, 7u32, 18u32, 26u32, 34u32,
             6u16, 5u16, 3u32, 4u32)
        } else {
            (SHSTRTAB_NO_RELA,
             1u32, 0u32, 7u32, 15u32, 23u32,
             5u16, 4u16, 2u32, 3u32)
        };

        let shoff = ELF64_HEADER_SIZE as u64;
        let text_offset = shoff + u64::from(section_count) * ELF64_SECTION_HEADER_SIZE as u64;
        let text_align = 16u64;
        let text_size_aligned = (code.len() as u64 + text_align - 1) & !(text_align - 1);
        let rela_offset = text_offset + text_size_aligned;
        let rela_size = rela_bytes.len() as u64;
        let symtab_offset = rela_offset + rela_size;
        let strtab_offset = symtab_offset + symtab_bytes.len() as u64;
        let shstrtab_offset = strtab_offset + strtab_bytes.len() as u64;

        write_elf64_header_with_shnum(&mut f, e_machine, section_count, shstrtab_index)?;
        // [0] NULL
        write_elf64_section_header(&mut f, 0, SHT_NULL, 0, 0, 0, 0)?;
        // [1] .text
        write_elf64_section_header(
            &mut f,
            name_text,
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            text_offset,
            code.len() as u64,
            text_align,
        )?;
        // [2] .rela.text (only when needed)
        if has_rela {
            write_elf64_section_header_full(
                &mut f,
                &Elf64SectionHeader {
                    name: name_rela_text,
                    section_type: SHT_RELA,
                    flags: SHF_INFO_LINK,
                    offset: rela_offset,
                    size: rela_size,
                    link: symtab_section_index,    // points to .symtab
                    info: TEXT_SECTION_INDEX as u32, // relocates .text
                    addralign: 8,
                    entsize: ELF64_RELA_SIZE,
                },
            )?;
        }
        // .symtab
        write_elf64_section_header_full(
            &mut f,
            &Elf64SectionHeader {
                name: name_symtab,
                section_type: SHT_SYMTAB,
                offset: symtab_offset,
                size: symtab_bytes.len() as u64,
                link: strtab_section_index,
                info: first_global_index,
                addralign: 8,
                entsize: ELF64_SYM_SIZE,
                ..Default::default()
            },
        )?;
        // .strtab
        write_elf64_section_header_full(
            &mut f,
            &Elf64SectionHeader {
                name: name_strtab,
                section_type: SHT_STRTAB,
                offset: strtab_offset,
                size: strtab_bytes.len() as u64,
                addralign: 1,
                ..Default::default()
            },
        )?;
        // .shstrtab
        write_elf64_section_header_full(
            &mut f,
            &Elf64SectionHeader {
                name: name_shstrtab,
                section_type: SHT_STRTAB,
                offset: shstrtab_offset,
                size: shstrtab.len() as u64,
                addralign: 1,
                ..Default::default()
            },
        )?;

        let map = |e: std::io::Error| RasError::ObjectError(format!("ELF write: {}", e));
        let padding = text_size_aligned as usize - code.len();
        f.write_all(code).map_err(map)?;
        if padding > 0 {
            f.write_all(&vec![0u8; padding]).map_err(map)?;
        }
        if has_rela {
            f.write_all(&rela_bytes).map_err(map)?;
        }
        f.write_all(&symtab_bytes).map_err(map)?;
        f.write_all(&strtab_bytes).map_err(map)?;
        f.write_all(shstrtab).map_err(map)?;

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

        let push_sym = |sym_bytes: &mut Vec<u8>, strtab: &mut Vec<u8>,
                             name: &str, value: u32, section: i16,
                             ty: u16, storage: u8| {
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
        push_sym(&mut sym_bytes, &mut strtab, ".text", 0, 1, 0, IMAGE_SYM_CLASS_STATIC);
        // Indices 1..=exported: defined global functions
        for sym in &exported {
            push_sym(&mut sym_bytes, &mut strtab, &sym.name,
                sym.value as u32, 1, 0x20, 2 /* EXTERNAL */);
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
                + extern_names.iter().position(|n| *n == r.symbol).unwrap_or(0);
            reloc_bytes.extend_from_slice(&(r.offset as u32).to_le_bytes()); // VirtualAddress
            reloc_bytes.extend_from_slice(&(sym_idx as u32).to_le_bytes());  // SymbolTableIndex
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
        f.write_all(&IMAGE_FILE_MACHINE_AMD64.to_le_bytes()).map_err(map)?;
        f.write_all(&1u16.to_le_bytes()).map_err(map)?;
        f.write_all(&0u32.to_le_bytes()).map_err(map)?;
        f.write_all(&(sym_off as u32).to_le_bytes()).map_err(map)?;
        f.write_all(&total_syms.to_le_bytes()).map_err(map)?;
        f.write_all(&0u16.to_le_bytes()).map_err(map)?;
        f.write_all(&0u16.to_le_bytes()).map_err(map)?;

        // .text section header (40 bytes)
        f.write_all(text_name).map_err(map)?;
        f.write_all(&0u32.to_le_bytes()).map_err(map)?;          // VirtualSize
        f.write_all(&0u32.to_le_bytes()).map_err(map)?;          // VirtualAddress
        f.write_all(&padded_len.to_le_bytes()).map_err(map)?;    // SizeOfRawData
        f.write_all(&(raw_data_off as u32).to_le_bytes()).map_err(map)?; // PointerToRawData
        if n_relocs > 0 {
            f.write_all(&(reloc_off as u32).to_le_bytes()).map_err(map)?; // PointerToRelocations
        } else {
            f.write_all(&0u32.to_le_bytes()).map_err(map)?;
        }
        f.write_all(&0u32.to_le_bytes()).map_err(map)?;          // PointerToLinenumbers
        f.write_all(&n_relocs.to_le_bytes()).map_err(map)?;      // NumberOfRelocations
        f.write_all(&0u16.to_le_bytes()).map_err(map)?;          // NumberOfLinenumbers
        f.write_all(&sec_flags.to_le_bytes()).map_err(map)?;     // Characteristics

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
        | TargetOperatingSystem::Redox => Ok(Box::new(ElfWriter::new())),
        TargetOperatingSystem::MacOS => Ok(Box::new(MachOWriter::new())),
        TargetOperatingSystem::Windows => Ok(Box::new(CoffWriter::new())),
        _ => Err(RasError::UnsupportedTarget(format!(
            "No object file format is mapped for operating system {:?}",
            target_os
        ))),
    }
}

#[cfg(test)]
fn build_symtab_and_strtab(symbols: &[ObjectSymbol]) -> (Vec<u8>, Vec<u8>, u32) {
    let (symtab, strtab, first_global, _) = build_symtab_with_externals(symbols, &[]);
    (symtab, strtab, first_global)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symtab_includes_globals_and_skips_local_temporaries() {
        let symbols = vec![
            ObjectSymbol {
                name: "main".to_string(),
                global: true,
                section: ".text".to_string(),
                value: 0,
            },
            ObjectSymbol {
                name: ".L_main_entry".to_string(),
                global: false,
                section: ".text".to_string(),
                value: 16,
            },
            ObjectSymbol {
                name: "helper".to_string(),
                global: false,
                section: ".text".to_string(),
                value: 32,
            },
        ];

        let (symtab, strtab, first_global) = build_symtab_and_strtab(&symbols);

        // Null entry + one local (helper) + one global (main); `.L` is dropped.
        assert_eq!(symtab.len(), 3 * ELF64_SYM_SIZE as usize);
        assert_eq!(first_global, 2);
        assert!(strtab.windows(4).any(|w| w == b"main"));
        assert!(strtab.windows(6).any(|w| w == b"helper"));
        assert!(!strtab.windows(2).any(|w| w == b".L"));
    }

    #[test]
    fn symtab_drops_symbols_outside_text() {
        let symbols = vec![ObjectSymbol {
            name: "data_label".to_string(),
            global: true,
            section: ".data".to_string(),
            value: 0,
        }];

        let (symtab, _strtab, first_global) = build_symtab_and_strtab(&symbols);

        assert_eq!(symtab.len(), ELF64_SYM_SIZE as usize);
        assert_eq!(first_global, 1);
    }

    #[test]
    fn test_elf64_x86_64_write() {
        let mut w = ElfWriter::new();
        let code = [0xc3u8];
        let sections = vec![Section {
            name: ".text".to_string(),
            flags: SectionFlags {
                alloc: true,
                exec: true,
                write: false,
            },
        }];
        let symbols: Vec<ObjectSymbol> = vec![];
        let path = std::env::temp_dir().join("ras_elf_test_x86_64.o");
        let opts = ObjectWriteOptions::default();
        let result = w.write_object_file(&path, &ObjectWriteRequest {
            code: &code, sections: &sections, symbols: &symbols, relocations: &[],
            target_arch: TargetArchitecture::X86_64,
            target_os: TargetOperatingSystem::Linux,
            opts: &opts,
        });
        let _ = std::fs::remove_file(&path);
        result.expect("ELF write should succeed");
    }

    #[test]
    fn test_elf64_aarch64_write() {
        let mut w = ElfWriter::new();
        let code = [0xc0, 0x03, 0x5f, 0xd6];
        let sections = vec![Section {
            name: ".text".to_string(),
            flags: SectionFlags { alloc: true, exec: true, write: false },
        }];
        let symbols: Vec<ObjectSymbol> = vec![];
        let path = std::env::temp_dir().join("ras_elf_test_aarch64.o");
        let opts = ObjectWriteOptions::default();
        let result = w.write_object_file(&path, &ObjectWriteRequest {
            code: &code, sections: &sections, symbols: &symbols, relocations: &[],
            target_arch: TargetArchitecture::Aarch64,
            target_os: TargetOperatingSystem::Linux,
            opts: &opts,
        });
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
        w.write_object_file(&path, &ObjectWriteRequest {
            code: &code, sections: &[], symbols: &[], relocations: &[],
            target_arch: TargetArchitecture::X86_64,
            target_os: TargetOperatingSystem::Linux,
            opts: &opts,
        }).expect("write");
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
        w.write_object_file(&path, &ObjectWriteRequest {
            code: &code, sections: &[], symbols: &[], relocations: &[],
            target_arch: TargetArchitecture::X86_64,
            target_os: TargetOperatingSystem::Linux,
            opts: &opts,
        }).expect("write");
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
        w.write_object_file(&path, &ObjectWriteRequest {
            code: &code, sections: &[], symbols: &[], relocations: &[],
            target_arch: TargetArchitecture::X86_64,
            target_os: TargetOperatingSystem::Windows,
            opts: &opts,
        }).expect("coff write");
        let buf = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert_eq!(&buf[0..2], &IMAGE_FILE_MACHINE_AMD64.to_le_bytes());
    }
}
