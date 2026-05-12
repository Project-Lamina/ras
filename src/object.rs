//! Object file generation: ELF (Linux/BSD), Mach-O (macOS), COFF/PE (Windows).

use crate::error::RasError;
#[cfg(test)]
use crate::parser::SectionFlags;
use crate::parser::{Section, Symbol};
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

pub trait ObjectWriter {
    #[allow(clippy::too_many_arguments)]
    fn write_object_file(
        &mut self,
        path: &std::path::Path,
        code: &[u8],
        sections: &[Section],
        symbols: &[Symbol],
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
        opts: &ObjectWriteOptions,
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
const SHT_STRTAB: u32 = 3;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;

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
        code: &[u8],
        _sections: &[Section],
        _symbols: &[Symbol],
        target_arch: TargetArchitecture,
        _target_os: TargetOperatingSystem,
        opts: &ObjectWriteOptions,
    ) -> Result<(), RasError> {
        let e_machine = elf64_e_machine(target_arch).ok_or_else(|| {
            RasError::ObjectError(format!("ELF does not support arch: {:?}", target_arch))
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

        let shoff = ELF64_HEADER_SIZE as u64;
        let text_offset = shoff + 3 * ELF64_SECTION_HEADER_SIZE as u64;
        let text_align = 16u64;
        let text_size_aligned = (code.len() as u64 + text_align - 1) & !(text_align - 1);
        let shstrtab_offset = text_offset + text_size_aligned;
        const SHSTRTAB: &[u8] = b"\0.text\0.shstrtab\0";
        let shstrtab_size = SHSTRTAB.len() as u64;

        write_elf64_header_with_shnum(&mut f, e_machine, 3, 2)?;
        write_elf64_section_header(&mut f, 0, SHT_NULL, 0, 0, 0, 0)?;
        write_elf64_section_header(
            &mut f,
            1,
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            text_offset,
            code.len() as u64,
            text_align,
        )?;
        write_elf64_section_header(&mut f, 7, SHT_STRTAB, 0, shstrtab_offset, shstrtab_size, 1)?;

        let padding = text_size_aligned as usize - code.len();
        f.write_all(code)
            .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
        if padding > 0 {
            f.write_all(&vec![0u8; padding])
                .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
        }
        f.write_all(SHSTRTAB)
            .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;

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
        _code: &[u8],
        _sections: &[Section],
        _symbols: &[Symbol],
        _target_arch: TargetArchitecture,
        _target_os: TargetOperatingSystem,
        _opts: &ObjectWriteOptions,
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

impl ObjectWriter for CoffWriter {
    fn write_object_file(
        &mut self,
        path: &std::path::Path,
        code: &[u8],
        _sections: &[Section],
        _symbols: &[Symbol],
        target_arch: TargetArchitecture,
        _target_os: TargetOperatingSystem,
        _opts: &ObjectWriteOptions,
    ) -> Result<(), RasError> {
        if target_arch != TargetArchitecture::X86_64 {
            return Err(RasError::ObjectError(format!(
                "COFF writer: unsupported architecture {:?}",
                target_arch
            )));
        }

        let hdr_sz = 20usize;
        let sec_hdr_sz = 40usize;
        let raw_off = hdr_sz + sec_hdr_sz;
        let align = 16usize;
        let padded_len = (code.len().div_ceil(align) * align) as u32;
        let sym_off = raw_off + padded_len as usize;

        let text_name = b".text\0\0\0";
        let sec_flags = IMAGE_SCN_CNT_CODE
            | IMAGE_SCN_ALIGN_16BYTES
            | IMAGE_SCN_MEM_EXECUTE
            | IMAGE_SCN_MEM_READ;

        let mut sym = [0u8; 18];
        sym[0..8].copy_from_slice(text_name);
        sym[8..12].copy_from_slice(&0u32.to_le_bytes());
        sym[12..14].copy_from_slice(&1i16.to_le_bytes());
        sym[14..16].copy_from_slice(&0u16.to_le_bytes());
        sym[16] = IMAGE_SYM_CLASS_STATIC;
        sym[17] = 0;

        let strtbl = [4u8, 0, 0, 0];

        let mut f = std::fs::File::create(path)
            .map_err(|e| RasError::ObjectError(format!("Failed to create COFF file: {}", e)))?;

        f.write_all(&IMAGE_FILE_MACHINE_AMD64.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&1u16.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&0u32.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&(sym_off as u32).to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&1u32.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&0u16.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&0u16.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;

        f.write_all(text_name)
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&0u32.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&0u32.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&padded_len.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&(raw_off as u32).to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&0u32.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&0u32.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&0u16.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&0u16.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&sec_flags.to_le_bytes())
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;

        f.write_all(code)
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        let pad = padded_len as usize - code.len();
        if pad > 0 {
            f.write_all(&vec![0u8; pad])
                .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        }

        f.write_all(&sym)
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;
        f.write_all(&strtbl)
            .map_err(|e| RasError::ObjectError(format!("COFF write: {}", e)))?;

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
mod tests {
    use super::*;

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
        let symbols: Vec<Symbol> = vec![];
        let path = std::env::temp_dir().join("ras_elf_test_x86_64.o");
        let opts = ObjectWriteOptions::default();
        let result = w.write_object_file(
            &path,
            &code,
            &sections,
            &symbols,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
            &opts,
        );
        let _ = std::fs::remove_file(&path);
        result.expect("ELF write should succeed");
    }

    #[test]
    fn test_elf64_aarch64_write() {
        let mut w = ElfWriter::new();
        let code = [0xc0, 0x03, 0x5f, 0xd6];
        let sections = vec![Section {
            name: ".text".to_string(),
            flags: SectionFlags {
                alloc: true,
                exec: true,
                write: false,
            },
        }];
        let symbols: Vec<Symbol> = vec![];
        let path = std::env::temp_dir().join("ras_elf_test_aarch64.o");
        let opts = ObjectWriteOptions::default();
        let result = w.write_object_file(
            &path,
            &code,
            &sections,
            &symbols,
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Linux,
            &opts,
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
        w.write_object_file(
            &path,
            &code,
            &[],
            &[],
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
            &ObjectWriteOptions::default(),
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
            &code,
            &[],
            &[],
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
            &opts,
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
            &code,
            &[],
            &[],
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Windows,
            &opts,
        )
        .expect("coff write");
        let buf = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert_eq!(&buf[0..2], &IMAGE_FILE_MACHINE_AMD64.to_le_bytes());
    }
}
