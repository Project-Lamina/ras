//! Object file generation: ELF (Linux/BSD), Mach-O (macOS), COFF/PE (Windows).

use crate::error::RasError;
use crate::parser::{Section, Symbol};
#[cfg(test)]
use crate::parser::SectionFlags;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::io::Write;

pub trait ObjectWriter {
    fn write_object_file(
        &mut self,
        path: &std::path::Path,
        code: &[u8],
        sections: &[Section],
        symbols: &[Symbol],
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Result<(), RasError>;
}

const ELF64_HEADER_SIZE: usize = 64;
const ELF64_SECTION_HEADER_SIZE: usize = 64;
const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ET_REL: u16 = 1;
const EM_X86_64: u16 = 62;
const EM_AARCH64: u16 = 183;
const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_STRTAB: u32 = 3;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;

fn elf64_e_machine(arch: TargetArchitecture) -> Option<u16> {
    match arch {
        TargetArchitecture::X86_64 => Some(EM_X86_64),
        TargetArchitecture::Aarch64 => Some(EM_AARCH64),
        _ => None,
    }
}

fn write_elf64_header<W: Write>(w: &mut W, e_machine: u16) -> Result<(), RasError> {
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
    w.write_all(&3u16.to_le_bytes())
        .map_err(|e| RasError::ObjectError(format!("ELF write: {}", e)))?;
    w.write_all(&2u16.to_le_bytes())
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
    ) -> Result<(), RasError> {
        let e_machine = elf64_e_machine(target_arch)
            .ok_or_else(|| RasError::ObjectError(format!("ELF does not support arch: {:?}", target_arch)))?;

        let shoff = ELF64_HEADER_SIZE as u64;
        let text_offset = shoff + 3 * ELF64_SECTION_HEADER_SIZE as u64;
        let text_align = 16u64;
        let text_size_aligned = (code.len() as u64 + text_align - 1) & !(text_align - 1);
        let shstrtab_offset = text_offset + text_size_aligned;
        const SHSTRTAB: &[u8] = b"\0.text\0.shstrtab\0";
        let shstrtab_size = SHSTRTAB.len() as u64;

        let mut f = std::fs::File::create(path)
            .map_err(|e| RasError::ObjectError(format!("Failed to create ELF file: {}", e)))?;

        write_elf64_header(&mut f, e_machine)?;
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
        write_elf64_section_header(
            &mut f,
            7,
            SHT_STRTAB,
            0,
            shstrtab_offset,
            shstrtab_size,
            1,
        )?;

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
        let result = w.write_object_file(
            &path,
            &code,
            &sections,
            &symbols,
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
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
        let result = w.write_object_file(
            &path,
            &code,
            &sections,
            &symbols,
            TargetArchitecture::Aarch64,
            TargetOperatingSystem::Linux,
        );
        let _ = std::fs::remove_file(&path);
        result.expect("ELF write should succeed");
    }

    #[test]
    fn test_elf64_has_valid_magic() {
        let mut w = ElfWriter::new();
        let code = [0xc3u8];
        let path = std::env::temp_dir().join("ras_elf_magic_test.o");
        w.write_object_file(&path, &code, &[], &[], TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .expect("write");
        let buf = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert!(buf.len() >= 64);
        assert_eq!(&buf[0..4], &[0x7f, b'E', b'L', b'F']);
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

impl ObjectWriter for CoffWriter {
    fn write_object_file(
        &mut self,
        _path: &std::path::Path,
        _code: &[u8],
        _sections: &[Section],
        _symbols: &[Symbol],
        _target_arch: TargetArchitecture,
        _target_os: TargetOperatingSystem,
    ) -> Result<(), RasError> {
        Err(RasError::ObjectError(
            "COFF/PE object file generation not yet implemented".to_string(),
        ))
    }
}
