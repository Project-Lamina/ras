//! ras - drop-in replacement for as / GAS
//!
//! Assembly source (.s) to relocatable object files (.o).
//! Supported: x86_64 and AArch64; ELF on Linux/BSD/Redox.

mod aarch64_ldst_imm64;

pub mod assembler;
pub mod encoder;
pub mod error;
pub mod object;
pub mod parser;
pub mod runtime;
pub mod target;

pub use assembler::RasAssembler;
pub use error::RasError;
pub use object::ObjectWriteOptions;
pub use runtime::{ExecutableMemory, RasRuntime};
pub use target::{
    is_assembly_target_supported, is_jit_arch_supported, is_object_file_supported,
    supported_jit_targets_hint,
};

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

/// Main assembler interface
pub struct Ras {
    assembler: RasAssembler,
}

impl Ras {
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Result<Self, RasError> {
        Self::with_object_write_options(target_arch, target_os, ObjectWriteOptions::default())
    }

    pub fn with_object_write_options(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
        object_write_options: ObjectWriteOptions,
    ) -> Result<Self, RasError> {
        Ok(Self {
            assembler: RasAssembler::with_object_write_options(
                target_arch,
                target_os,
                object_write_options,
            )?,
        })
    }

    pub fn assemble(
        &mut self,
        asm_text: &str,
        output_path: &std::path::Path,
    ) -> Result<(), RasError> {
        self.assembler
            .assemble_text_to_object(asm_text, output_path)
    }

    pub fn assemble_file(
        &mut self,
        input_path: &std::path::Path,
        output_path: &std::path::Path,
    ) -> Result<(), RasError> {
        let asm_text = std::fs::read_to_string(input_path)
            .map_err(|e| RasError::IoError(format!("Failed to read input: {}", e)))?;
        self.assemble(&asm_text, output_path)
    }

    #[cfg(feature = "encoder")]
    pub fn compile_mir_to_binary(
        &mut self,
        module: &lamina_mir::Module,
    ) -> Result<Vec<u8>, RasError> {
        self.assembler.compile_mir_to_binary(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assemble_gas_style_to_elf() {
        let asm = r#"
.text
.globl main
main:
    movq $42, %rax
    ret
"#;
        let mut ras =
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux).expect("ras new");
        let path = std::env::temp_dir().join("ras_gas_style_test.o");
        ras.assemble(asm, &path).expect("assemble");
        let buf = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert!(buf.len() >= 64);
        assert_eq!(&buf[0..4], &[0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn test_assemble_memory_operands() {
        let asm = r#"
.text
.globl main
main:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq $42, %rax
    movq %rax, -8(%rbp)
    movq -8(%rbp), %rax
    addq $16, %rsp
    popq %rbp
    ret
"#;
        let mut ras =
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux).expect("ras new");
        let path = std::env::temp_dir().join("ras_mem_test.o");
        ras.assemble(asm, &path).expect("assemble");
        let buf = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert!(buf.len() >= 64);
        assert_eq!(&buf[0..4], &[0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn test_assemble_jmp_label_resolution() {
        let asm = r#"
.text
.globl main
main:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq $0, %rax
    jmp .L_exit
.L_exit:
    addq $16, %rsp
    popq %rbp
    ret
"#;
        let mut ras =
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux).expect("ras new");
        let path = std::env::temp_dir().join("ras_jmp_test.o");
        ras.assemble(asm, &path).expect("assemble");
        let buf = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert!(buf.len() >= 64);
        assert_eq!(&buf[0..4], &[0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn test_assemble_arx64_label_resolution_to_elf() {
        let asm = r#"
.text
.globl main
main:
    addi r5, r0, 1
    j done
    addi r5, r0, 2
done:
    ret
"#;
        let mut ras =
            Ras::new(TargetArchitecture::Arx64, TargetOperatingSystem::Linux).expect("ras new");
        let path = std::env::temp_dir().join("ras_arx64_label_test.o");
        ras.assemble(asm, &path).expect("assemble");
        let buf = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert!(buf.len() >= 64);
        assert_eq!(&buf[0..4], &[0x7f, b'E', b'L', b'F']);
    }
}
