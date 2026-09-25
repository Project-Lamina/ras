//! Assembler: assembly text to object file, MIR to binary (JIT)

pub mod core;

#[cfg(feature = "encoder")]
pub mod x86_64;

#[cfg(feature = "encoder")]
pub mod aarch64;

#[cfg(feature = "encoder")]
pub mod riscv;

pub use core::RasAssembler;
