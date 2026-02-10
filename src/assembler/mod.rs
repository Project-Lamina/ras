//! Assembler: assembly text to object file, MIR to binary (JIT)

pub mod core;
pub mod common;

#[cfg(feature = "encoder")]
pub mod x86_64;

#[cfg(feature = "encoder")]
pub mod aarch64;

pub use core::RasAssembler;
