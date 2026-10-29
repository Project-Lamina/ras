//! Binary instruction encoders for x86_64, AArch64, RISC-V, and PowerPC64.

pub mod aarch64;
pub mod arx64;
pub mod powerpc;
pub mod riscv;
pub mod traits;
pub mod x86_64;

pub use traits::{InstructionEncoder, ParsedInstruction};
