//! Binary instruction encoders for x86_64, AArch64, and RISC-V.

pub mod aarch64;
pub mod riscv;
pub mod traits;
pub mod x86_64;

pub use traits::{InstructionEncoder, ParsedInstruction};
