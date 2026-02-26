//! Binary instruction encoders for x86_64 and AArch64.

pub mod traits;
pub mod x86_64;
pub mod aarch64;

pub use traits::{InstructionEncoder, ParsedInstruction};
