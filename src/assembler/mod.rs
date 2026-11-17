//! Assembler: assembly text to object file, MIR to binary (JIT)

pub mod core;

#[cfg(feature = "encoder")]
pub mod x86_64;

#[cfg(feature = "encoder")]
pub mod aarch64;

#[cfg(feature = "encoder")]
pub mod riscv;

pub use core::RasAssembler;

/// The compiler-runtime routines lamina calls for 128-bit arithmetic and conversions.
/// Lamina's module declares them with pointer operands, while a host runtime exports
/// them with the by-value 128-bit ABI, so a JIT must not call the host copy.
#[cfg(feature = "encoder")]
pub(crate) fn is_wide_integer_helper(name: &str) -> bool {
    matches!(
        name.trim_start_matches('@'),
        "__divti3"
            | "__modti3"
            | "__udivti3"
            | "__umodti3"
            | "__fixdfti"
            | "__fixunsdfti"
            | "__fixsfti"
            | "__fixunssfti"
            | "__floattidf"
            | "__floatuntidf"
            | "__floattisf"
            | "__floatuntisf"
    )
}
