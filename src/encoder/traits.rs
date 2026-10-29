//! Instruction encoder trait

use crate::error::RasError;

/// Thread-local storage model of a memory operand that the linker resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsModel {
    InitialExec,
    LocalExec,
}

/// Displacement field of a memory operand whose value the linker fills in: a
/// RIP-relative symbol, or a thread-local offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RipFixup {
    pub disp_offset: usize,
    pub trailing: usize,
    pub tls: Option<TlsModel>,
}

#[derive(Debug, Clone)]
pub struct ParsedInstruction {
    pub opcode: String,
    pub operands: Vec<String>,
}

pub trait InstructionEncoder {
    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError>;

    fn current_position(&self) -> usize;

    fn take_rip_fixup(&mut self) -> Option<RipFixup> {
        None
    }
}
