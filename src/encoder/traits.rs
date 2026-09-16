//! Instruction encoder trait

use crate::error::RasError;

/// Displacement field of a RIP-relative operand inside an encoded instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RipFixup {
    pub disp_offset: usize,
    pub trailing: usize,
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
