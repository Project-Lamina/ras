//! Instruction encoder trait

use crate::error::RasError;

#[derive(Debug, Clone)]
pub struct ParsedInstruction {
    pub opcode: String,
    pub operands: Vec<String>,
}

pub trait InstructionEncoder {
    fn encode_instruction(
        &mut self,
        inst: &ParsedInstruction,
    ) -> Result<Vec<u8>, RasError>;

    fn current_position(&self) -> usize;
}
