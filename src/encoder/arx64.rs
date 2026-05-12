//! ARX64 binary instruction encoder.
//!
//! ARX64 base integer encodings currently track the RISC-style 32-bit formats
//! implemented by the reference emulator. This encoder keeps ARX register
//! spelling (`r0`-`r31`) at the assembly surface and delegates bit encoding to
//! the existing RISC-V format encoder for the shared base subset.

use crate::encoder::riscv::RiscVEncoder;
use crate::encoder::traits::{InstructionEncoder, ParsedInstruction};
use crate::error::RasError;

pub struct Arx64Encoder {
    inner: RiscVEncoder,
}

impl Arx64Encoder {
    pub fn new() -> Self {
        Self {
            inner: RiscVEncoder::new(true),
        }
    }

    fn normalize_operand(operand: &str) -> String {
        let mut out = operand.to_string();
        for reg in (0..32).rev() {
            out = out.replace(&format!("r{}", reg), &format!("x{}", reg));
            out = out.replace(&format!("%r{}", reg), &format!("x{}", reg));
        }
        out
    }

    fn normalize_instruction(inst: &ParsedInstruction) -> ParsedInstruction {
        ParsedInstruction {
            opcode: inst.opcode.clone(),
            operands: inst
                .operands
                .iter()
                .map(|operand| Self::normalize_operand(operand))
                .collect(),
        }
    }
}

impl Default for Arx64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl InstructionEncoder for Arx64Encoder {
    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let normalized = Self::normalize_instruction(inst);
        self.inner.encode_instruction(&normalized)
    }

    fn current_position(&self) -> usize {
        self.inner.current_position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instr(opcode: &str, operands: &[&str]) -> ParsedInstruction {
        ParsedInstruction {
            opcode: opcode.to_string(),
            operands: operands.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn encodes_arx_register_names() {
        let bytes = Arx64Encoder::new()
            .encode_instruction(&instr("add", &["r5", "r6", "r7"]))
            .expect("encode");
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!((word >> 7) & 0x1f, 5);
        assert_eq!((word >> 15) & 0x1f, 6);
        assert_eq!((word >> 20) & 0x1f, 7);
    }

    #[test]
    fn encodes_arx_memory_operand_names() {
        let bytes = Arx64Encoder::new()
            .encode_instruction(&instr("ld", &["r5", "8(r2)"]))
            .expect("encode");
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7f, 0x03);
        assert_eq!((word >> 7) & 0x1f, 5);
        assert_eq!((word >> 15) & 0x1f, 2);
    }
}
