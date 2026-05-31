//! RISC-V binary instruction encoder (RV32I / RV64I + M-extension)
//!
//! All RISC-V instructions are 32 bits wide and stored little-endian.
//! Supports the standard register ABI names (zero/ra/sp/…/a0-a7/s0-s11/t0-t6).

use crate::encoder::traits::{InstructionEncoder, ParsedInstruction};
use crate::error::RasError;

pub struct RiscVEncoder {
    position: usize,
    /// true = RV64 (64-bit), false = RV32
    rv64: bool,
}

impl Default for RiscVEncoder {
    fn default() -> Self {
        Self::new(true)
    }
}

impl RiscVEncoder {
    pub fn new(rv64: bool) -> Self {
        Self { position: 0, rv64 }
    }

    // -----------------------------------------------------------------------
    // Register parser
    // -----------------------------------------------------------------------

    fn parse_register(&self, s: &str) -> Result<u8, RasError> {
        let s = s.trim_start_matches('%').trim();
        match s {
            // Numeric names
            "x0" => Ok(0),
            "x1" => Ok(1),
            "x2" => Ok(2),
            "x3" => Ok(3),
            "x4" => Ok(4),
            "x5" => Ok(5),
            "x6" => Ok(6),
            "x7" => Ok(7),
            "x8" => Ok(8),
            "x9" => Ok(9),
            "x10" => Ok(10),
            "x11" => Ok(11),
            "x12" => Ok(12),
            "x13" => Ok(13),
            "x14" => Ok(14),
            "x15" => Ok(15),
            "x16" => Ok(16),
            "x17" => Ok(17),
            "x18" => Ok(18),
            "x19" => Ok(19),
            "x20" => Ok(20),
            "x21" => Ok(21),
            "x22" => Ok(22),
            "x23" => Ok(23),
            "x24" => Ok(24),
            "x25" => Ok(25),
            "x26" => Ok(26),
            "x27" => Ok(27),
            "x28" => Ok(28),
            "x29" => Ok(29),
            "x30" => Ok(30),
            "x31" => Ok(31),
            // ABI names
            "zero" => Ok(0),
            "ra" => Ok(1),
            "sp" => Ok(2),
            "gp" => Ok(3),
            "tp" => Ok(4),
            "t0" => Ok(5),
            "t1" => Ok(6),
            "t2" => Ok(7),
            "s0" | "fp" => Ok(8),
            "s1" => Ok(9),
            "a0" => Ok(10),
            "a1" => Ok(11),
            "a2" => Ok(12),
            "a3" => Ok(13),
            "a4" => Ok(14),
            "a5" => Ok(15),
            "a6" => Ok(16),
            "a7" => Ok(17),
            "s2" => Ok(18),
            "s3" => Ok(19),
            "s4" => Ok(20),
            "s5" => Ok(21),
            "s6" => Ok(22),
            "s7" => Ok(23),
            "s8" => Ok(24),
            "s9" => Ok(25),
            "s10" => Ok(26),
            "s11" => Ok(27),
            "t3" => Ok(28),
            "t4" => Ok(29),
            "t5" => Ok(30),
            "t6" => Ok(31),
            _ => Err(RasError::EncodingError(format!(
                "Unknown RISC-V register: {}",
                s
            ))),
        }
    }

    fn parse_imm(&self, s: &str) -> Result<i64, RasError> {
        let s = s.trim();
        // Strip optional '#' or leading '+'.
        let s = s.trim_start_matches('#');
        if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            i64::from_str_radix(hex, 16)
                .map_err(|_| RasError::EncodingError(format!("Invalid hex immediate: {}", s)))
        } else {
            s.parse::<i64>()
                .map_err(|_| RasError::EncodingError(format!("Invalid immediate: {}", s)))
        }
    }

    /// Parse `base_reg` and optional offset from an address operand like
    /// `(sp)`, `8(sp)`, `-4(s0)`.
    fn parse_mem_operand(&self, s: &str) -> Result<(u8, i32), RasError> {
        let s = s.trim();
        if let Some(paren) = s.find('(') {
            let offset_str = s[..paren].trim();
            let reg_str = s[paren + 1..].trim_end_matches(')').trim();
            let offset = if offset_str.is_empty() {
                0i32
            } else {
                self.parse_imm(offset_str)? as i32
            };
            let base = self.parse_register(reg_str)?;
            Ok((base, offset))
        } else {
            // No offset: treat the whole string as a register with offset 0
            Ok((self.parse_register(s)?, 0))
        }
    }

    // -----------------------------------------------------------------------
    // Instruction format encoders
    // -----------------------------------------------------------------------

    #[inline]
    fn r_type(funct7: u8, rs2: u8, rs1: u8, funct3: u8, rd: u8, opcode: u8) -> u32 {
        ((funct7 as u32) << 25)
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | ((funct3 as u32) << 12)
            | ((rd as u32) << 7)
            | (opcode as u32)
    }

    #[inline]
    fn i_type(imm12: i32, rs1: u8, funct3: u8, rd: u8, opcode: u8) -> u32 {
        ((imm12 as u32 & 0xFFF) << 20)
            | ((rs1 as u32) << 15)
            | ((funct3 as u32) << 12)
            | ((rd as u32) << 7)
            | (opcode as u32)
    }

    #[inline]
    fn s_type(imm12: i32, rs2: u8, rs1: u8, funct3: u8, opcode: u8) -> u32 {
        let imm = imm12 as u32 & 0xFFF;
        ((imm >> 5) << 25)
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | ((funct3 as u32) << 12)
            | ((imm & 0x1F) << 7)
            | (opcode as u32)
    }

    /// B-type: offset is in bytes, must be 2-byte aligned.
    /// Bits: [31]=imm[12], [30:25]=imm[10:5], [24:20]=rs2, [19:15]=rs1,
    ///       [14:12]=funct3, [11:8]=imm[4:1], [7]=imm[11], [6:0]=opcode
    #[inline]
    fn b_type(offset: i32, rs2: u8, rs1: u8, funct3: u8, opcode: u8) -> u32 {
        let o = offset as u32;
        let imm12 = (o >> 12) & 1;
        let imm11 = (o >> 11) & 1;
        let imm10_5 = (o >> 5) & 0x3F;
        let imm4_1 = (o >> 1) & 0xF;
        (imm12 << 31)
            | (imm10_5 << 25)
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | ((funct3 as u32) << 12)
            | (imm4_1 << 8)
            | (imm11 << 7)
            | (opcode as u32)
    }

    /// U-type: imm20 occupies bits [31:12].
    #[inline]
    fn u_type(imm20: i32, rd: u8, opcode: u8) -> u32 {
        ((imm20 as u32 & 0xF_FFFF) << 12) | ((rd as u32) << 7) | (opcode as u32)
    }

    /// J-type: offset is in bytes, must be 2-byte aligned.
    /// Bits: [31]=imm[20], [30:21]=imm[10:1], [20]=imm[11], [19:12]=imm[19:12],
    ///       [11:7]=rd, [6:0]=opcode
    #[inline]
    fn j_type(offset: i32, rd: u8, opcode: u8) -> u32 {
        let o = offset as u32;
        let imm20 = (o >> 20) & 1;
        let imm19_12 = (o >> 12) & 0xFF;
        let imm11 = (o >> 11) & 1;
        let imm10_1 = (o >> 1) & 0x3FF;
        (imm20 << 31)
            | (imm10_1 << 21)
            | (imm11 << 20)
            | (imm19_12 << 12)
            | ((rd as u32) << 7)
            | (opcode as u32)
    }

    #[inline]
    fn emit(word: u32) -> Vec<u8> {
        word.to_le_bytes().to_vec()
    }

    // -----------------------------------------------------------------------
    // Convenience dispatch for common 3-operand R-type ops
    // -----------------------------------------------------------------------

    fn encode_r3(
        &self,
        ops: &[String],
        funct7: u8,
        funct3: u8,
        opcode: u8,
    ) -> Result<Vec<u8>, RasError> {
        if ops.len() != 3 {
            return Err(RasError::EncodingError(
                "R-type instruction requires 3 operands: rd, rs1, rs2".to_string(),
            ));
        }
        let rd = self.parse_register(&ops[0])?;
        let rs1 = self.parse_register(&ops[1])?;
        let rs2 = self.parse_register(&ops[2])?;
        Ok(Self::emit(Self::r_type(
            funct7, rs2, rs1, funct3, rd, opcode,
        )))
    }

    /// 3-operand I-type: `mnemonic rd, rs1, imm`
    fn encode_i3(&self, ops: &[String], funct3: u8, opcode: u8) -> Result<Vec<u8>, RasError> {
        if ops.len() != 3 {
            return Err(RasError::EncodingError(
                "I-type instruction requires 3 operands: rd, rs1, imm".to_string(),
            ));
        }
        let rd = self.parse_register(&ops[0])?;
        let rs1 = self.parse_register(&ops[1])?;
        let imm = self.parse_imm(&ops[2])? as i32;
        if !(-2048..=2047).contains(&imm) {
            return Err(RasError::EncodingError(format!(
                "Immediate {} out of 12-bit signed range [-2048, 2047]",
                imm
            )));
        }
        Ok(Self::emit(Self::i_type(imm, rs1, funct3, rd, opcode)))
    }

    /// Shift-immediate: uses the upper 7 bits as funct7 and lower 5 (or 6 for RV64) as shamt.
    fn encode_shift_imm(
        &self,
        ops: &[String],
        funct7: u8,
        funct3: u8,
        opcode: u8,
    ) -> Result<Vec<u8>, RasError> {
        if ops.len() != 3 {
            return Err(RasError::EncodingError(
                "Shift immediate requires 3 operands: rd, rs1, shamt".to_string(),
            ));
        }
        let rd = self.parse_register(&ops[0])?;
        let rs1 = self.parse_register(&ops[1])?;
        let shamt = self.parse_imm(&ops[2])? as u32;
        let max_shamt = if self.rv64 { 63u32 } else { 31u32 };
        if shamt > max_shamt {
            return Err(RasError::EncodingError(format!(
                "Shift amount {} exceeds maximum {}",
                shamt, max_shamt
            )));
        }
        // For RV64: imm[11:0] = funct7[6:1] | shamt[5:0]
        let imm12 = ((funct7 as i32) << 5) | (shamt as i32 & 0x3F);
        Ok(Self::emit(Self::i_type(imm12, rs1, funct3, rd, opcode)))
    }

    /// B-type branch: `mnemonic rs1, rs2, offset`
    fn encode_branch(&self, ops: &[String], funct3: u8) -> Result<Vec<u8>, RasError> {
        if ops.len() != 3 {
            return Err(RasError::EncodingError(
                "Branch requires 3 operands: rs1, rs2, offset".to_string(),
            ));
        }
        let rs1 = self.parse_register(&ops[0])?;
        let rs2 = self.parse_register(&ops[1])?;
        let offset = self.parse_imm(&ops[2])? as i32;
        Ok(Self::emit(Self::b_type(offset, rs2, rs1, funct3, 0x63)))
    }

    /// Load: `mnemonic rd, offset(rs1)`
    fn encode_load(&self, ops: &[String], funct3: u8) -> Result<Vec<u8>, RasError> {
        if ops.len() != 2 {
            return Err(RasError::EncodingError(
                "Load requires 2 operands: rd, offset(rs1)".to_string(),
            ));
        }
        let rd = self.parse_register(&ops[0])?;
        let (rs1, offset) = self.parse_mem_operand(&ops[1])?;
        Ok(Self::emit(Self::i_type(offset, rs1, funct3, rd, 0x03)))
    }

    /// Store: `mnemonic rs2, offset(rs1)`
    fn encode_store(&self, ops: &[String], funct3: u8) -> Result<Vec<u8>, RasError> {
        if ops.len() != 2 {
            return Err(RasError::EncodingError(
                "Store requires 2 operands: rs2, offset(rs1)".to_string(),
            ));
        }
        let rs2 = self.parse_register(&ops[0])?;
        let (rs1, offset) = self.parse_mem_operand(&ops[1])?;
        Ok(Self::emit(Self::s_type(offset, rs2, rs1, funct3, 0x23)))
    }
}

impl InstructionEncoder for RiscVEncoder {
    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let opcode = inst.opcode.to_lowercase();
        let ops = &inst.operands;

        let bytes = match opcode.as_str() {
            // ------- Integer Arithmetic (R-type, opcode 0x33) -------
            "add" => self.encode_r3(ops, 0x00, 0x0, 0x33)?,
            "sub" => self.encode_r3(ops, 0x20, 0x0, 0x33)?,
            "sll" => self.encode_r3(ops, 0x00, 0x1, 0x33)?,
            "slt" => self.encode_r3(ops, 0x00, 0x2, 0x33)?,
            "sltu" => self.encode_r3(ops, 0x00, 0x3, 0x33)?,
            "xor" => self.encode_r3(ops, 0x00, 0x4, 0x33)?,
            "srl" => self.encode_r3(ops, 0x00, 0x5, 0x33)?,
            "sra" => self.encode_r3(ops, 0x20, 0x5, 0x33)?,
            "or" => self.encode_r3(ops, 0x00, 0x6, 0x33)?,
            "and" => self.encode_r3(ops, 0x00, 0x7, 0x33)?,

            // ------- M-extension multiply/divide (R-type, funct7=1, opcode 0x33) -------
            "mul" => self.encode_r3(ops, 0x01, 0x0, 0x33)?,
            "mulh" => self.encode_r3(ops, 0x01, 0x1, 0x33)?,
            "mulhsu" => self.encode_r3(ops, 0x01, 0x2, 0x33)?,
            "mulhu" => self.encode_r3(ops, 0x01, 0x3, 0x33)?,
            "div" => self.encode_r3(ops, 0x01, 0x4, 0x33)?,
            "divu" => self.encode_r3(ops, 0x01, 0x5, 0x33)?,
            "rem" => self.encode_r3(ops, 0x01, 0x6, 0x33)?,
            "remu" => self.encode_r3(ops, 0x01, 0x7, 0x33)?,

            // ------- RV64 word-width R-type (opcode 0x3B) -------
            "addw" => self.encode_r3(ops, 0x00, 0x0, 0x3B)?,
            "subw" => self.encode_r3(ops, 0x20, 0x0, 0x3B)?,
            "sllw" => self.encode_r3(ops, 0x00, 0x1, 0x3B)?,
            "srlw" => self.encode_r3(ops, 0x00, 0x5, 0x3B)?,
            "sraw" => self.encode_r3(ops, 0x20, 0x5, 0x3B)?,
            "mulw" => self.encode_r3(ops, 0x01, 0x0, 0x3B)?,
            "divw" => self.encode_r3(ops, 0x01, 0x4, 0x3B)?,
            "divuw" => self.encode_r3(ops, 0x01, 0x5, 0x3B)?,
            "remw" => self.encode_r3(ops, 0x01, 0x6, 0x3B)?,
            "remuw" => self.encode_r3(ops, 0x01, 0x7, 0x3B)?,

            // ------- Integer Immediate (I-type, opcode 0x13) -------
            "addi" => self.encode_i3(ops, 0x0, 0x13)?,
            "slti" => self.encode_i3(ops, 0x2, 0x13)?,
            "sltiu" => self.encode_i3(ops, 0x3, 0x13)?,
            "xori" => self.encode_i3(ops, 0x4, 0x13)?,
            "ori" => self.encode_i3(ops, 0x6, 0x13)?,
            "andi" => self.encode_i3(ops, 0x7, 0x13)?,
            "slli" => self.encode_shift_imm(ops, 0x00, 0x1, 0x13)?,
            "srli" => self.encode_shift_imm(ops, 0x00, 0x5, 0x13)?,
            "srai" => self.encode_shift_imm(ops, 0x20, 0x5, 0x13)?,

            // ------- RV64 word-width immediate (opcode 0x1B) -------
            "addiw" => self.encode_i3(ops, 0x0, 0x1B)?,
            "slliw" => self.encode_shift_imm(ops, 0x00, 0x1, 0x1B)?,
            "srliw" => self.encode_shift_imm(ops, 0x00, 0x5, 0x1B)?,
            "sraiw" => self.encode_shift_imm(ops, 0x20, 0x5, 0x1B)?,

            // ------- Loads -------
            "lb" => self.encode_load(ops, 0x0)?,
            "lh" => self.encode_load(ops, 0x1)?,
            "lw" => self.encode_load(ops, 0x2)?,
            "ld" => self.encode_load(ops, 0x3)?,
            "lbu" => self.encode_load(ops, 0x4)?,
            "lhu" => self.encode_load(ops, 0x5)?,
            "lwu" => self.encode_load(ops, 0x6)?,

            // ------- Stores -------
            "sb" => self.encode_store(ops, 0x0)?,
            "sh" => self.encode_store(ops, 0x1)?,
            "sw" => self.encode_store(ops, 0x2)?,
            "sd" => self.encode_store(ops, 0x3)?,

            // ------- Branches -------
            "beq" => self.encode_branch(ops, 0x0)?,
            "bne" => self.encode_branch(ops, 0x1)?,
            "blt" => self.encode_branch(ops, 0x4)?,
            "bge" => self.encode_branch(ops, 0x5)?,
            "bltu" => self.encode_branch(ops, 0x6)?,
            "bgeu" => self.encode_branch(ops, 0x7)?,

            // ------- JAL (J-type) -------
            "jal" => {
                if ops.len() != 2 {
                    return Err(RasError::EncodingError(
                        "JAL requires 2 operands: rd, offset".to_string(),
                    ));
                }
                let rd = self.parse_register(&ops[0])?;
                let offset = self.parse_imm(&ops[1])? as i32;
                Self::emit(Self::j_type(offset, rd, 0x6F))
            }

            // ------- JALR (I-type) -------
            "jalr" => {
                if ops.len() == 3 {
                    // jalr rd, rs1, imm  (explicit form)
                    let rd = self.parse_register(&ops[0])?;
                    let rs1 = self.parse_register(&ops[1])?;
                    let imm = self.parse_imm(&ops[2])? as i32;
                    Self::emit(Self::i_type(imm, rs1, 0x0, rd, 0x67))
                } else if ops.len() == 2 {
                    // jalr rd, offset(rs1)  (memory form)
                    let rd = self.parse_register(&ops[0])?;
                    let (rs1, offset) = self.parse_mem_operand(&ops[1])?;
                    Self::emit(Self::i_type(offset, rs1, 0x0, rd, 0x67))
                } else if ops.len() == 1 {
                    // jalr rs1  (implicit rd=ra, offset=0)
                    let rs1 = self.parse_register(&ops[0])?;
                    Self::emit(Self::i_type(0, rs1, 0x0, 1, 0x67))
                } else {
                    return Err(RasError::EncodingError(
                        "JALR requires 1–3 operands".to_string(),
                    ));
                }
            }

            // ------- LUI / AUIPC (U-type) -------
            "lui" => {
                if ops.len() != 2 {
                    return Err(RasError::EncodingError(
                        "LUI requires 2 operands: rd, imm20".to_string(),
                    ));
                }
                let rd = self.parse_register(&ops[0])?;
                let imm20 = self.parse_imm(&ops[1])? as i32;
                Self::emit(Self::u_type(imm20, rd, 0x37))
            }
            "auipc" => {
                if ops.len() != 2 {
                    return Err(RasError::EncodingError(
                        "AUIPC requires 2 operands: rd, imm20".to_string(),
                    ));
                }
                let rd = self.parse_register(&ops[0])?;
                let imm20 = self.parse_imm(&ops[1])? as i32;
                Self::emit(Self::u_type(imm20, rd, 0x17))
            }

            // ------- SYSTEM / FENCE -------
            "ecall" => Self::emit(Self::i_type(0, 0, 0, 0, 0x73)),
            "ebreak" => Self::emit(Self::i_type(1, 0, 0, 0, 0x73)),
            "fence" | "fence.i" => Self::emit(Self::i_type(0, 0, 0, 0, 0x0F)),

            // ------- Pseudo-instructions -------
            // `nop` = addi x0, x0, 0
            "nop" => Self::emit(Self::i_type(0, 0, 0, 0, 0x13)),
            // `ret` = jalr x0, 0(ra)
            "ret" => Self::emit(Self::i_type(0, 1, 0, 0, 0x67)),
            // `mv rd, rs` = addi rd, rs, 0
            "mv" => {
                if ops.len() != 2 {
                    return Err(RasError::EncodingError(
                        "MV requires 2 operands: rd, rs".to_string(),
                    ));
                }
                let rd = self.parse_register(&ops[0])?;
                let rs = self.parse_register(&ops[1])?;
                Self::emit(Self::i_type(0, rs, 0, rd, 0x13))
            }
            // `li rd, imm` = addi rd, x0, imm  (small immediates only)
            "li" => {
                if ops.len() != 2 {
                    return Err(RasError::EncodingError(
                        "LI requires 2 operands: rd, imm".to_string(),
                    ));
                }
                let rd = self.parse_register(&ops[0])?;
                let imm = self.parse_imm(&ops[1])? as i32;
                if !(-2048..=2047).contains(&imm) {
                    return Err(RasError::EncodingError(format!(
                        "LI pseudo-instruction only supports 12-bit immediates ({} out of range)",
                        imm
                    )));
                }
                Self::emit(Self::i_type(imm, 0, 0, rd, 0x13))
            }
            // `j offset` = jal x0, offset
            "j" => {
                if ops.len() != 1 {
                    return Err(RasError::EncodingError(
                        "J requires 1 operand: offset".to_string(),
                    ));
                }
                let offset = self.parse_imm(&ops[0])? as i32;
                Self::emit(Self::j_type(offset, 0, 0x6F))
            }
            // `call offset` = jal ra, offset
            "call" => {
                if ops.len() != 1 {
                    return Err(RasError::EncodingError(
                        "CALL requires 1 operand: offset".to_string(),
                    ));
                }
                let offset = self.parse_imm(&ops[0])? as i32;
                Self::emit(Self::j_type(offset, 1, 0x6F))
            }

            _ => {
                return Err(RasError::EncodingError(format!(
                    "Unknown RISC-V instruction: {}",
                    opcode
                )));
            }
        };

        self.position += bytes.len();
        Ok(bytes)
    }

    fn current_position(&self) -> usize {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::traits::ParsedInstruction;

    fn instr(opcode: &str, operands: &[&str]) -> ParsedInstruction {
        ParsedInstruction {
            opcode: opcode.to_string(),
            operands: operands.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn enc() -> RiscVEncoder {
        RiscVEncoder::new(true)
    }

    #[test]
    fn test_nop_is_four_bytes() {
        let bytes = enc().encode_instruction(&instr("nop", &[])).unwrap();
        assert_eq!(bytes.len(), 4);
    }

    #[test]
    fn test_add_encoding() {
        // add a0, a1, a2  → R-type opcode=0x33 funct3=0 funct7=0
        let bytes = enc()
            .encode_instruction(&instr("add", &["a0", "a1", "a2"]))
            .unwrap();
        assert_eq!(bytes.len(), 4);
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x33); // opcode
        assert_eq!((word >> 12) & 7, 0); // funct3
        assert_eq!((word >> 25) & 0x7F, 0); // funct7
        assert_eq!((word >> 7) & 0x1F, 10); // rd = a0 = 10
    }

    #[test]
    fn test_addi_encoding() {
        // addi t0, zero, 42  → I-type
        let bytes = enc()
            .encode_instruction(&instr("addi", &["t0", "zero", "42"]))
            .unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x13); // opcode OP-IMM
        assert_eq!((word >> 7) & 0x1F, 5); // rd = t0 = 5
        assert_eq!((word >> 20) as i32 as i32, 42); // imm = 42 (sign-extended)
    }

    #[test]
    fn test_ret_pseudo() {
        // ret → jalr x0, 0(ra) → I-type i=0, rs1=1, funct3=0, rd=0, opcode=0x67
        let bytes = enc().encode_instruction(&instr("ret", &[])).unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x67); // JALR opcode
        assert_eq!((word >> 7) & 0x1F, 0); // rd = x0
        assert_eq!((word >> 15) & 0x1F, 1); // rs1 = ra
        assert_eq!((word >> 20), 0); // imm = 0
    }

    #[test]
    fn test_load_store_encoding() {
        // sw a0, 0(sp)
        let bytes = enc()
            .encode_instruction(&instr("sw", &["a0", "0(sp)"]))
            .unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x23); // STORE opcode
        assert_eq!((word >> 12) & 7, 0x2); // funct3 = SW

        // lw a1, 4(sp)
        let bytes = enc()
            .encode_instruction(&instr("lw", &["a1", "4(sp)"]))
            .unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x03); // LOAD opcode
        assert_eq!((word >> 12) & 7, 0x2); // funct3 = LW
    }

    #[test]
    fn test_position_advances() {
        let mut e = enc();
        e.encode_instruction(&instr("nop", &[])).unwrap();
        assert_eq!(e.current_position(), 4);
        e.encode_instruction(&instr("nop", &[])).unwrap();
        assert_eq!(e.current_position(), 8);
    }

    #[test]
    fn test_abi_register_aliases() {
        // Both 'zero' and 'x0' should parse to 0
        let mut e = enc();
        let b1 = e
            .encode_instruction(&instr("addi", &["zero", "zero", "0"]))
            .unwrap();
        let b2 = e
            .encode_instruction(&instr("addi", &["x0", "x0", "0"]))
            .unwrap();
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_mul_encoding() {
        // mul a0, a1, a2 → R-type funct7=1
        let bytes = enc()
            .encode_instruction(&instr("mul", &["a0", "a1", "a2"]))
            .unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!((word >> 25) & 0x7F, 1); // funct7 = M-extension
    }
}
