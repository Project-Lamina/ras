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

    fn parse_register(&self, s: &str) -> Result<u8, RasError> {
        let s = s.trim_start_matches('%').trim();
        match s {
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

    fn imm12(&self, s: &str) -> Result<i32, RasError> {
        let imm = self.parse_imm(s)?;
        if !(-2048..=2047).contains(&imm) {
            return Err(RasError::EncodingError(format!(
                "Immediate {} out of 12-bit signed range [-2048, 2047]",
                imm
            )));
        }
        Ok(imm as i32)
    }

    fn imm20(&self, s: &str) -> Result<i32, RasError> {
        let imm = self.parse_imm(s)?;
        if !(0..=0xF_FFFF).contains(&imm) {
            return Err(RasError::EncodingError(format!(
                "Immediate {} out of 20-bit range [0, 1048575]",
                imm
            )));
        }
        Ok(imm as i32)
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
                self.imm12(offset_str)?
            };
            let base = self.parse_register(reg_str)?;
            Ok((base, offset))
        } else {
            // No offset: treat the whole string as a register with offset 0
            Ok((self.parse_register(s)?, 0))
        }
    }

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
    pub(crate) fn b_type(offset: i32, rs2: u8, rs1: u8, funct3: u8, opcode: u8) -> u32 {
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
    pub(crate) fn j_type(offset: i32, rd: u8, opcode: u8) -> u32 {
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
        let imm = self.imm12(&ops[2])?;
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
        let shamt = self.parse_imm(&ops[2])?;
        let max_shamt = if self.rv64 { 63 } else { 31 };
        if !(0..=max_shamt).contains(&shamt) {
            return Err(RasError::EncodingError(format!(
                "Shift amount {} exceeds maximum {}",
                shamt, max_shamt
            )));
        }
        let shamt = shamt as u32;
        // For RV64: imm[11:0] = funct7[6:1] | shamt[5:0]
        let imm12 = ((funct7 as i32) << 5) | (shamt as i32 & 0x3F);
        Ok(Self::emit(Self::i_type(imm12, rs1, funct3, rd, opcode)))
    }

    /// Byte offset of a B-type branch: even, within ±4 KiB.
    pub(crate) fn branch_offset(offset: i64) -> Result<i32, RasError> {
        if offset % 2 != 0 {
            return Err(RasError::EncodingError(
                "RISC-V branch offset is not 2-byte aligned".to_string(),
            ));
        }
        if !(-4096..=4094).contains(&offset) {
            return Err(RasError::EncodingError(
                "RISC-V branch out of range".to_string(),
            ));
        }
        Ok(offset as i32)
    }

    /// Byte offset of a J-type jump: even, within ±1 MiB.
    pub(crate) fn jump_offset(offset: i64) -> Result<i32, RasError> {
        if offset % 2 != 0 {
            return Err(RasError::EncodingError(
                "RISC-V jump offset is not 2-byte aligned".to_string(),
            ));
        }
        if !(-(1 << 20)..=(1 << 20) - 2).contains(&offset) {
            return Err(RasError::EncodingError(
                "RISC-V jump out of range".to_string(),
            ));
        }
        Ok(offset as i32)
    }

    fn branch_word(&self, rs1: u8, rs2: u8, offset: &str, funct3: u8) -> Result<Vec<u8>, RasError> {
        let offset = Self::branch_offset(self.parse_imm(offset)?)?;
        Ok(Self::emit(Self::b_type(offset, rs2, rs1, funct3, 0x63)))
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
        self.branch_word(rs1, rs2, &ops[2], funct3)
    }

    /// Branch pseudo-instructions: compare against `zero`, or swap the two
    /// registers for the `bgt`/`ble` family.
    fn encode_branch_pseudo(
        &self,
        opcode: &str,
        ops: &[String],
    ) -> Result<Option<Vec<u8>>, RasError> {
        let funct3 = match opcode {
            "beqz" => 0x0,
            "bnez" => 0x1,
            "bltz" | "bgtz" | "bgt" => 0x4,
            "bgez" | "blez" | "ble" => 0x5,
            "bgtu" => 0x6,
            "bleu" => 0x7,
            _ => return Ok(None),
        };
        let two_regs = matches!(opcode, "bgt" | "ble" | "bgtu" | "bleu");
        let expected = if two_regs { 3 } else { 2 };
        if ops.len() != expected {
            return Err(RasError::EncodingError(format!(
                "{} requires {} operands",
                opcode.to_uppercase(),
                expected
            )));
        }
        let (rs1, rs2, offset) = if two_regs {
            let rs = self.parse_register(&ops[0])?;
            let rt = self.parse_register(&ops[1])?;
            (rt, rs, &ops[2])
        } else {
            let rs = self.parse_register(&ops[0])?;
            let (rs1, rs2) = match opcode {
                "blez" | "bgtz" => (0, rs),
                _ => (rs, 0),
            };
            (rs1, rs2, &ops[1])
        };
        self.branch_word(rs1, rs2, offset, funct3).map(Some)
    }

    /// `fence` takes predecessor and successor sets, each a subset of `iorw`
    /// or a 4-bit number. With no operands it orders everything (`iorw, iorw`).
    fn encode_fence(&self, ops: &[String]) -> Result<Vec<u8>, RasError> {
        let (pred, succ) = match ops {
            [] => (0xF, 0xF),
            [pred, succ] => (Self::fence_set(pred)?, Self::fence_set(succ)?),
            _ => {
                return Err(RasError::EncodingError(
                    "FENCE requires 0 or 2 operands: pred, succ".to_string(),
                ));
            }
        };
        Ok(Self::emit((pred << 24) | (succ << 20) | 0x0F))
    }

    fn fence_set(s: &str) -> Result<u32, RasError> {
        let s = s.trim();
        if !s.is_empty() && s.chars().all(|c| "iorw".contains(c)) {
            return Ok(s.chars().fold(0, |bits, c| {
                bits | match c {
                    'i' => 8,
                    'o' => 4,
                    'r' => 2,
                    _ => 1,
                }
            }));
        }
        match s.parse::<u32>() {
            Ok(value) if value <= 0xF => Ok(value),
            _ => Err(RasError::EncodingError(format!("Invalid FENCE set: {}", s))),
        }
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

        if let Some(bytes) = self.encode_branch_pseudo(&opcode, ops)? {
            self.position += bytes.len();
            return Ok(bytes);
        }

        let bytes = match opcode.as_str() {
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

            "mul" => self.encode_r3(ops, 0x01, 0x0, 0x33)?,
            "mulh" => self.encode_r3(ops, 0x01, 0x1, 0x33)?,
            "mulhsu" => self.encode_r3(ops, 0x01, 0x2, 0x33)?,
            "mulhu" => self.encode_r3(ops, 0x01, 0x3, 0x33)?,
            "div" => self.encode_r3(ops, 0x01, 0x4, 0x33)?,
            "divu" => self.encode_r3(ops, 0x01, 0x5, 0x33)?,
            "rem" => self.encode_r3(ops, 0x01, 0x6, 0x33)?,
            "remu" => self.encode_r3(ops, 0x01, 0x7, 0x33)?,

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

            "addi" => self.encode_i3(ops, 0x0, 0x13)?,
            "slti" => self.encode_i3(ops, 0x2, 0x13)?,
            "sltiu" => self.encode_i3(ops, 0x3, 0x13)?,
            "xori" => self.encode_i3(ops, 0x4, 0x13)?,
            "ori" => self.encode_i3(ops, 0x6, 0x13)?,
            "andi" => self.encode_i3(ops, 0x7, 0x13)?,
            "slli" => self.encode_shift_imm(ops, 0x00, 0x1, 0x13)?,
            "srli" => self.encode_shift_imm(ops, 0x00, 0x5, 0x13)?,
            "srai" => self.encode_shift_imm(ops, 0x20, 0x5, 0x13)?,

            "addiw" => self.encode_i3(ops, 0x0, 0x1B)?,
            "slliw" => self.encode_shift_imm(ops, 0x00, 0x1, 0x1B)?,
            "srliw" => self.encode_shift_imm(ops, 0x00, 0x5, 0x1B)?,
            "sraiw" => self.encode_shift_imm(ops, 0x20, 0x5, 0x1B)?,

            "lb" => self.encode_load(ops, 0x0)?,
            "lh" => self.encode_load(ops, 0x1)?,
            "lw" => self.encode_load(ops, 0x2)?,
            "ld" => self.encode_load(ops, 0x3)?,
            "lbu" => self.encode_load(ops, 0x4)?,
            "lhu" => self.encode_load(ops, 0x5)?,
            "lwu" => self.encode_load(ops, 0x6)?,

            "sb" => self.encode_store(ops, 0x0)?,
            "sh" => self.encode_store(ops, 0x1)?,
            "sw" => self.encode_store(ops, 0x2)?,
            "sd" => self.encode_store(ops, 0x3)?,

            "beq" => self.encode_branch(ops, 0x0)?,
            "bne" => self.encode_branch(ops, 0x1)?,
            "blt" => self.encode_branch(ops, 0x4)?,
            "bge" => self.encode_branch(ops, 0x5)?,
            "bltu" => self.encode_branch(ops, 0x6)?,
            "bgeu" => self.encode_branch(ops, 0x7)?,

            "jal" => match ops.len() {
                1 => {
                    let offset = Self::jump_offset(self.parse_imm(&ops[0])?)?;
                    Self::emit(Self::j_type(offset, 1, 0x6F))
                }
                2 => {
                    let rd = self.parse_register(&ops[0])?;
                    let offset = Self::jump_offset(self.parse_imm(&ops[1])?)?;
                    Self::emit(Self::j_type(offset, rd, 0x6F))
                }
                _ => {
                    return Err(RasError::EncodingError(
                        "JAL requires 1 or 2 operands: [rd,] offset".to_string(),
                    ));
                }
            },

            "jalr" => {
                if ops.len() == 3 {
                    // jalr rd, rs1, imm  (explicit form)
                    let rd = self.parse_register(&ops[0])?;
                    let rs1 = self.parse_register(&ops[1])?;
                    let imm = self.imm12(&ops[2])?;
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

            "lui" => {
                if ops.len() != 2 {
                    return Err(RasError::EncodingError(
                        "LUI requires 2 operands: rd, imm20".to_string(),
                    ));
                }
                let rd = self.parse_register(&ops[0])?;
                let imm20 = self.imm20(&ops[1])?;
                Self::emit(Self::u_type(imm20, rd, 0x37))
            }
            "auipc" => {
                if ops.len() != 2 {
                    return Err(RasError::EncodingError(
                        "AUIPC requires 2 operands: rd, imm20".to_string(),
                    ));
                }
                let rd = self.parse_register(&ops[0])?;
                let imm20 = self.imm20(&ops[1])?;
                Self::emit(Self::u_type(imm20, rd, 0x17))
            }

            "ecall" => Self::emit(Self::i_type(0, 0, 0, 0, 0x73)),
            "ebreak" => Self::emit(Self::i_type(1, 0, 0, 0, 0x73)),
            "fence" => self.encode_fence(ops)?,
            "fence.i" => Self::emit(0x0000_100F),
            "fence.tso" => Self::emit(0x8330_000F),
            "pause" => Self::emit(0x0100_000F),

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
                let imm = self.parse_imm(&ops[1])?;
                if !(-2048..=2047).contains(&imm) {
                    return Err(RasError::EncodingError(format!(
                        "LI pseudo-instruction only supports 12-bit immediates ({} out of range)",
                        imm
                    )));
                }
                Self::emit(Self::i_type(imm as i32, 0, 0, rd, 0x13))
            }
            // `j offset` = jal x0, offset
            "j" => {
                if ops.len() != 1 {
                    return Err(RasError::EncodingError(
                        "J requires 1 operand: offset".to_string(),
                    ));
                }
                let offset = Self::jump_offset(self.parse_imm(&ops[0])?)?;
                Self::emit(Self::j_type(offset, 0, 0x6F))
            }
            // `call offset` = jal ra, offset
            "call" => {
                if ops.len() != 1 {
                    return Err(RasError::EncodingError(
                        "CALL requires 1 operand: offset".to_string(),
                    ));
                }
                let offset = Self::jump_offset(self.parse_imm(&ops[0])?)?;
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
        let bytes = enc()
            .encode_instruction(&instr("add", &["a0", "a1", "a2"]))
            .unwrap();
        assert_eq!(bytes.len(), 4);
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x33);
        assert_eq!((word >> 12) & 7, 0);
        assert_eq!((word >> 25) & 0x7F, 0);
        assert_eq!((word >> 7) & 0x1F, 10);
    }

    #[test]
    fn test_addi_encoding() {
        let bytes = enc()
            .encode_instruction(&instr("addi", &["t0", "zero", "42"]))
            .unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x13);
        assert_eq!((word >> 7) & 0x1F, 5);
        assert_eq!((word >> 20) as i32, 42);
    }

    #[test]
    fn test_ret_pseudo() {
        let bytes = enc().encode_instruction(&instr("ret", &[])).unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x67);
        assert_eq!((word >> 7) & 0x1F, 0);
        assert_eq!((word >> 15) & 0x1F, 1);
        assert_eq!(word >> 20, 0);
    }

    #[test]
    fn test_load_store_encoding() {
        let bytes = enc()
            .encode_instruction(&instr("sw", &["a0", "0(sp)"]))
            .unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x23);
        assert_eq!((word >> 12) & 7, 0x2);

        let bytes = enc()
            .encode_instruction(&instr("lw", &["a1", "4(sp)"]))
            .unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(word & 0x7F, 0x03);
        assert_eq!((word >> 12) & 7, 0x2);
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
    fn test_fence_sets_and_fence_i() {
        let word = |opcode: &str, ops: &[&str]| {
            let bytes = enc().encode_instruction(&instr(opcode, ops)).unwrap();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        };
        assert_eq!(word("fence", &[]), 0x0FF0_000F);
        assert_eq!(word("fence", &["rw", "w"]), 0x0310_000F);
        assert_eq!(word("fence", &["iorw", "i"]), 0x0F80_000F);
        assert_eq!(word("fence.i", &[]), 0x0000_100F);
        assert_eq!(word("fence.tso", &[]), 0x8330_000F);
    }

    #[test]
    fn test_branch_and_jump_offset_limits() {
        assert!(
            enc()
                .encode_instruction(&instr("beq", &["a0", "a1", "4094"]))
                .is_ok()
        );
        assert!(
            enc()
                .encode_instruction(&instr("beq", &["a0", "a1", "-4096"]))
                .is_ok()
        );
        assert!(
            enc()
                .encode_instruction(&instr("beq", &["a0", "a1", "4096"]))
                .is_err()
        );
        assert!(
            enc()
                .encode_instruction(&instr("beq", &["a0", "a1", "3"]))
                .is_err()
        );
        assert!(
            enc()
                .encode_instruction(&instr("jal", &["ra", "1048574"]))
                .is_ok()
        );
        assert!(
            enc()
                .encode_instruction(&instr("jal", &["ra", "1048576"]))
                .is_err()
        );
        assert!(enc().encode_instruction(&instr("j", &["-1048576"])).is_ok());
        assert!(
            enc()
                .encode_instruction(&instr("j", &["-1048578"]))
                .is_err()
        );
    }

    #[test]
    fn test_mul_encoding() {
        let bytes = enc()
            .encode_instruction(&instr("mul", &["a0", "a1", "a2"]))
            .unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!((word >> 25) & 0x7F, 1);
    }
}
