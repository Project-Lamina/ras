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
    fn encode_atomic(&self, name: &str, ops: &[String]) -> Result<Vec<u8>, RasError> {
        let Some((funct5, funct3, ordering)) = atomic_encoding(name) else {
            return Err(RasError::EncodingError(format!(
                "Unknown RISC-V instruction: {name}"
            )));
        };
        let is_lr = funct5 == 0b00010;
        let (rd, rs2, mem) = match (is_lr, ops) {
            (true, [rd, mem]) => (rd, "zero", mem),
            (false, [rd, rs2, mem]) => (rd, rs2.as_str(), mem),
            _ => {
                return Err(RasError::EncodingError(format!(
                    "{name} takes {}",
                    if is_lr { "rd, (rs1)" } else { "rd, rs2, (rs1)" }
                )));
            }
        };
        let (rs1, offset) = self.parse_mem_operand(mem)?;
        if offset != 0 {
            return Err(RasError::EncodingError(format!(
                "{name} takes no address offset"
            )));
        }
        let rd = self.parse_register(rd)?;
        let rs2 = self.parse_register(rs2)?;
        Ok(Self::emit(Self::r_type(
            (funct5 << 2) | ordering,
            rs2,
            rs1,
            funct3,
            rd,
            0x2F,
        )))
    }

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
    /// `li rd, imm`: the shortest lui/addi(w)/slli/srli sequence for the value.
    fn encode_li(&self, ops: &[String]) -> Result<Vec<u8>, RasError> {
        if ops.len() != 2 {
            return Err(RasError::EncodingError(
                "LI requires 2 operands: rd, imm".to_string(),
            ));
        }
        let rd = self.parse_register(&ops[0])?;
        let value = parse_li_immediate(&ops[1])?;
        if !self.rv64 && !fits_signed(value as i128, 32) {
            return Err(RasError::EncodingError(format!(
                "LI immediate {value} does not fit 32 bits on RV32"
            )));
        }
        let mut bytes = Vec::new();
        let mut source = 0u8;
        for step in li_expansion(value, self.rv64) {
            let word = match step {
                LiStep::Lui(imm) => Self::u_type(imm, rd, 0x37),
                LiStep::Addi(imm) => Self::i_type(imm, source, 0x0, rd, 0x13),
                LiStep::Addiw(imm) => Self::i_type(imm, source, 0x0, rd, 0x1B),
                LiStep::Slli(shamt) => Self::i_type(shamt as i32, source, 0x1, rd, 0x13),
                LiStep::Srli(shamt) => Self::i_type(shamt as i32, source, 0x5, rd, 0x13),
            };
            bytes.extend(Self::emit(word));
            source = rd;
        }
        Ok(bytes)
    }

    fn parse_fregister(&self, s: &str) -> Result<u8, RasError> {
        let s = s.trim().trim_start_matches('%');
        fp_register(s).ok_or_else(|| {
            RasError::EncodingError(format!("Unknown RISC-V floating-point register: {s}"))
        })
    }

    /// F and D extensions. In OP-FP, funct7 is `funct5 << 2 | fmt`, with fmt 0
    /// for single and 1 for double, and the rounding mode sits in funct3.
    /// Returns `None` for names this encoder does not own.
    fn encode_fp(&self, name: &str, ops: &[String]) -> Result<Option<Vec<u8>>, RasError> {
        const OP_FP: u8 = 0x53;
        let f = |s: &String| self.parse_fregister(s);
        let x = |s: &String| self.parse_register(s);
        let word = match name {
            "flw" | "fld" => {
                let [rd, mem] = ops else {
                    return Err(fp_operands(name, "rd, offset(rs1)"));
                };
                let funct3 = if name == "flw" { 0x2 } else { 0x3 };
                let rd = f(rd)?;
                let (rs1, offset) = self.parse_mem_operand(mem)?;
                Self::i_type(offset, rs1, funct3, rd, 0x07)
            }
            "fsw" | "fsd" => {
                let [rs2, mem] = ops else {
                    return Err(fp_operands(name, "rs2, offset(rs1)"));
                };
                let funct3 = if name == "fsw" { 0x2 } else { 0x3 };
                let rs2 = f(rs2)?;
                let (rs1, offset) = self.parse_mem_operand(mem)?;
                Self::s_type(offset, rs2, rs1, funct3, 0x27)
            }
            "fmv.x.w" | "fmv.x.d" => {
                let [rd, rs1] = ops else {
                    return Err(fp_operands(name, "rd, rs1"));
                };
                let funct7 = if name == "fmv.x.w" { 0x70 } else { 0x71 };
                Self::r_type(funct7, 0, f(rs1)?, 0, x(rd)?, OP_FP)
            }
            "fmv.w.x" | "fmv.d.x" => {
                let [rd, rs1] = ops else {
                    return Err(fp_operands(name, "rd, rs1"));
                };
                let funct7 = if name == "fmv.w.x" { 0x78 } else { 0x79 };
                Self::r_type(funct7, 0, x(rs1)?, 0, f(rd)?, OP_FP)
            }
            _ => {
                if let Some((funct7, rs2, int_dst, int_src)) = fcvt_encoding(name) {
                    let [rd, rs1, rest @ ..] = ops else {
                        return Err(fp_operands(name, "rd, rs1[, rm]"));
                    };
                    let default = if funct7 == 0x21 { 0 } else { 0b111 };
                    let rm = fp_rounding(rest, default)?;
                    let rd = if int_dst { x(rd)? } else { f(rd)? };
                    let rs1 = if int_src { x(rs1)? } else { f(rs1)? };
                    return Ok(Some(Self::emit(Self::r_type(
                        funct7, rs2, rs1, rm, rd, OP_FP,
                    ))));
                }
                let Some((base, fmt)) = split_format(name) else {
                    return Ok(None);
                };
                match base {
                    "fadd" | "fsub" | "fmul" | "fdiv" => {
                        let [rd, rs1, rs2, rest @ ..] = ops else {
                            return Err(fp_operands(name, "rd, rs1, rs2[, rm]"));
                        };
                        let funct5 = match base {
                            "fadd" => 0x00,
                            "fsub" => 0x01,
                            "fmul" => 0x02,
                            _ => 0x03,
                        };
                        let rm = fp_rounding(rest, 0b111)?;
                        Self::r_type((funct5 << 2) | fmt, f(rs2)?, f(rs1)?, rm, f(rd)?, OP_FP)
                    }
                    "fsqrt" => {
                        let [rd, rs1, rest @ ..] = ops else {
                            return Err(fp_operands(name, "rd, rs1[, rm]"));
                        };
                        let rm = fp_rounding(rest, 0b111)?;
                        Self::r_type((0x0b << 2) | fmt, 0, f(rs1)?, rm, f(rd)?, OP_FP)
                    }
                    "fsgnj" | "fsgnjn" | "fsgnjx" | "fmin" | "fmax" => {
                        let [rd, rs1, rs2] = ops else {
                            return Err(fp_operands(name, "rd, rs1, rs2"));
                        };
                        let (funct5, rm) = match base {
                            "fsgnj" => (0x04, 0),
                            "fsgnjn" => (0x04, 1),
                            "fsgnjx" => (0x04, 2),
                            "fmin" => (0x05, 0),
                            _ => (0x05, 1),
                        };
                        Self::r_type((funct5 << 2) | fmt, f(rs2)?, f(rs1)?, rm, f(rd)?, OP_FP)
                    }
                    "feq" | "flt" | "fle" => {
                        let [rd, rs1, rs2] = ops else {
                            return Err(fp_operands(name, "rd, rs1, rs2"));
                        };
                        let rm = match base {
                            "fle" => 0,
                            "flt" => 1,
                            _ => 2,
                        };
                        Self::r_type(0x50 | fmt, f(rs2)?, f(rs1)?, rm, x(rd)?, OP_FP)
                    }
                    "fclass" => {
                        let [rd, rs1] = ops else {
                            return Err(fp_operands(name, "rd, rs1"));
                        };
                        Self::r_type(0x70 | fmt, 0, f(rs1)?, 1, x(rd)?, OP_FP)
                    }
                    "fmv" | "fneg" | "fabs" => {
                        let [rd, rs] = ops else {
                            return Err(fp_operands(name, "rd, rs"));
                        };
                        let rm = match base {
                            "fmv" => 0,
                            "fneg" => 1,
                            _ => 2,
                        };
                        let (rd, rs) = (f(rd)?, f(rs)?);
                        Self::r_type(0x04 << 2 | fmt, rs, rs, rm, rd, OP_FP)
                    }
                    _ => return Ok(None),
                }
            }
        };
        Ok(Some(Self::emit(word)))
    }

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

fn fp_operands(name: &str, shape: &str) -> RasError {
    RasError::EncodingError(format!(
        "{} requires operands: {shape}",
        name.to_uppercase()
    ))
}

/// The optional trailing rounding mode, `default` when omitted.
fn fp_rounding(rest: &[String], default: u8) -> Result<u8, RasError> {
    let text = match rest {
        [] => return Ok(default),
        [text] => text.trim(),
        _ => {
            return Err(RasError::EncodingError(
                "Too many operands for an FP instruction".into(),
            ));
        }
    };
    Ok(match text {
        "rne" => 0,
        "rtz" => 1,
        "rdn" => 2,
        "rup" => 3,
        "rmm" => 4,
        "dyn" => 7,
        other => {
            return Err(RasError::EncodingError(format!(
                "Unknown RISC-V rounding mode: {other}"
            )));
        }
    })
}

/// The `.s` / `.d` suffix of an F or D mnemonic, with the base name before it.
fn split_format(name: &str) -> Option<(&str, u8)> {
    let (base, suffix) = name.rsplit_once('.')?;
    match suffix {
        "s" => Some((base, 0)),
        "d" => Some((base, 1)),
        _ => None,
    }
}

/// `fcvt.<dst>.<src>` as `(funct7, rs2, rd is integer, rs1 is integer)`.
/// The integer selector in rs2 is 0 for W, 1 for WU, 2 for L and 3 for LU.
fn fcvt_encoding(name: &str) -> Option<(u8, u8, bool, bool)> {
    let (dst, src) = name.strip_prefix("fcvt.")?.split_once('.')?;
    let integer = |t: &str| match t {
        "w" => Some(0u8),
        "wu" => Some(1),
        "l" => Some(2),
        "lu" => Some(3),
        _ => None,
    };
    let format = |t: &str| match t {
        "s" => Some(0u8),
        "d" => Some(1),
        _ => None,
    };
    if let (Some(sel), Some(fmt)) = (integer(dst), format(src)) {
        return Some((0x60 | fmt, sel, true, false));
    }
    if let (Some(fmt), Some(sel)) = (format(dst), integer(src)) {
        return Some((0x68 | fmt, sel, false, true));
    }
    match (format(dst), format(src)) {
        (Some(dst_fmt), Some(src_fmt)) if dst_fmt != src_fmt => {
            Some((0x20 | dst_fmt, src_fmt, false, false))
        }
        _ => None,
    }
}

/// An `f` register by number or ABI name (ft0-11, fs0-11, fa0-7).
fn fp_register(name: &str) -> Option<u8> {
    if let Some(number) = name.strip_prefix('f').and_then(|n| n.parse::<u8>().ok()) {
        return (number < 32).then_some(number);
    }
    let (class, index) = name.split_at_checked(2)?;
    let index: u8 = index.parse().ok()?;
    match (class, index) {
        ("ft", 0..=7) => Some(index),
        ("ft", 8..=11) => Some(index + 20),
        ("fs", 0..=1) => Some(index + 8),
        ("fs", 2..=11) => Some(index + 16),
        ("fa", 0..=7) => Some(index + 10),
        _ => None,
    }
}

/// Single-instruction ALU and jump pseudo-instructions, rewritten to the base
/// mnemonic they stand for.
fn atomic_encoding(name: &str) -> Option<(u8, u8, u8)> {
    let (base, ordering) = [(".aqrl", 3), (".aq", 2), (".rl", 1)]
        .into_iter()
        .find_map(|(suffix, bits)| Some((name.strip_suffix(suffix)?, bits)))
        .unwrap_or((name, 0));
    let (op, width) = base.rsplit_once('.')?;
    let funct3 = match width {
        "w" => 0b010,
        "d" => 0b011,
        _ => return None,
    };
    let funct5 = match op {
        "lr" => 0b00010,
        "sc" => 0b00011,
        "amoswap" => 0b00001,
        "amoadd" => 0b00000,
        "amoxor" => 0b00100,
        "amoand" => 0b01100,
        "amoor" => 0b01000,
        "amomin" => 0b10000,
        "amomax" => 0b10100,
        "amominu" => 0b11000,
        "amomaxu" => 0b11100,
        _ => return None,
    };
    Some((funct5, funct3, ordering))
}

fn expand_alu_pseudo(
    opcode: &str,
    ops: &[String],
) -> Result<Option<(&'static str, Vec<String>)>, RasError> {
    let (base, operands): (&'static str, Vec<String>) = match (opcode, ops) {
        ("not", [rd, rs]) => ("xori", vec![rd.clone(), rs.clone(), "-1".into()]),
        ("neg", [rd, rs]) => ("sub", vec![rd.clone(), "zero".into(), rs.clone()]),
        ("negw", [rd, rs]) => ("subw", vec![rd.clone(), "zero".into(), rs.clone()]),
        ("sext.w", [rd, rs]) => ("addiw", vec![rd.clone(), rs.clone(), "0".into()]),
        ("seqz", [rd, rs]) => ("sltiu", vec![rd.clone(), rs.clone(), "1".into()]),
        ("snez", [rd, rs]) => ("sltu", vec![rd.clone(), "zero".into(), rs.clone()]),
        ("sltz", [rd, rs]) => ("slt", vec![rd.clone(), rs.clone(), "zero".into()]),
        ("sgtz", [rd, rs]) => ("slt", vec![rd.clone(), "zero".into(), rs.clone()]),
        ("sgt", [rd, rs, rt]) => ("slt", vec![rd.clone(), rt.clone(), rs.clone()]),
        ("sgtu", [rd, rs, rt]) => ("sltu", vec![rd.clone(), rt.clone(), rs.clone()]),
        ("jr", [rs]) => ("jalr", vec!["zero".into(), format!("0({rs})")]),
        (
            "not" | "neg" | "negw" | "sext.w" | "seqz" | "snez" | "sltz" | "sgtz" | "sgt" | "sgtu"
            | "jr",
            _,
        ) => {
            return Err(RasError::EncodingError(format!(
                "{} has the wrong number of operands",
                opcode.to_uppercase()
            )));
        }
        _ => return Ok(None),
    };
    Ok(Some((base, operands)))
}

/// One instruction of a `li` expansion, in emission order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiStep {
    Lui(i32),
    Addi(i32),
    Addiw(i32),
    Slli(u32),
    Srli(u32),
}

fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

fn fits_signed(value: i128, bits: u32) -> bool {
    let half = 1i128 << (bits - 1);
    (-half..half).contains(&value)
}

fn li_base(value: i64, rv64: bool) -> Vec<LiStep> {
    if fits_signed(value as i128, 32) {
        let hi20 = ((value + 0x800) >> 12) & 0xF_FFFF;
        let lo12 = sign_extend(value as u64, 12);
        let mut steps = Vec::new();
        if hi20 != 0 {
            steps.push(LiStep::Lui(hi20 as i32));
        }
        if lo12 != 0 || hi20 == 0 {
            steps.push(if rv64 && hi20 != 0 {
                LiStep::Addiw(lo12 as i32)
            } else {
                LiStep::Addi(lo12 as i32)
            });
        }
        return steps;
    }
    let lo12 = sign_extend(value as u64, 12);
    let hi52 = (value as u64).wrapping_add(0x800) >> 12;
    let shift = 12 + hi52.trailing_zeros();
    let hi = sign_extend(hi52 >> (shift - 12), 64 - shift);
    let mut steps = li_base(hi, rv64);
    steps.push(LiStep::Slli(shift));
    if lo12 != 0 {
        steps.push(LiStep::Addi(lo12 as i32));
    }
    steps
}

fn li_sequence(value: i64, rv64: bool) -> Vec<LiStep> {
    let mut best = li_base(value, rv64);
    if value > 0 && best.len() > 2 {
        let lz = value.leading_zeros();
        for fill in [(1u64 << lz) - 1, 0] {
            let shifted = ((value as u64) << lz) | fill;
            let mut cand = li_base(shifted as i64, rv64);
            cand.push(LiStep::Srli(lz));
            if cand.len() < best.len() {
                best = cand;
            }
        }
    }
    if value != 0 {
        let tz = value.trailing_zeros();
        if tz > 0 {
            let shifted = value >> tz;
            let mut cand = li_sequence(shifted, rv64);
            cand.push(LiStep::Slli(tz));
            if cand.len() < best.len()
                || (cand.len() == best.len() && fits_signed(shifted as i128, 6))
            {
                best = cand;
            }
        }
    }
    best
}

fn li_expansion(value: i64, rv64: bool) -> Vec<LiStep> {
    let mut best = li_sequence(value, rv64);
    if value <= 0 || fits_signed(value as i128, 32) {
        return best;
    }
    let lo12 = sign_extend(value as u64, 12);
    let rest = value as i128 - lo12 as i128;
    if rest != 0 && rest.trailing_zeros() >= 12 {
        let tz = rest.trailing_zeros();
        let upper = rest >> tz;
        if fits_signed(upper, 20) {
            let mut cand = vec![LiStep::Lui(upper as i32)];
            if tz > 12 {
                cand.push(LiStep::Slli(tz - 12));
            }
            if lo12 != 0 {
                cand.push(LiStep::Addi(lo12 as i32));
            }
            if cand.len() < best.len() {
                best = cand;
            }
        }
    }
    best
}

/// Parses a `li` immediate: optional sign, decimal or 0x hex, up to the full
/// 64-bit range (unsigned values wrap to their two's-complement bit pattern).
fn parse_li_immediate(s: &str) -> Result<i64, RasError> {
    let s = s.trim().trim_start_matches('#');
    let (negative, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest.trim_start()),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let magnitude = match body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => body.parse::<u64>(),
    }
    .map_err(|_| RasError::EncodingError(format!("Invalid immediate: {s}")))?;
    let value = if negative {
        -(magnitude as i128)
    } else {
        magnitude as i128
    };
    if value < i64::MIN as i128 {
        return Err(RasError::EncodingError(format!(
            "LI immediate {s} does not fit 64 bits"
        )));
    }
    Ok(value as i64)
}

impl InstructionEncoder for RiscVEncoder {
    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let opcode = inst.opcode.to_lowercase();
        let ops = &inst.operands;

        if let Some(bytes) = self.encode_branch_pseudo(&opcode, ops)? {
            self.position += bytes.len();
            return Ok(bytes);
        }
        if let Some((base, operands)) = expand_alu_pseudo(&opcode, ops)? {
            return self.encode_instruction(&ParsedInstruction {
                opcode: base.to_string(),
                operands,
            });
        }
        if let Some(bytes) = self.encode_fp(&opcode, ops)? {
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
            "li" => self.encode_li(ops)?,
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

            name if atomic_encoding(name).is_some() => self.encode_atomic(name, ops)?,
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
    fn test_fp_and_compare_pseudos_match_spec_fields() {
        let word = |opcode: &str, ops: &[&str]| {
            let bytes = enc().encode_instruction(&instr(opcode, ops)).unwrap();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        };
        assert_eq!(word("fadd.d", &["fa0", "fa1", "fa2"]), 0x02C5_F553);
        assert_eq!(word("fld", &["fa0", "8(sp)"]), 0x0081_3507);
        assert_eq!(word("fsd", &["fa0", "8(sp)"]), 0x00A1_3427);
        assert_eq!(word("fmv.d.x", &["fa0", "t0"]), 0xF202_8553);
        assert_eq!(word("fmv.w.x", &["fa0", "t0"]), 0xF002_8553);
        assert_eq!(word("fcvt.l.d", &["a0", "fa0", "rtz"]), 0xC225_1553);
        assert_eq!(word("fcvt.d.l", &["fa0", "a0"]), 0xD225_7553);
        assert_eq!(word("feq.d", &["a0", "fa0", "fa1"]), 0xA2B5_2553);
        assert_eq!(word("fneg.d", &["fa0", "fa0"]), 0x22A5_1553);
        assert_eq!(word("fcvt.d.s", &["fa0", "fa0"]), 0x4205_0553);
        assert_eq!(word("sgt", &["a0", "a0", "a1"]), 0x00A5_A533);
        assert_eq!(word("sgtu", &["a0", "a0", "a1"]), 0x00A5_B533);
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
    fn test_unary_pseudos_match_their_base_instruction() {
        let pairs = [
            (
                ("not", &["a0", "a1"][..]),
                ("xori", &["a0", "a1", "-1"][..]),
            ),
            (
                ("neg", &["a0", "a1"][..]),
                ("sub", &["a0", "zero", "a1"][..]),
            ),
            (
                ("seqz", &["a0", "a1"][..]),
                ("sltiu", &["a0", "a1", "1"][..]),
            ),
            (
                ("sgtz", &["a0", "a1"][..]),
                ("slt", &["a0", "zero", "a1"][..]),
            ),
            (("jr", &["ra"][..]), ("jalr", &["zero", "0(ra)"][..])),
        ];
        for ((pseudo, pseudo_ops), (base, base_ops)) in pairs {
            assert_eq!(
                enc()
                    .encode_instruction(&instr(pseudo, pseudo_ops))
                    .unwrap(),
                enc().encode_instruction(&instr(base, base_ops)).unwrap(),
                "{pseudo}"
            );
        }
    }

    #[test]
    fn test_mul_encoding() {
        let bytes = enc()
            .encode_instruction(&instr("mul", &["a0", "a1", "a2"]))
            .unwrap();
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!((word >> 25) & 0x7F, 1);
    }

    #[test]
    fn atomic_encodings_match_unprivileged_spec() {
        let word = |op: &str, ops: &[&str]| {
            let bytes = enc().encode_instruction(&instr(op, ops)).unwrap();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        };
        assert_eq!(word("lr.w", &["a0", "(a1)"]), 0x1005_A52F);
        assert_eq!(word("amoadd.d.aqrl", &["a0", "a2", "(a1)"]), 0x06C5_B52F);
        assert_eq!(word("sc.w.rl", &["a3", "a4", "(a5)"]), 0x1AE7_A6AF);
    }
}
