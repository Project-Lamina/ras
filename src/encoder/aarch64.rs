//! AArch64 binary instruction encoder

// The binary literals in this file use field-aligned groupings that mirror the
// AArch64 instruction encoding spec (e.g. op|S|Rn|imm12|Rd).  Reformatting
// them into standard 4-bit nibble groups would destroy that readability.
#![allow(clippy::unusual_byte_groupings)]
// Several field OR expressions have a `| (0 << k)` term for documentation;
// they are no-ops but make the encoding explicit.
#![allow(clippy::identity_op)]

use crate::aarch64_ldst_imm64::{encode_ldr_imm64, encode_str_imm64};
use crate::encoder::traits::{InstructionEncoder, ParsedInstruction};
use crate::error::RasError;

pub struct AArch64Encoder {
    position: usize,
}

impl Default for AArch64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AArch64Encoder {
    pub fn new() -> Self {
        Self { position: 0 }
    }

    fn parse_register(&self, reg: &str) -> Result<u8, RasError> {
        let reg = reg.trim_start_matches('%');
        match reg {
            "x0" | "w0" => Ok(0),
            "x1" | "w1" => Ok(1),
            "x2" | "w2" => Ok(2),
            "x3" | "w3" => Ok(3),
            "x4" | "w4" => Ok(4),
            "x5" | "w5" => Ok(5),
            "x6" | "w6" => Ok(6),
            "x7" | "w7" => Ok(7),
            "x8" | "w8" => Ok(8),
            "x9" | "w9" => Ok(9),
            "x10" | "w10" => Ok(10),
            "x11" | "w11" => Ok(11),
            "x12" | "w12" => Ok(12),
            "x13" | "w13" => Ok(13),
            "x14" | "w14" => Ok(14),
            "x15" | "w15" => Ok(15),
            "x16" | "w16" | "ip0" => Ok(16),
            "x17" | "w17" | "ip1" => Ok(17),
            "x18" | "w18" => Ok(18),
            "x19" | "w19" => Ok(19),
            "x20" | "w20" => Ok(20),
            "x21" | "w21" => Ok(21),
            "x22" | "w22" => Ok(22),
            "x23" | "w23" => Ok(23),
            "x24" | "w24" => Ok(24),
            "x25" | "w25" => Ok(25),
            "x26" | "w26" => Ok(26),
            "x27" | "w27" => Ok(27),
            "x28" | "w28" => Ok(28),
            "x29" | "w29" | "fp" => Ok(29),
            "x30" | "w30" | "lr" => Ok(30),
            "x31" | "w31" | "sp" | "xzr" | "wzr" => Ok(31),
            _ => Err(RasError::EncodingError(format!(
                "Unknown register: {}",
                reg
            ))),
        }
    }

    fn encode_u32(&self, inst: u32) -> Vec<u8> {
        inst.to_le_bytes().to_vec()
    }

    fn encode_mov_reg(&self, dst: u8, src: u8) -> Vec<u8> {
        let inst = 0b1_01_01010_0_0_000000_11111_00000_00000;
        let inst = inst | (dst as u32);
        let inst = inst | ((31u32) << 5);
        let inst = inst | ((src as u32) << 16);
        self.encode_u32(inst)
    }

    fn encode_mov_imm64(&self, dst: u8, imm: u64) -> Result<Vec<u8>, RasError> {
        if imm <= 0xFFFF {
            let inst = 0b1_100101_10_0000000000000000_00000;
            let inst = inst | (dst as u32);
            let inst = inst | ((imm as u32 & 0xFFFF) << 5);
            Ok(self.encode_u32(inst))
        } else if (imm & 0xFFFF) == 0 && (imm >> 16) <= 0xFFFF {
            let inst = 0b1_100101_10_01_0000000000000000_00000;
            let inst = inst | (dst as u32);
            let inst = inst | (((imm >> 16) as u32 & 0xFFFF) << 5);
            Ok(self.encode_u32(inst))
        } else {
            Err(RasError::EncodingError(
                "Large immediate values require MOVZ+MOVK sequence (not yet implemented)"
                    .to_string(),
            ))
        }
    }

    fn encode_add_reg(&self, dst: u8, src1: u8, src2: u8) -> Vec<u8> {
        // ADD (shifted register), 64-bit: base 0x8B000000, Rm at 20:16.
        let inst = 0x8B00_0000 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.encode_u32(inst)
    }

    fn encode_add_imm(&self, dst: u8, src: u8, imm: u32) -> Result<Vec<u8>, RasError> {
        if imm > 0xFFF {
            return Err(RasError::EncodingError(
                "ADD immediate must be 12 bits or less".to_string(),
            ));
        }
        let inst = 0b1_0_0_100010_0_000000000000_00000_00000;
        let inst = inst | (dst as u32);
        let inst = inst | ((src as u32) << 5);
        let inst = inst | ((imm & 0xFFF) << 10);
        Ok(self.encode_u32(inst))
    }

    fn encode_str(&self, src: u8, base: u8, offset: i32) -> Result<Vec<u8>, RasError> {
        encode_str_imm64(src, base, offset)
    }

    fn encode_ldr(&self, dst: u8, base: u8, offset: i32) -> Result<Vec<u8>, RasError> {
        encode_ldr_imm64(dst, base, offset)
    }

    fn encode_ret(&self, reg: u8) -> Vec<u8> {
        self.encode_u32(0xD65F_0000 | ((reg as u32) << 5))
    }
}

impl InstructionEncoder for AArch64Encoder {
    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let opcode = inst.opcode.to_lowercase();
        let mut code = Vec::new();

        match opcode.as_str() {
            "mov" | "movz" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "mov requires 2 operands".to_string(),
                    ));
                }
                let dst = &inst.operands[0];
                let src = &inst.operands[1];
                let dst_reg = self.parse_register(dst)?;

                if src.starts_with('#') {
                    let imm: u64 = src
                        .trim_start_matches('#')
                        .parse()
                        .map_err(|_| RasError::EncodingError("Invalid immediate".to_string()))?;
                    code.extend_from_slice(&self.encode_mov_imm64(dst_reg, imm)?);
                } else {
                    let src_reg = self.parse_register(src)?;
                    code.extend_from_slice(&self.encode_mov_reg(dst_reg, src_reg));
                }
            }
            "ret" => {
                if inst.operands.is_empty() {
                    code.extend_from_slice(&self.encode_ret(30));
                } else {
                    let reg = self.parse_register(&inst.operands[0])?;
                    code.extend_from_slice(&self.encode_ret(reg));
                }
            }
            "add" => {
                if inst.operands.len() != 3 {
                    return Err(RasError::EncodingError(
                        "add requires 3 operands".to_string(),
                    ));
                }
                let dst = &inst.operands[0];
                let src1 = &inst.operands[1];
                let src2 = &inst.operands[2];
                let dst_reg = self.parse_register(dst)?;
                let src1_reg = self.parse_register(src1)?;

                if src2.starts_with('#') {
                    let imm: u32 = src2
                        .trim_start_matches('#')
                        .parse()
                        .map_err(|_| RasError::EncodingError("Invalid immediate".to_string()))?;
                    code.extend_from_slice(&self.encode_add_imm(dst_reg, src1_reg, imm)?);
                } else {
                    let src2_reg = self.parse_register(src2)?;
                    code.extend_from_slice(&self.encode_add_reg(dst_reg, src1_reg, src2_reg));
                }
            }
            "str" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "str requires 2 operands".to_string(),
                    ));
                }
                let src = &inst.operands[0];
                let mem = &inst.operands[1];
                let src_reg = self.parse_register(src)?;

                if mem.starts_with('[') && mem.ends_with(']') {
                    let inner = &mem[1..mem.len() - 1];
                    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
                    let base_reg = self.parse_register(parts[0])?;
                    let offset = if parts.len() > 1 {
                        parts[1]
                            .trim_start_matches('#')
                            .parse::<i32>()
                            .map_err(|_| RasError::EncodingError("Invalid offset".to_string()))?
                    } else {
                        0
                    };
                    code.extend_from_slice(&self.encode_str(src_reg, base_reg, offset)?);
                } else {
                    return Err(RasError::EncodingError(
                        "str requires memory operand [reg, #offset]".to_string(),
                    ));
                }
            }
            "ldr" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "ldr requires 2 operands".to_string(),
                    ));
                }
                let dst = &inst.operands[0];
                let mem = &inst.operands[1];
                let dst_reg = self.parse_register(dst)?;

                if mem.starts_with('[') && mem.ends_with(']') {
                    let inner = &mem[1..mem.len() - 1];
                    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
                    let base_reg = self.parse_register(parts[0])?;
                    let offset = if parts.len() > 1 {
                        parts[1]
                            .trim_start_matches('#')
                            .parse::<i32>()
                            .map_err(|_| RasError::EncodingError("Invalid offset".to_string()))?
                    } else {
                        0
                    };
                    code.extend_from_slice(&self.encode_ldr(dst_reg, base_reg, offset)?);
                } else {
                    return Err(RasError::EncodingError(
                        "ldr requires memory operand [reg, #offset]".to_string(),
                    ));
                }
            }
            _ => {
                return Err(RasError::EncodingError(format!(
                    "Unsupported instruction: {}",
                    opcode
                )));
            }
        }

        self.position += code.len();
        Ok(code)
    }

    fn current_position(&self) -> usize {
        self.position
    }
}

#[cfg(test)]
mod aarch64_ldr_str_encoder_tests {
    use super::*;

    fn encode_ldr_line(base: &str, dst: &str, offset: i32) -> Vec<u8> {
        let mem = format!("[{}, #{}]", base, offset);
        let mut enc = AArch64Encoder::new();
        enc.encode_instruction(&ParsedInstruction {
            opcode: "ldr".into(),
            operands: vec![dst.into(), mem],
        })
        .expect("ldr encode")
    }

    fn encode_str_line(base: &str, src: &str, offset: i32) -> Vec<u8> {
        let mem = format!("[{}, #{}]", base, offset);
        let mut enc = AArch64Encoder::new();
        enc.encode_instruction(&ParsedInstruction {
            opcode: "str".into(),
            operands: vec![src.into(), mem],
        })
        .expect("str encode")
    }

    #[test]
    fn ldr_scaled_1024_byte_offset_uses_imm12_times_eight() {
        let b = encode_ldr_line("x1", "x0", 1024);
        assert_eq!(b, vec![0x20, 0x00, 0x42, 0xF9]);
    }

    #[test]
    fn str_scaled_512_byte_offset_not_ldur_range() {
        let b = encode_str_line("x3", "x2", 512);
        assert_eq!(b, vec![0x62, 0x00, 0x01, 0xF9]);
    }

    #[test]
    fn ldr_unscaled_small_positive_uses_ldur() {
        let b = encode_ldr_line("x5", "x4", 24);
        assert_eq!(b.len(), 4);
        assert_eq!(b, vec![0xA4, 0x80, 0x41, 0xF8]);
    }

    #[test]
    fn ldr_unaligned_positive_beyond_ldur_range_errors() {
        let mem = "[x1, #260]";
        let mut enc = AArch64Encoder::new();
        let err = enc
            .encode_instruction(&ParsedInstruction {
                opcode: "ldr".into(),
                operands: vec!["x0".into(), mem.into()],
            })
            .expect_err("260 not multiple of 8 and > 255");
        match err {
            RasError::EncodingError(msg) => {
                assert!(msg.contains("260"), "{}", msg);
            }
            _ => panic!("expected EncodingError"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_reg_is_an_add_not_a_branch() {
        let enc = AArch64Encoder::new();
        assert_eq!(
            enc.encode_add_reg(0, 1, 2),
            0x8B02_0020u32.to_le_bytes().to_vec()
        );
    }
}
