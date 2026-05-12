//! x86-64 binary instruction encoder

use crate::encoder::traits::{InstructionEncoder, ParsedInstruction};
use crate::error::RasError;

pub struct X86_64Encoder {
    position: usize,
}

impl Default for X86_64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl X86_64Encoder {
    pub fn new() -> Self {
        Self { position: 0 }
    }

    fn encode_rex(&self, w: bool, r: u8, x: u8, b: u8) -> u8 {
        let mut rex = 0x40;
        if w {
            rex |= 0x08;
        }
        if r > 0 {
            rex |= 0x04;
        }
        if x > 0 {
            rex |= 0x02;
        }
        if b > 0 {
            rex |= 0x01;
        }
        rex
    }

    fn parse_register(&self, reg: &str) -> Result<u8, RasError> {
        let reg = reg.trim_start_matches('%');
        match reg {
            "rax" | "eax" | "ax" | "al" => Ok(0),
            "rcx" | "ecx" | "cx" | "cl" => Ok(1),
            "rdx" | "edx" | "dx" | "dl" => Ok(2),
            "rbx" | "ebx" | "bx" | "bl" => Ok(3),
            "rsp" | "esp" | "sp" | "ah" => Ok(4),
            "rbp" | "ebp" | "bp" | "ch" => Ok(5),
            "rsi" | "esi" | "si" | "dh" => Ok(6),
            "rdi" | "edi" | "di" | "bh" => Ok(7),
            "r8" => Ok(8),
            "r9" => Ok(9),
            "r10" => Ok(10),
            "r11" => Ok(11),
            "r12" => Ok(12),
            "r13" => Ok(13),
            "r14" => Ok(14),
            "r15" => Ok(15),
            _ => Err(RasError::EncodingError(format!(
                "Unknown register: {}",
                reg
            ))),
        }
    }

    fn parse_memory(&self, op: &str) -> Result<(i32, u8), RasError> {
        let op = op.trim();
        if !op.contains('(') || !op.contains(')') {
            return Err(RasError::EncodingError(format!(
                "Invalid memory operand: {}",
                op
            )));
        }
        let paren_start = op.find('(').unwrap();
        let paren_end = op.find(')').unwrap();
        let disp_str = op[..paren_start].trim();
        let base_str = op[paren_start + 1..paren_end].trim();

        let disp: i32 = if disp_str.is_empty() {
            0
        } else {
            disp_str.parse().map_err(|_| {
                RasError::EncodingError(format!("Invalid displacement: {}", disp_str))
            })?
        };

        let base = self.parse_register(base_str)?;
        Ok((disp, base))
    }

    fn is_memory(&self, op: &str) -> bool {
        op.contains('(') && op.contains(')')
    }

    fn encode_mov_mem_reg(&self, mem: &str, reg: u8, store: bool) -> Result<Vec<u8>, RasError> {
        let (disp, base) = self.parse_memory(mem)?;
        let rex = self.encode_rex(true, reg >> 3, 0, base >> 3);
        let mod_reg = (reg & 7) << 3;

        let (mod_bits, r_m, sib_opt, disp_bytes): (u8, u8, Option<u8>, Vec<u8>) = if base == 4 {
            (0, 4, Some(0x24), vec![])
        } else if base >= 8 {
            let sib = (2 << 6) | (4 << 3) | (base & 7);
            if disp == 0 {
                (0, 4, Some(sib), vec![])
            } else if (-128..=127).contains(&disp) {
                (1, 4, Some(sib), vec![disp as u8])
            } else {
                (2, 4, Some(sib), disp.to_le_bytes().to_vec())
            }
        } else if disp == 0 && base != 5 {
            (0, base & 7, None, vec![])
        } else if base == 5 && disp == 0 {
            (1, 5, None, vec![0])
        } else if (-128..=127).contains(&disp) {
            (1, base & 7, None, vec![disp as u8])
        } else {
            (2, base & 7, None, disp.to_le_bytes().to_vec())
        };

        let modrm = (mod_bits << 6) | mod_reg | r_m;
        let mut code = vec![rex];
        code.push(if store { 0x89 } else { 0x8B });
        code.push(modrm);
        if let Some(sib) = sib_opt {
            code.push(sib);
        }
        code.extend(disp_bytes);
        Ok(code)
    }
}

impl InstructionEncoder for X86_64Encoder {
    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let opcode = inst.opcode.to_lowercase();
        let mut code = Vec::new();

        match opcode.as_str() {
            "movq" | "mov" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "mov requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];

                if a.starts_with('$') && !b.starts_with('$') {
                    let imm: i64 = a
                        .trim_start_matches('$')
                        .parse()
                        .map_err(|_| RasError::EncodingError("Invalid immediate".to_string()))?;
                    let dst_reg = self.parse_register(b)?;
                    code.push(self.encode_rex(true, 0, 0, dst_reg >> 3));
                    code.push(0xB8 | (dst_reg & 7));
                    code.extend_from_slice(&imm.to_le_bytes());
                } else if !a.starts_with('$') && b.starts_with('$') {
                    let imm: i64 = b
                        .trim_start_matches('$')
                        .parse()
                        .map_err(|_| RasError::EncodingError("Invalid immediate".to_string()))?;
                    let dst_reg = self.parse_register(a)?;
                    code.push(self.encode_rex(true, 0, 0, dst_reg >> 3));
                    code.push(0xB8 | (dst_reg & 7));
                    code.extend_from_slice(&imm.to_le_bytes());
                } else if self.is_memory(a) && !self.is_memory(b) {
                    let reg = self.parse_register(b)?;
                    code.extend(self.encode_mov_mem_reg(a, reg, false)?);
                } else if self.is_memory(b) && !self.is_memory(a) {
                    let reg = self.parse_register(a)?;
                    code.extend(self.encode_mov_mem_reg(b, reg, true)?);
                } else if !a.starts_with('$')
                    && !b.starts_with('$')
                    && !self.is_memory(a)
                    && !self.is_memory(b)
                {
                    let dst_reg = self.parse_register(a)?;
                    let src_reg = self.parse_register(b)?;
                    code.push(self.encode_rex(true, src_reg >> 3, 0, dst_reg >> 3));
                    code.push(0x89);
                    code.push(0xC0 | (src_reg << 3) | (dst_reg & 7));
                } else {
                    return Err(RasError::EncodingError(
                        "mov: one operand must be a register".to_string(),
                    ));
                }
            }
            "ret" => code.push(0xC3),
            "addq" | "add" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "add requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];

                let (dst_reg, imm_opt) =
                    if a.starts_with('$') {
                        let imm: i32 = a.trim_start_matches('$').parse().map_err(|_| {
                            RasError::EncodingError("Invalid immediate".to_string())
                        })?;
                        (self.parse_register(b)?, Some(imm))
                    } else if b.starts_with('$') {
                        let imm: i32 = b.trim_start_matches('$').parse().map_err(|_| {
                            RasError::EncodingError("Invalid immediate".to_string())
                        })?;
                        (self.parse_register(a)?, Some(imm))
                    } else {
                        let dst_reg = self.parse_register(a)?;
                        let src_reg = self.parse_register(b)?;
                        code.push(self.encode_rex(true, src_reg >> 3, 0, dst_reg >> 3));
                        code.push(0x01);
                        code.push(0xC0 | ((src_reg & 7) << 3) | (dst_reg & 7));
                        self.position += code.len();
                        return Ok(code);
                    };

                if let Some(imm) = imm_opt {
                    if (-128..=127).contains(&imm) {
                        code.push(self.encode_rex(true, 0, 0, dst_reg >> 3));
                        code.push(0x83);
                        code.push(0xC0 | (dst_reg & 7));
                        code.push(imm as u8);
                    } else {
                        code.push(self.encode_rex(true, 0, 0, dst_reg >> 3));
                        code.push(0x81);
                        code.push(0xC0 | (dst_reg & 7));
                        code.extend_from_slice(&imm.to_le_bytes());
                    }
                }
            }
            "pushq" | "push" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "push requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 8 {
                    code.push(0x41);
                }
                code.push(0x50 | (reg & 7));
            }
            "popq" | "pop" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "pop requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 8 {
                    code.push(0x41);
                }
                code.push(0x58 | (reg & 7));
            }
            "leaq" | "lea" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "lea requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];
                if !self.is_memory(a) || self.is_memory(b) {
                    return Err(RasError::EncodingError(
                        "lea: first operand must be memory, second register".to_string(),
                    ));
                }
                let reg = self.parse_register(b)?;
                let bytes = self.encode_mov_mem_reg(a, reg, false)?;
                let mut lea_code = vec![bytes[0]];
                lea_code.push(0x8D);
                lea_code.extend_from_slice(&bytes[2..]);
                code.extend(lea_code);
            }
            "imulq" | "imul" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "imul requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];
                if self.is_memory(a) || self.is_memory(b) {
                    return Err(RasError::EncodingError(
                        "imul reg,reg only supported".to_string(),
                    ));
                }
                let src_reg = self.parse_register(a)?;
                let dst_reg = self.parse_register(b)?;
                code.push(self.encode_rex(true, src_reg >> 3, 0, dst_reg >> 3));
                code.extend_from_slice(&[0x0F, 0xAF]);
                code.push(0xC0 | (src_reg << 3) | (dst_reg & 7));
            }
            "subq" | "sub" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "sub requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];
                let (dst_reg, imm_opt) =
                    if a.starts_with('$') {
                        let imm: i32 = a.trim_start_matches('$').parse().map_err(|_| {
                            RasError::EncodingError("Invalid immediate".to_string())
                        })?;
                        (self.parse_register(b)?, Some(imm))
                    } else if b.starts_with('$') {
                        let imm: i32 = b.trim_start_matches('$').parse().map_err(|_| {
                            RasError::EncodingError("Invalid immediate".to_string())
                        })?;
                        (self.parse_register(a)?, Some(imm))
                    } else {
                        let dst_reg = self.parse_register(a)?;
                        let src_reg = self.parse_register(b)?;
                        code.push(self.encode_rex(true, src_reg >> 3, 0, dst_reg >> 3));
                        code.push(0x29);
                        code.push(0xC0 | ((src_reg & 7) << 3) | (dst_reg & 7));
                        self.position += code.len();
                        return Ok(code);
                    };
                if let Some(imm) = imm_opt {
                    if (-128..=127).contains(&imm) {
                        code.push(self.encode_rex(true, 0, 0, dst_reg >> 3));
                        code.push(0x83);
                        code.push(0xE8 | (dst_reg & 7));
                        code.push(imm as u8);
                    } else {
                        code.push(self.encode_rex(true, 0, 0, dst_reg >> 3));
                        code.push(0x81);
                        code.push(0xE8 | (dst_reg & 7));
                        code.extend_from_slice(&imm.to_le_bytes());
                    }
                }
            }
            // REX.W 0x99: sign-extend rax into rdx:rax (needed before idivq)
            "cqto" | "cqo" => {
                code.push(0x48); // REX.W
                code.push(0x99);
            }
            // REX.W 0x99 for 32-bit sign-extend eax into edx:eax
            "cdq" => {
                code.push(0x99);
            }
            // idivq %reg: REX.W F7 /7
            "idivq" | "idiv" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "idivq requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                code.push(self.encode_rex(true, 0, 0, reg >> 3));
                code.push(0xF7);
                code.push(0xF8 | (reg & 7)); // ModRM: /7 = 111, mod=11
            }
            // divq %reg: REX.W F7 /6 (unsigned divide)
            "divq" | "div" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "divq requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                code.push(self.encode_rex(true, 0, 0, reg >> 3));
                code.push(0xF7);
                code.push(0xF0 | (reg & 7)); // ModRM: /6 = 110, mod=11
            }
            // xorq reg, reg
            "xorq" | "xor" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "xorq requires 2 operands".to_string(),
                    ));
                }
                let dst = self.parse_register(&inst.operands[0])?;
                let src = self.parse_register(&inst.operands[1])?;
                code.push(self.encode_rex(true, src >> 3, 0, dst >> 3));
                code.push(0x31);
                code.push(0xC0 | ((src & 7) << 3) | (dst & 7));
            }
            // andq reg/imm, reg
            "andq" | "and" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "andq requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];
                if a.starts_with('$') {
                    let imm: i32 = a.trim_start_matches('$').parse().map_err(|_| {
                        RasError::EncodingError("Invalid immediate for andq".to_string())
                    })?;
                    let dst = self.parse_register(b)?;
                    if (-128..=127).contains(&imm) {
                        code.push(self.encode_rex(true, 0, 0, dst >> 3));
                        code.push(0x83);
                        code.push(0xE0 | (dst & 7)); // /4
                        code.push(imm as u8);
                    } else {
                        code.push(self.encode_rex(true, 0, 0, dst >> 3));
                        code.push(0x81);
                        code.push(0xE0 | (dst & 7));
                        code.extend_from_slice(&imm.to_le_bytes());
                    }
                } else {
                    let dst = self.parse_register(a)?;
                    let src = self.parse_register(b)?;
                    code.push(self.encode_rex(true, src >> 3, 0, dst >> 3));
                    code.push(0x21);
                    code.push(0xC0 | ((src & 7) << 3) | (dst & 7));
                }
            }
            // orq reg/imm, reg
            "orq" | "or" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "orq requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];
                if a.starts_with('$') {
                    let imm: i32 = a.trim_start_matches('$').parse().map_err(|_| {
                        RasError::EncodingError("Invalid immediate for orq".to_string())
                    })?;
                    let dst = self.parse_register(b)?;
                    if (-128..=127).contains(&imm) {
                        code.push(self.encode_rex(true, 0, 0, dst >> 3));
                        code.push(0x83);
                        code.push(0xC8 | (dst & 7)); // /1
                        code.push(imm as u8);
                    } else {
                        code.push(self.encode_rex(true, 0, 0, dst >> 3));
                        code.push(0x81);
                        code.push(0xC8 | (dst & 7));
                        code.extend_from_slice(&imm.to_le_bytes());
                    }
                } else {
                    let dst = self.parse_register(a)?;
                    let src = self.parse_register(b)?;
                    code.push(self.encode_rex(true, src >> 3, 0, dst >> 3));
                    code.push(0x09);
                    code.push(0xC0 | ((src & 7) << 3) | (dst & 7));
                }
            }
            // sarq / salq / shrq / shlq (shift instructions)
            "sarq" | "sar" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "sarq requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let dst = self.parse_register(&inst.operands[1])?;
                if a.starts_with('$') {
                    let imm: u8 = a.trim_start_matches('$').parse().map_err(|_| {
                        RasError::EncodingError("Invalid immediate for sarq".to_string())
                    })?;
                    code.push(self.encode_rex(true, 0, 0, dst >> 3));
                    code.push(0xC1);
                    code.push(0xF8 | (dst & 7)); // /7
                    code.push(imm);
                } else if a == "%cl" || a == "cl" {
                    code.push(self.encode_rex(true, 0, 0, dst >> 3));
                    code.push(0xD3);
                    code.push(0xF8 | (dst & 7));
                } else {
                    return Err(RasError::EncodingError(
                        "sarq: unsupported shift source".to_string(),
                    ));
                }
            }
            "shlq" | "salq" | "shl" | "sal" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "shlq requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let dst = self.parse_register(&inst.operands[1])?;
                if a.starts_with('$') {
                    let imm: u8 = a.trim_start_matches('$').parse().map_err(|_| {
                        RasError::EncodingError("Invalid immediate for shlq".to_string())
                    })?;
                    code.push(self.encode_rex(true, 0, 0, dst >> 3));
                    code.push(0xC1);
                    code.push(0xE0 | (dst & 7)); // /4
                    code.push(imm);
                } else if a == "%cl" || a == "cl" {
                    code.push(self.encode_rex(true, 0, 0, dst >> 3));
                    code.push(0xD3);
                    code.push(0xE0 | (dst & 7));
                } else {
                    return Err(RasError::EncodingError(
                        "shlq: unsupported shift source".to_string(),
                    ));
                }
            }
            "shrq" | "shr" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "shrq requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let dst = self.parse_register(&inst.operands[1])?;
                if a.starts_with('$') {
                    let imm: u8 = a.trim_start_matches('$').parse().map_err(|_| {
                        RasError::EncodingError("Invalid immediate for shrq".to_string())
                    })?;
                    code.push(self.encode_rex(true, 0, 0, dst >> 3));
                    code.push(0xC1);
                    code.push(0xE8 | (dst & 7)); // /5
                    code.push(imm);
                } else if a == "%cl" || a == "cl" {
                    code.push(self.encode_rex(true, 0, 0, dst >> 3));
                    code.push(0xD3);
                    code.push(0xE8 | (dst & 7));
                } else {
                    return Err(RasError::EncodingError(
                        "shrq: unsupported shift source".to_string(),
                    ));
                }
            }
            // cmpq: compare two values (sets flags)
            "cmpq" | "cmp" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "cmpq requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];
                let dst = self.parse_register(b)?;
                if a.starts_with('$') {
                    let imm: i32 = a.trim_start_matches('$').parse().map_err(|_| {
                        RasError::EncodingError("Invalid immediate for cmpq".to_string())
                    })?;
                    if (-128..=127).contains(&imm) {
                        code.push(self.encode_rex(true, 0, 0, dst >> 3));
                        code.push(0x83);
                        code.push(0xF8 | (dst & 7)); // /7
                        code.push(imm as u8);
                    } else {
                        code.push(self.encode_rex(true, 0, 0, dst >> 3));
                        code.push(0x81);
                        code.push(0xF8 | (dst & 7));
                        code.extend_from_slice(&imm.to_le_bytes());
                    }
                } else {
                    let src = self.parse_register(a)?;
                    code.push(self.encode_rex(true, src >> 3, 0, dst >> 3));
                    code.push(0x3B);
                    code.push(0xC0 | ((dst & 7) << 3) | (src & 7));
                }
            }
            // testq: bitwise AND without storing result (sets flags)
            "testq" | "test" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "testq requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];
                if a.starts_with('$') {
                    let imm: i32 = a.trim_start_matches('$').parse().map_err(|_| {
                        RasError::EncodingError("Invalid immediate for testq".to_string())
                    })?;
                    let dst = self.parse_register(b)?;
                    code.push(self.encode_rex(true, 0, 0, dst >> 3));
                    code.push(0xF7);
                    code.push(0xC0 | (dst & 7)); // /0
                    code.extend_from_slice(&imm.to_le_bytes());
                } else {
                    let dst = self.parse_register(a)?;
                    let src = self.parse_register(b)?;
                    code.push(self.encode_rex(true, src >> 3, 0, dst >> 3));
                    code.push(0x85);
                    code.push(0xC0 | ((src & 7) << 3) | (dst & 7));
                }
            }
            // SETcc instructions (byte result from flags)
            "sete" | "setz" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "sete requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                } // REX prefix for spl/bpl/sil/dil and r8b-r15b
                code.push(0x0F);
                code.push(0x94);
                code.push(0xC0 | (reg & 7));
            }
            "setne" | "setnz" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "setne requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                }
                code.push(0x0F);
                code.push(0x95);
                code.push(0xC0 | (reg & 7));
            }
            "setl" | "setnge" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "setl requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                }
                code.push(0x0F);
                code.push(0x9C);
                code.push(0xC0 | (reg & 7));
            }
            "setle" | "setng" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "setle requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                }
                code.push(0x0F);
                code.push(0x9E);
                code.push(0xC0 | (reg & 7));
            }
            "setg" | "setnle" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "setg requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                }
                code.push(0x0F);
                code.push(0x9F);
                code.push(0xC0 | (reg & 7));
            }
            "setge" | "setnl" => {
                if inst.operands.len() != 1 {
                    return Err(RasError::EncodingError(
                        "setge requires 1 operand".to_string(),
                    ));
                }
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                }
                code.push(0x0F);
                code.push(0x9D);
                code.push(0xC0 | (reg & 7));
            }
            // movzbq: zero-extend byte to 64-bit
            "movzbq" | "movzbl" | "movzx" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "movzbq requires 2 operands".to_string(),
                    ));
                }
                let src_str = &inst.operands[0];
                let dst_str = &inst.operands[1];
                let src = self.parse_register(src_str)?;
                let dst = self.parse_register(dst_str)?;
                // REX.W 0F B6 /r
                code.push(self.encode_rex(true, dst >> 3, 0, src >> 3));
                code.push(0x0F);
                code.push(0xB6);
                code.push(0xC0 | ((dst & 7) << 3) | (src & 7));
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
