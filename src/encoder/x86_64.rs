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

        // r/m 100 means a SIB follows; r/m 101 at mod 00 means disp32-only.
        let low = base & 7;
        let needs_sib = low == 4;
        // scale 00, index 100 (none)
        let sib_opt = needs_sib.then_some((4 << 3) | low);
        let r_m = if needs_sib { 4 } else { low };

        let (mod_bits, disp_bytes): (u8, Vec<u8>) = if disp == 0 && low != 5 {
            (0, vec![])
        } else if (-128..=127).contains(&disp) {
            (1, vec![disp as u8])
        } else {
            (2, disp.to_le_bytes().to_vec())
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
                    let a_xmm = parse_xmm(a.trim().trim_start_matches('%'));
                    let b_xmm = parse_xmm(b.trim().trim_start_matches('%'));
                    if let (Ok(gpr), Some(xmm)) = (self.parse_register(a), b_xmm) {
                        // movq %gpr, %xmm  →  66 REX.W 0F 6E /r
                        code.push(0x66);
                        code.push(0x48 | ((xmm >> 3) << 2) | (gpr >> 3));
                        code.extend_from_slice(&[0x0F, 0x6E]);
                        code.push(0xC0 | ((xmm & 7) << 3) | (gpr & 7));
                    } else if let (Some(xmm), Ok(gpr)) = (a_xmm, self.parse_register(b)) {
                        // movq %xmm, %gpr  →  66 REX.W 0F 7E /r
                        code.push(0x66);
                        code.push(0x48 | ((xmm >> 3) << 2) | (gpr >> 3));
                        code.extend_from_slice(&[0x0F, 0x7E]);
                        code.push(0xC0 | ((xmm & 7) << 3) | (gpr & 7));
                    } else {
                        // AT&T order: `mov src, dst`. 0x89 is MOV r/m64, r64 where
                        // the reg field is the source and r/m is the destination.
                        let src_reg = self.parse_register(a)?;
                        let dst_reg = self.parse_register(b)?;
                        code.push(self.encode_rex(true, src_reg >> 3, 0, dst_reg >> 3));
                        code.push(0x89);
                        code.push(0xC0 | ((src_reg & 7) << 3) | (dst_reg & 7));
                    }
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
                        // AT&T order: `add src, dst`. 0x01 is ADD r/m64, r64
                        // (reg = source added into r/m = destination).
                        let src_reg = self.parse_register(a)?;
                        let dst_reg = self.parse_register(b)?;
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
                // AT&T order: `imul src, dst`. 0F AF is IMUL r64, r/m64 where
                // the reg field is the destination and r/m is the source.
                let src_reg = self.parse_register(a)?;
                let dst_reg = self.parse_register(b)?;
                code.push(self.encode_rex(true, dst_reg >> 3, 0, src_reg >> 3));
                code.extend_from_slice(&[0x0F, 0xAF]);
                code.push(0xC0 | ((dst_reg & 7) << 3) | (src_reg & 7));
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
                        // AT&T order: `sub src, dst`. 0x29 is SUB r/m64, r64
                        // (r/m = destination, reg = subtracted source).
                        let src_reg = self.parse_register(a)?;
                        let dst_reg = self.parse_register(b)?;
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
                // AT&T order: `xor src, dst`. 0x31 is XOR r/m64, r64
                // (r/m = destination, reg = source).
                let src = self.parse_register(&inst.operands[0])?;
                let dst = self.parse_register(&inst.operands[1])?;
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
                    // AT&T order: `and src, dst`. 0x21 is AND r/m64, r64
                    // (r/m = destination, reg = source).
                    let src = self.parse_register(a)?;
                    let dst = self.parse_register(b)?;
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
                    // AT&T order: `or src, dst`. 0x09 is OR r/m64, r64
                    // (r/m = destination, reg = source).
                    let src = self.parse_register(a)?;
                    let dst = self.parse_register(b)?;
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
                    // AT&T order: `cmp src, dst` computes dst - src. 0x3B is
                    // CMP r64, r/m64 (reg = dst, r/m = src).
                    let src = self.parse_register(a)?;
                    code.push(self.encode_rex(true, dst >> 3, 0, src >> 3));
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
            // xorl: 32-bit XOR (no REX.W; zeroing a reg also zero-extends to 64-bit)
            "xorl" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError("xorl requires 2 operands".into()));
                }
                let src = self.parse_register(&inst.operands[0])?;
                let dst = self.parse_register(&inst.operands[1])?;
                // Only REX prefix if accessing r8-r15
                if src >= 8 || dst >= 8 {
                    code.push(0x40 | ((src >> 3) << 2) | (dst >> 3));
                }
                code.push(0x31);
                code.push(0xC0 | ((src & 7) << 3) | (dst & 7));
            }
            // movb: byte store to memory or byte reg move
            "movb" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError("movb requires 2 operands".into()));
                }
                let src_str = &inst.operands[0];
                let dst_str = &inst.operands[1];
                if self.is_memory(dst_str) {
                    // movb %reg8, (mem)  — 88 /r
                    let (disp, base) = self.parse_memory(dst_str)?;
                    let src = self.parse_register(src_str)?;
                    // REX prefix if needed for high-numbered regs
                    if src >= 4 || base >= 8 {
                        code.push(0x40 | ((src >> 3) << 2) | (base >> 3));
                    }
                    code.push(0x88);
                    if base == 4 {
                        code.push(((src & 7) << 3) | 4);
                        code.push(0x24); // SIB: index=none, base=rsp
                    } else if disp == 0 && base != 5 {
                        code.push(((src & 7) << 3) | (base & 7));
                    } else if (-128..=127).contains(&disp) {
                        code.push(0x40 | ((src & 7) << 3) | (base & 7));
                        code.push(disp as u8);
                    } else {
                        code.push(0x80 | ((src & 7) << 3) | (base & 7));
                        code.extend_from_slice(&disp.to_le_bytes());
                    }
                } else if src_str.starts_with('$') {
                    // movb $imm8, (mem)
                    let imm: i8 = src_str.trim_start_matches('$').parse().map_err(|_| {
                        RasError::EncodingError("Invalid immediate for movb".into())
                    })?;
                    let (disp, base) = self.parse_memory(dst_str)?;
                    if base >= 8 {
                        code.push(0x41);
                    }
                    code.push(0xC6);
                    if disp == 0 && base != 5 {
                        code.push(base & 7);
                    } else if (-128..=127).contains(&disp) {
                        code.push(0x40 | (base & 7));
                        code.push(disp as u8);
                    } else {
                        code.push(0x80 | (base & 7));
                        code.extend_from_slice(&disp.to_le_bytes());
                    }
                    code.push(imm as u8);
                } else {
                    return Err(RasError::EncodingError(
                        "movb: unsupported operand form".into(),
                    ));
                }
            }
            // syscall: 0F 05
            "syscall" => {
                code.extend_from_slice(&[0x0F, 0x05]);
            }
            // SETcc for unsigned comparisons
            "seta" | "setnbe" => {
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                }
                code.extend_from_slice(&[0x0F, 0x97, 0xC0 | (reg & 7)]);
            }
            "setae" | "setnb" | "setnc" => {
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                }
                code.extend_from_slice(&[0x0F, 0x93, 0xC0 | (reg & 7)]);
            }
            "setb" | "setnae" | "setc" => {
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                }
                code.extend_from_slice(&[0x0F, 0x92, 0xC0 | (reg & 7)]);
            }
            "setbe" | "setna" => {
                let reg = self.parse_register(&inst.operands[0])?;
                if reg >= 4 {
                    code.push(0x40 | (reg >> 3));
                }
                code.extend_from_slice(&[0x0F, 0x96, 0xC0 | (reg & 7)]);
            }
            // cmovzq / cmovz (conditional move if ZF=1)
            "cmovzq" | "cmovz" | "cmoveq" | "cmove" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError("cmovzq requires 2 operands".into()));
                }
                let src = self.parse_register(&inst.operands[0])?;
                let dst = self.parse_register(&inst.operands[1])?;
                code.push(self.encode_rex(true, dst >> 3, 0, src >> 3));
                code.extend_from_slice(&[0x0F, 0x44]);
                code.push(0xC0 | ((dst & 7) << 3) | (src & 7));
            }
            // movd: move 32/64-bit GPR to/from XMM  (66 [REX] 0F 6E /r)
            "movd" | "movq_xmm" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError("movd requires 2 operands".into()));
                }
                let src_str = inst.operands[0].trim().trim_start_matches('%');
                let dst_str = inst.operands[1].trim().trim_start_matches('%');
                // movd %gpr, %xmm  →  66 [REX.W] 0F 6E /r
                if let (Ok(gpr), Some(xmm)) = (
                    self.parse_register(inst.operands[0].trim()),
                    parse_xmm(dst_str),
                ) {
                    let need_rex_w = inst.operands[0]
                        .trim()
                        .trim_start_matches('%')
                        .starts_with('r');
                    code.push(0x66);
                    if need_rex_w || gpr >= 8 || xmm >= 8 {
                        code.push(
                            0x40 | (if need_rex_w { 8 } else { 0 })
                                | ((xmm >> 3) << 2)
                                | (gpr >> 3),
                        );
                    }
                    code.extend_from_slice(&[0x0F, 0x6E]);
                    code.push(0xC0 | ((xmm & 7) << 3) | (gpr & 7));
                } else if let (Some(xmm), Ok(gpr)) = (
                    parse_xmm(src_str),
                    self.parse_register(inst.operands[1].trim()),
                ) {
                    // movd %xmm, %gpr  →  66 [REX.W] 0F 7E /r
                    let need_rex_w = inst.operands[1]
                        .trim()
                        .trim_start_matches('%')
                        .starts_with('r');
                    code.push(0x66);
                    if need_rex_w || xmm >= 8 || gpr >= 8 {
                        code.push(
                            0x40 | (if need_rex_w { 8 } else { 0 })
                                | ((xmm >> 3) << 2)
                                | (gpr >> 3),
                        );
                    }
                    code.extend_from_slice(&[0x0F, 0x7E]);
                    code.push(0xC0 | ((xmm & 7) << 3) | (gpr & 7));
                } else {
                    return Err(RasError::EncodingError(
                        "movd: expected gpr↔xmm operands".into(),
                    ));
                }
            }
            // xorps: 0F 57 /r
            "xorps" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError("xorps requires 2 operands".into()));
                }
                let src = parse_xmm(inst.operands[0].trim().trim_start_matches('%'))
                    .ok_or_else(|| RasError::EncodingError("xorps: expected xmm src".into()))?;
                let dst = parse_xmm(inst.operands[1].trim().trim_start_matches('%'))
                    .ok_or_else(|| RasError::EncodingError("xorps: expected xmm dst".into()))?;
                if src >= 8 || dst >= 8 {
                    code.push(0x40 | ((dst >> 3) << 2) | (src >> 3));
                }
                code.extend_from_slice(&[0x0F, 0x57]);
                code.push(0xC0 | ((dst & 7) << 3) | (src & 7));
            }
            // xorpd: 66 0F 57 /r
            "xorpd" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError("xorpd requires 2 operands".into()));
                }
                let src = parse_xmm(inst.operands[0].trim().trim_start_matches('%'))
                    .ok_or_else(|| RasError::EncodingError("xorpd: expected xmm src".into()))?;
                let dst = parse_xmm(inst.operands[1].trim().trim_start_matches('%'))
                    .ok_or_else(|| RasError::EncodingError("xorpd: expected xmm dst".into()))?;
                code.push(0x66);
                if src >= 8 || dst >= 8 {
                    code.push(0x40 | ((dst >> 3) << 2) | (src >> 3));
                }
                code.extend_from_slice(&[0x0F, 0x57]);
                code.push(0xC0 | ((dst & 7) << 3) | (src & 7));
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

fn parse_xmm(s: &str) -> Option<u8> {
    let s = s.trim_start_matches('%');
    let s = s.strip_prefix("xmm")?;
    let n: u8 = s.parse().ok()?;
    if n < 16 { Some(n) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected bytes come from GNU as.
    #[test]
    fn memory_operands_match_gnu_as() {
        let enc = X86_64Encoder::new();
        let load = |mem: &str| enc.encode_mov_mem_reg(mem, 0, false).expect("encode");

        assert_eq!(load("8(%rsp)"), vec![0x48, 0x8b, 0x44, 0x24, 0x08]);
        assert_eq!(load("(%rsp)"), vec![0x48, 0x8b, 0x04, 0x24]);

        assert_eq!(load("(%r13)"), vec![0x49, 0x8b, 0x45, 0x00]);
        assert_eq!(load("8(%r13)"), vec![0x49, 0x8b, 0x45, 0x08]);

        assert_eq!(load("(%rbp)"), vec![0x48, 0x8b, 0x45, 0x00]);

        assert_eq!(load("(%r12)"), vec![0x49, 0x8b, 0x04, 0x24]);
    }
}
