//! x86-64 binary instruction encoder

use crate::{
    encoder::traits::{InstructionEncoder, ParsedInstruction, RipFixup},
    error::RasError,
};

pub struct X86_64Encoder {
    position: usize,
    rip_fixup: Option<RipFixup>,
}

impl Default for X86_64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl X86_64Encoder {
    pub fn new() -> Self {
        Self {
            position: 0,
            rip_fixup: None,
        }
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
        if let Some(number) = reg.strip_prefix('r') {
            let number = number.trim_end_matches(['b', 'w', 'd']);
            if let Ok(n) = number.parse::<u8>()
                && (8..=15).contains(&n)
            {
                return Ok(n);
            }
        }
        match reg {
            "rax" | "eax" | "ax" | "al" => Ok(0),
            "rcx" | "ecx" | "cx" | "cl" => Ok(1),
            "rdx" | "edx" | "dx" | "dl" => Ok(2),
            "rbx" | "ebx" | "bx" | "bl" => Ok(3),
            "rsp" | "esp" | "sp" | "spl" | "ah" => Ok(4),
            "rbp" | "ebp" | "bp" | "bpl" | "ch" => Ok(5),
            "rsi" | "esi" | "si" | "sil" | "dh" => Ok(6),
            "rdi" | "edi" | "di" | "dil" | "bh" => Ok(7),
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

    fn parse_mem(&self, op: &str) -> Result<Mem, RasError> {
        let invalid = || RasError::EncodingError(format!("Invalid memory operand: {}", op));
        let mut text = op.trim();
        let fs = match text.strip_prefix("%fs:") {
            Some(rest) => {
                text = rest;
                true
            }
            None => false,
        };
        let (disp_text, inner) = match text.find('(') {
            Some(open) => {
                let inner = text[open + 1..].strip_suffix(')').ok_or_else(invalid)?;
                (&text[..open], Some(inner))
            }
            None => (text, None),
        };
        let disp_text = disp_text.trim();
        let mut mem = Mem {
            fs,
            disp: 0,
            label: None,
            base: None,
            index: None,
            rip: false,
        };
        if !disp_text.is_empty() {
            match parse_imm(disp_text) {
                Ok(value) => mem.disp = value,
                Err(_) if is_symbol(disp_text) => mem.label = Some(disp_text.to_string()),
                Err(_) => return Err(invalid()),
            }
        }
        if let Some(inner) = inner {
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            let (base, index) = match parts.as_slice() {
                [base] => (*base, None),
                [base, index] => (*base, Some((*index, "1"))),
                [base, index, scale] => (*base, Some((*index, *scale))),
                _ => return Err(invalid()),
            };
            if base.trim_start_matches('%') == "rip" {
                if index.is_some() || fs {
                    return Err(invalid());
                }
                mem.rip = true;
            } else if !base.is_empty() {
                let reg = gpr(base)
                    .filter(|g| g.width == 64 && !g.high)
                    .ok_or_else(invalid)?;
                mem.base = Some(reg.num);
            }
            if let Some((index, scale)) = index {
                let reg = gpr(index)
                    .filter(|g| g.width == 64 && !g.high && g.num != 4)
                    .ok_or_else(invalid)?;
                let scale_bits = match scale {
                    "1" => 0,
                    "2" => 1,
                    "4" => 2,
                    "8" => 3,
                    _ => return Err(invalid()),
                };
                mem.index = Some((reg.num, scale_bits));
            }
        }
        if mem.label.is_some() && !mem.rip {
            return Err(RasError::EncodingError(
                "Absolute symbol operands are not supported".into(),
            ));
        }
        Ok(mem)
    }

    fn emit(&mut self, encoded: Encoded) -> Vec<u8> {
        self.rip_fixup = encoded.rip;
        encoded.bytes
    }

    fn is_memory(&self, op: &str) -> bool {
        let op = op.trim();
        (op.contains('(') && op.contains(')')) || op.starts_with("%fs:")
    }

    fn parse_rm(&self, op: &str, xmm: bool) -> Result<(Rm, Option<Gpr>), RasError> {
        let op = op.trim();
        if self.is_memory(op) {
            return Ok((Rm::Mem(self.parse_mem(op)?), None));
        }
        if xmm && let Some(number) = parse_xmm(op) {
            return Ok((Rm::Reg(number), None));
        }
        if xmm {
            return Err(RasError::EncodingError(format!(
                "Expected xmm or memory operand: {}",
                op
            )));
        }
        let gpr =
            gpr(op).ok_or_else(|| RasError::EncodingError(format!("Unknown register: {}", op)))?;
        Ok((Rm::Reg(gpr.num), Some(gpr)))
    }

    fn encode_integer_move(
        &self,
        width: Option<u8>,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError("mov requires two operands".into()));
        }
        let (src, dst) = if operands[1].starts_with('$') {
            (&operands[1], &operands[0])
        } else {
            (&operands[0], &operands[1])
        };
        if let Some(imm_text) = src.trim().strip_prefix('$') {
            let imm = parse_imm(imm_text)?;
            let (rm, gpr) = self.parse_rm(dst, false)?;
            let width = operand_width(width, &[gpr])?;
            let legacy: &[u8] = if width == 16 { &[0x66] } else { &[] };
            if matches!(rm, Rm::Mem(_)) || (width == 64 && i32::try_from(imm).is_ok()) {
                if width == 64 && i32::try_from(imm).is_err() {
                    return Err(RasError::EncodingError(
                        "Memory move immediate must fit signed 32 bits".into(),
                    ));
                }
                let (byte_rex, high) = byte_flags(&[gpr]);
                let opcode = [if width == 8 { 0xc6 } else { 0xc7 }];
                return encode_op(Op {
                    legacy,
                    wide: width == 64,
                    byte_rex,
                    high,
                    opcode: &opcode,
                    rm,
                    imm: &imm_bytes(imm, width)?,
                    ..Op::default()
                });
            }
            let gpr = gpr.ok_or_else(|| RasError::EncodingError("Expected gpr".into()))?;
            let (byte_rex, high) = byte_flags(&[Some(gpr)]);
            let rex = 0x40 | (u8::from(width == 64) << 3) | (gpr.num >> 3);
            let needs_rex = rex != 0x40 || byte_rex;
            if high && needs_rex {
                return Err(RasError::EncodingError(
                    "High byte registers cannot use REX".into(),
                ));
            }
            let mut bytes = legacy.to_vec();
            if needs_rex {
                bytes.push(rex);
            }
            bytes.push((if width == 8 { 0xb0 } else { 0xb8 }) | (gpr.num & 7));
            if width == 64 {
                bytes.extend_from_slice(&imm.to_le_bytes());
            } else {
                bytes.extend(imm_bytes(imm, width)?);
            }
            return Ok(Encoded { bytes, rip: None });
        }
        let (src_rm, src_gpr) = self.parse_rm(src, false)?;
        let (dst_rm, dst_gpr) = self.parse_rm(dst, false)?;
        let width = operand_width(width, &[src_gpr, dst_gpr])?;
        let legacy: &[u8] = if width == 16 { &[0x66] } else { &[] };
        let (byte_rex, high) = byte_flags(&[src_gpr, dst_gpr]);
        let (opcode, reg, rm) = match (src_rm, dst_rm) {
            (Rm::Mem(_), Rm::Mem(_)) => {
                return Err(RasError::EncodingError(
                    "mov: memory to memory is not encodable".into(),
                ));
            }
            (Rm::Mem(mem), Rm::Reg(reg)) => {
                ([if width == 8 { 0x8a } else { 0x8b }], reg, Rm::Mem(mem))
            }
            (Rm::Reg(reg), rm) => ([if width == 8 { 0x88 } else { 0x89 }], reg, rm),
        };
        encode_op(Op {
            legacy,
            wide: width == 64,
            byte_rex,
            high,
            opcode: &opcode,
            reg,
            rm,
            ..Op::default()
        })
    }

    fn encode_movabs(&self, operands: &[String]) -> Result<Encoded, RasError> {
        let [src, dst] = operands else {
            return Err(RasError::EncodingError(
                "movabsq requires 2 operands".into(),
            ));
        };
        let imm = src
            .trim()
            .strip_prefix('$')
            .ok_or_else(|| RasError::EncodingError("movabsq: source must be immediate".into()))?;
        let imm = parse_imm(imm)?;
        let dst = gpr(dst)
            .filter(|gpr| gpr.width == 64 && !gpr.high)
            .ok_or_else(|| RasError::EncodingError("movabsq: destination must be 64-bit".into()))?;
        let mut bytes = vec![0x48 | (dst.num >> 3), 0xb8 | (dst.num & 7)];
        bytes.extend_from_slice(&imm.to_le_bytes());
        Ok(Encoded { bytes, rip: None })
    }

    fn encode_alu(
        &self,
        stem: &str,
        width: Option<u8>,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError(format!(
                "{} requires 2 operands",
                stem
            )));
        }
        let (src, dst) = (operands[0].trim(), operands[1].trim());
        let is_test = stem == "test";
        let digit: u8 = match stem {
            "or" => 1,
            "and" => 4,
            "sub" => 5,
            "xor" => 6,
            "cmp" => 7,
            _ => 0,
        };
        let base = digit << 3;
        if let Some(imm_text) = src.strip_prefix('$') {
            let imm = parse_imm(imm_text)?;
            let (dst_rm, dst_gpr) = self.parse_rm(dst, false)?;
            let width = operand_width(width, &[dst_gpr])?;
            check_alu_width(width)?;
            let imm = if width == 32 {
                i64::from(imm as u32 as i32)
            } else {
                imm
            };
            let accumulator = matches!(dst_rm, Rm::Reg(0)) && dst_gpr.is_some_and(|gpr| !gpr.high);
            let fits8 = (-128..=127).contains(&imm);
            if accumulator && (is_test || width == 8 || !fits8) {
                let opcode = match (is_test, width == 8) {
                    (true, true) => 0xa8,
                    (true, false) => 0xa9,
                    (false, true) => base + 4,
                    (false, false) => base + 5,
                };
                let mut bytes = Vec::new();
                if width == 64 {
                    bytes.push(0x48);
                }
                bytes.push(opcode);
                bytes.extend(imm_bytes(imm, width)?);
                return Ok(Encoded { bytes, rip: None });
            }
            let (opcode, imm_field) = if width == 8 {
                (if is_test { 0xf6 } else { 0x80 }, imm_bytes(imm, 8)?)
            } else if fits8 && !is_test {
                (0x83, imm_bytes(imm, 8)?)
            } else {
                (if is_test { 0xf7 } else { 0x81 }, imm_bytes(imm, width)?)
            };
            let (byte_rex, high) = byte_flags(&[dst_gpr]);
            return encode_op(Op {
                wide: width == 64,
                byte_rex,
                high,
                opcode: &[opcode],
                reg: digit,
                rm: dst_rm,
                imm: &imm_field,
                ..Op::default()
            });
        }
        let (src_rm, src_gpr) = self.parse_rm(src, false)?;
        let (dst_rm, dst_gpr) = self.parse_rm(dst, false)?;
        let width = operand_width(width, &[src_gpr, dst_gpr])?;
        check_alu_width(width)?;
        let (byte_rex, high) = byte_flags(&[src_gpr, dst_gpr]);
        let size = u8::from(width != 8);
        let (opcode, reg, rm) = match (src_rm, dst_rm) {
            (Rm::Reg(reg), rm) => (if is_test { 0x84 + size } else { base + size }, reg, rm),
            (Rm::Mem(mem), Rm::Reg(reg)) if is_test => (0x84 + size, reg, Rm::Mem(mem)),
            (Rm::Mem(mem), Rm::Reg(reg)) => (base + 2 + size, reg, Rm::Mem(mem)),
            (Rm::Mem(_), Rm::Mem(_)) => {
                return Err(RasError::EncodingError(
                    "memory to memory is not encodable".into(),
                ));
            }
        };
        encode_op(Op {
            wide: width == 64,
            byte_rex,
            high,
            opcode: &[opcode],
            reg,
            rm,
            ..Op::default()
        })
    }

    fn encode_shift(
        &self,
        digit: u8,
        width: Option<u8>,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        let (count, dst) = match operands {
            [dst] => (None, dst),
            [count, dst] => (Some(count.trim()), dst),
            _ => {
                return Err(RasError::EncodingError(
                    "shift requires 1 or 2 operands".into(),
                ));
            }
        };
        let (rm, gpr) = self.parse_rm(dst, false)?;
        let width = operand_width(width, &[gpr])?;
        check_alu_width(width)?;
        let byte = width == 8;
        let (byte_rex, high) = byte_flags(&[gpr]);
        let mut imm = Vec::new();
        let opcode = match count {
            None => [if byte { 0xd0 } else { 0xd1 }],
            Some("%cl" | "cl") => [if byte { 0xd2 } else { 0xd3 }],
            Some(text) => {
                let value = parse_imm(text.strip_prefix('$').ok_or_else(|| {
                    RasError::EncodingError("shift count must be an immediate or %cl".into())
                })?)?;
                if value == 1 {
                    [if byte { 0xd0 } else { 0xd1 }]
                } else {
                    imm.push(u8::try_from(value).map_err(|_| {
                        RasError::EncodingError(format!("Shift count out of range: {}", value))
                    })?);
                    [if byte { 0xc0 } else { 0xc1 }]
                }
            }
        };
        encode_op(Op {
            wide: width == 64,
            byte_rex,
            high,
            opcode: &opcode,
            reg: digit,
            rm,
            imm: &imm,
            ..Op::default()
        })
    }

    fn encode_unary(
        &self,
        digit: u8,
        width: Option<u8>,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        let [operand] = operands else {
            return Err(RasError::EncodingError(
                "unary op requires 1 operand".into(),
            ));
        };
        let (rm, gpr) = self.parse_rm(operand, false)?;
        let width = operand_width(width, &[gpr])?;
        check_alu_width(width)?;
        let (byte_rex, high) = byte_flags(&[gpr]);
        encode_op(Op {
            wide: width == 64,
            byte_rex,
            high,
            opcode: &[if width == 8 { 0xf6 } else { 0xf7 }],
            reg: digit,
            rm,
            ..Op::default()
        })
    }

    fn encode_sse_rm(
        &self,
        legacy: &[u8],
        opcode: &[u8],
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError("SSE op requires 2 operands".into()));
        }
        let (src, _) = self.parse_rm(&operands[0], true)?;
        let (dst, _) = self.parse_rm(&operands[1], true)?;
        let Rm::Reg(reg) = dst else {
            return Err(RasError::EncodingError(
                "SSE op: destination must be an xmm register".into(),
            ));
        };
        encode_op(Op {
            legacy,
            opcode,
            reg,
            rm: src,
            ..Op::default()
        })
    }

    fn encode_sse_move(
        &self,
        legacy: &[u8],
        load: &[u8],
        store: &[u8],
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError(
                "SSE move requires 2 operands".into(),
            ));
        }
        let (src, _) = self.parse_rm(&operands[0], true)?;
        let (dst, _) = self.parse_rm(&operands[1], true)?;
        let (opcode, reg, rm) = match (src, dst) {
            (Rm::Mem(mem), Rm::Reg(dst_reg)) => (load, dst_reg, Rm::Mem(mem)),
            (Rm::Reg(src_reg), Rm::Reg(dst_reg)) => (load, dst_reg, Rm::Reg(src_reg)),
            (Rm::Reg(src_reg), Rm::Mem(mem)) => (store, src_reg, Rm::Mem(mem)),
            (Rm::Mem(_), Rm::Mem(_)) => {
                return Err(RasError::EncodingError(
                    "SSE move: memory to memory is not encodable".into(),
                ));
            }
        };
        encode_op(Op {
            legacy,
            opcode,
            reg,
            rm,
            ..Op::default()
        })
    }

    fn encode_sse_shift(&self, digit: u8, operands: &[String]) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError("shift requires 2 operands".into()));
        }
        let imm = operands[0]
            .trim()
            .strip_prefix('$')
            .map(parse_imm)
            .ok_or_else(|| RasError::EncodingError("shift count must be an immediate".into()))??;
        let imm = u8::try_from(imm)
            .map_err(|_| RasError::EncodingError(format!("Shift count out of range: {}", imm)))?;
        let (dst, _) = self.parse_rm(&operands[1], true)?;
        if !matches!(dst, Rm::Reg(_)) {
            return Err(RasError::EncodingError(
                "shift: destination must be an xmm register".into(),
            ));
        }
        encode_op(Op {
            legacy: &[0x66],
            opcode: &[0x0f, 0x73],
            reg: digit,
            rm: dst,
            imm: &[imm],
            ..Op::default()
        })
    }

    fn encode_cvttsd2si(&self, wide: bool, operands: &[String]) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError(
                "cvttsd2si requires 2 operands".into(),
            ));
        }
        let (src, _) = self.parse_rm(&operands[0], true)?;
        let (dst, dst_gpr) = self.parse_rm(&operands[1], false)?;
        let Rm::Reg(reg) = dst else {
            return Err(RasError::EncodingError(
                "cvttsd2si: destination must be a register".into(),
            ));
        };
        let expected = if wide { 64 } else { 32 };
        if dst_gpr.is_none_or(|gpr| gpr.width != expected) {
            return Err(RasError::EncodingError(
                "cvttsd2si: destination width does not match suffix".into(),
            ));
        }
        encode_op(Op {
            legacy: &[0xf2],
            wide,
            opcode: &[0x0f, 0x2c],
            reg,
            rm: src,
            ..Op::default()
        })
    }

    fn encode_movsxd(&self, operands: &[String]) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError("movslq requires 2 operands".into()));
        }
        let (src, src_gpr) = self.parse_rm(&operands[0], false)?;
        if src_gpr.is_some_and(|gpr| gpr.width != 32) {
            return Err(RasError::EncodingError(
                "movslq: source must be 32-bit".into(),
            ));
        }
        let (dst, dst_gpr) = self.parse_rm(&operands[1], false)?;
        let Rm::Reg(reg) = dst else {
            return Err(RasError::EncodingError(
                "movslq: destination must be a register".into(),
            ));
        };
        if dst_gpr.is_none_or(|gpr| gpr.width != 64) {
            return Err(RasError::EncodingError(
                "movslq: destination must be 64-bit".into(),
            ));
        }
        encode_op(Op {
            wide: true,
            opcode: &[0x63],
            reg,
            rm: src,
            ..Op::default()
        })
    }

    fn encode_lea(&self, width: Option<u8>, operands: &[String]) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError("lea requires 2 operands".into()));
        }
        let (src, _) = self.parse_rm(&operands[0], false)?;
        let Rm::Mem(_) = src else {
            return Err(RasError::EncodingError(
                "lea: first operand must be memory".into(),
            ));
        };
        let (dst, dst_gpr) = self.parse_rm(&operands[1], false)?;
        let Rm::Reg(reg) = dst else {
            return Err(RasError::EncodingError(
                "lea: second operand must be a register".into(),
            ));
        };
        let width = operand_width(width, &[dst_gpr])?;
        if width != 32 && width != 64 {
            return Err(RasError::EncodingError(
                "lea: unsupported operand width".into(),
            ));
        }
        encode_op(Op {
            wide: width == 64,
            opcode: &[0x8d],
            reg,
            rm: src,
            ..Op::default()
        })
    }

    fn encode_movzx(&self, opcode: &str, operands: &[String]) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError(
                "movzx requires two operands".into(),
            ));
        }
        let (src_bits, dst_width): (Option<u8>, Option<u8>) = match opcode {
            "movzbq" => (Some(8), Some(64)),
            "movzbl" => (Some(8), Some(32)),
            "movzwq" => (Some(16), Some(64)),
            "movzwl" => (Some(16), Some(32)),
            _ => (None, None),
        };
        let (src, src_gpr) = self.parse_rm(&operands[0], false)?;
        let (dst, dst_gpr) = self.parse_rm(&operands[1], false)?;
        let Rm::Reg(reg) = dst else {
            return Err(RasError::EncodingError(
                "movzx: destination must be a register".into(),
            ));
        };
        let src_bits = match (src_bits, src_gpr) {
            (Some(bits), _) => bits,
            (None, Some(gpr)) => gpr.width,
            (None, None) => {
                return Err(RasError::EncodingError(
                    "movzx: cannot infer source width".into(),
                ));
            }
        };
        if src_bits != 8 && src_bits != 16 {
            return Err(RasError::EncodingError("movzx: bad source width".into()));
        }
        let width = operand_width(dst_width, &[dst_gpr])?;
        if width != 16 && width != 32 && width != 64 {
            return Err(RasError::EncodingError(
                "movzx: bad destination width".into(),
            ));
        }
        let (byte_rex, high) = byte_flags(&[src_gpr]);
        encode_op(Op {
            legacy: if width == 16 { &[0x66] } else { &[] },
            wide: width == 64,
            byte_rex,
            high,
            opcode: &[0x0f, if src_bits == 8 { 0xb6 } else { 0xb7 }],
            reg,
            rm: src,
            ..Op::default()
        })
    }
}

impl InstructionEncoder for X86_64Encoder {
    fn take_rip_fixup(&mut self) -> Option<RipFixup> {
        self.rip_fixup.take()
    }

    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        self.rip_fixup = None;
        let opcode = inst.opcode.to_lowercase();
        if let Some((stem, width)) = alu_mnemonic(&opcode) {
            let encoded = self.encode_alu(stem, width, &inst.operands)?;
            let bytes = self.emit(encoded);
            self.position += bytes.len();
            return Ok(bytes);
        }
        if let Some((digit, width)) = shift_mnemonic(&opcode) {
            let encoded = self.encode_shift(digit, width, &inst.operands)?;
            let bytes = self.emit(encoded);
            self.position += bytes.len();
            return Ok(bytes);
        }
        if let Some((digit, width)) = unary_mnemonic(&opcode) {
            let encoded = self.encode_unary(digit, width, &inst.operands)?;
            let bytes = self.emit(encoded);
            self.position += bytes.len();
            return Ok(bytes);
        }
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

                if parse_xmm(a).is_none() && parse_xmm(b).is_none() {
                    code = self.emit(self.encode_integer_move(
                        if opcode == "movq" { Some(64) } else { None },
                        &inst.operands,
                    )?);
                } else {
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
                }
            }
            "movabsq" => {
                code = self.emit(self.encode_movabs(&inst.operands)?);
            }
            "ret" => code.push(0xC3),
            "endbr64" => code.extend_from_slice(&[0xf3, 0x0f, 0x1e, 0xfa]),
            "leave" => code.push(0xC9),
            "cvttsd2sil" | "cvttsd2siq" => {
                let enc = self.encode_cvttsd2si(opcode == "cvttsd2siq", &inst.operands)?;
                code = self.emit(enc);
            }
            "movsd" => {
                let enc =
                    self.encode_sse_move(&[0xf2], &[0x0f, 0x10], &[0x0f, 0x11], &inst.operands)?;
                code = self.emit(enc);
            }
            "movapd" => {
                let enc =
                    self.encode_sse_move(&[0x66], &[0x0f, 0x28], &[0x0f, 0x29], &inst.operands)?;
                code = self.emit(enc);
            }
            "movaps" => {
                let enc =
                    self.encode_sse_move(&[], &[0x0f, 0x28], &[0x0f, 0x29], &inst.operands)?;
                code = self.emit(enc);
            }
            "movups" => {
                let enc =
                    self.encode_sse_move(&[], &[0x0f, 0x10], &[0x0f, 0x11], &inst.operands)?;
                code = self.emit(enc);
            }
            "pxor" => {
                let enc = self.encode_sse_rm(&[0x66], &[0x0f, 0xef], &inst.operands)?;
                code = self.emit(enc);
            }
            "movdqa" => {
                let enc =
                    self.encode_sse_move(&[0x66], &[0x0f, 0x6f], &[0x0f, 0x7f], &inst.operands)?;
                code = self.emit(enc);
            }
            "addsd" | "addss" | "subsd" | "subss" | "mulsd" | "mulss" | "divsd" | "divss"
            | "paddd" => {
                let legacy = match opcode.as_str() {
                    "addss" | "subss" | "mulss" | "divss" => 0xf3,
                    "paddd" => 0x66,
                    _ => 0xf2,
                };
                let op = match opcode.as_str() {
                    "addsd" | "addss" => 0x58,
                    "mulsd" | "mulss" => 0x59,
                    "subsd" | "subss" => 0x5c,
                    "divsd" | "divss" => 0x5e,
                    _ => 0xfe,
                };
                let enc = self.encode_sse_rm(&[legacy], &[0x0f, op], &inst.operands)?;
                code = self.emit(enc);
            }
            "psrldq" => {
                let enc = self.encode_sse_shift(3, &inst.operands)?;
                code = self.emit(enc);
            }
            "cltq" => code.extend_from_slice(&[0x48, 0x98]),
            "movslq" => {
                let enc = self.encode_movsxd(&inst.operands)?;
                code = self.emit(enc);
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
            "leaq" | "lea" | "leal" => {
                let width = match opcode.as_str() {
                    "leal" => Some(32),
                    "leaq" => Some(64),
                    _ => None,
                };
                code = self.emit(self.encode_lea(width, &inst.operands)?);
            }
            "imulq" | "imull" | "imul" => {
                let wide = opcode != "imull";
                // Three-operand form: imul $imm, src, dst
                if inst.operands.len() == 3 {
                    let imm = parse_imm(inst.operands[0].trim_start_matches('$'))?;
                    let src_reg = self.parse_register(&inst.operands[1])?;
                    let dst_reg = self.parse_register(&inst.operands[2])?;
                    let rex = self.encode_rex(wide, dst_reg >> 3, 0, src_reg >> 3);
                    if wide || rex != 0x40 {
                        code.push(rex);
                    }
                    let modrm = 0xC0 | ((dst_reg & 7) << 3) | (src_reg & 7);
                    if (-128..=127).contains(&imm) {
                        code.push(0x6B);
                        code.push(modrm);
                        code.push(imm as u8);
                    } else {
                        code.push(0x69);
                        code.push(modrm);
                        code.extend_from_slice(&(imm as i32).to_le_bytes());
                    }
                    self.position += code.len();
                    return Ok(code);
                }
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "imul requires 2 or 3 operands".to_string(),
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
                let rex = self.encode_rex(wide, dst_reg >> 3, 0, src_reg >> 3);
                if wide || rex != 0x40 {
                    code.push(rex);
                }
                code.extend_from_slice(&[0x0F, 0xAF]);
                code.push(0xC0 | ((dst_reg & 7) << 3) | (src_reg & 7));
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
            "movb" | "movw" | "movl" => {
                let width = match opcode.as_str() {
                    "movb" => 8,
                    "movw" => 16,
                    _ => 32,
                };
                code = self.emit(self.encode_integer_move(Some(width), &inst.operands)?);
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
            "cmovzq" | "cmovz" | "cmoveq" | "cmove" | "cmovel" | "cmovzl" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError("cmovzq requires 2 operands".into()));
                }
                let src = self.parse_register(&inst.operands[0])?;
                let dst = self.parse_register(&inst.operands[1])?;
                let rex = self.encode_rex(!opcode.ends_with('l'), dst >> 3, 0, src >> 3);
                if rex != 0x40 {
                    code.push(rex);
                }
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
            "movzbq" | "movzbl" | "movzx" | "movzwq" | "movzwl" => {
                code = self.emit(self.encode_movzx(&opcode, &inst.operands)?);
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

#[derive(Clone, Copy)]
struct Gpr {
    num: u8,
    width: u8,
    high: bool,
}

const GPR_NAMES: [[&str; 4]; 16] = [
    ["rax", "eax", "ax", "al"],
    ["rcx", "ecx", "cx", "cl"],
    ["rdx", "edx", "dx", "dl"],
    ["rbx", "ebx", "bx", "bl"],
    ["rsp", "esp", "sp", "spl"],
    ["rbp", "ebp", "bp", "bpl"],
    ["rsi", "esi", "si", "sil"],
    ["rdi", "edi", "di", "dil"],
    ["r8", "r8d", "r8w", "r8b"],
    ["r9", "r9d", "r9w", "r9b"],
    ["r10", "r10d", "r10w", "r10b"],
    ["r11", "r11d", "r11w", "r11b"],
    ["r12", "r12d", "r12w", "r12b"],
    ["r13", "r13d", "r13w", "r13b"],
    ["r14", "r14d", "r14w", "r14b"],
    ["r15", "r15d", "r15w", "r15b"],
];

fn gpr(name: &str) -> Option<Gpr> {
    let name = name.trim().trim_start_matches('%');
    for (num, names) in GPR_NAMES.iter().enumerate() {
        if let Some(index) = names.iter().position(|candidate| *candidate == name) {
            let width = [64, 32, 16, 8][index];
            return Some(Gpr {
                num: num as u8,
                width,
                high: false,
            });
        }
    }
    ["ah", "ch", "dh", "bh"]
        .iter()
        .position(|candidate| *candidate == name)
        .map(|index| Gpr {
            num: 4 + index as u8,
            width: 8,
            high: true,
        })
}

#[derive(Clone)]
struct Mem {
    fs: bool,
    disp: i64,
    label: Option<String>,
    base: Option<u8>,
    index: Option<(u8, u8)>,
    rip: bool,
}

struct Encoded {
    bytes: Vec<u8>,
    rip: Option<RipFixup>,
}

enum Rm {
    Reg(u8),
    Mem(Mem),
}

impl Default for Rm {
    fn default() -> Self {
        Rm::Reg(0)
    }
}

#[derive(Default)]
struct Op<'a> {
    legacy: &'a [u8],
    wide: bool,
    byte_rex: bool,
    high: bool,
    opcode: &'a [u8],
    reg: u8,
    rm: Rm,
    imm: &'a [u8],
}

fn encode_op(op: Op<'_>) -> Result<Encoded, RasError> {
    let mut tail = Vec::new();
    let mut rex_x = 0;
    let mut rex_b = 0;
    let reg_field = (op.reg & 7) << 3;
    let mut seg = false;
    let mut rip_disp = None;
    match &op.rm {
        Rm::Reg(number) => {
            tail.push(0xc0 | reg_field | (number & 7));
            rex_b = number >> 3;
        }
        Rm::Mem(mem) if mem.rip => {
            tail.push(reg_field | 5);
            rip_disp = Some(tail.len());
            tail.extend_from_slice(&disp32(mem.disp)?);
        }
        Rm::Mem(mem) => {
            seg = mem.fs;
            if let Some((index, _)) = mem.index {
                rex_x = index >> 3;
            }
            let index_field = mem.index.map_or(4, |(index, _)| index & 7);
            let scale = mem.index.map_or(0, |(_, bits)| bits);
            match mem.base {
                None => {
                    tail.push(reg_field | 4);
                    tail.push((scale << 6) | (index_field << 3) | 5);
                    tail.extend_from_slice(&disp32(mem.disp)?);
                }
                Some(base) => {
                    rex_b = base >> 3;
                    let need_sib = mem.index.is_some() || base & 7 == 4;
                    let mode = if mem.disp == 0 && base & 7 != 5 {
                        0
                    } else if i8::try_from(mem.disp).is_ok() {
                        1
                    } else {
                        2
                    };
                    let rm_field = if need_sib { 4 } else { base & 7 };
                    tail.push((mode << 6) | reg_field | rm_field);
                    if need_sib {
                        tail.push((scale << 6) | (index_field << 3) | (base & 7));
                    }
                    match mode {
                        1 => tail.push(mem.disp as u8),
                        2 => tail.extend_from_slice(&disp32(mem.disp)?),
                        _ => {}
                    }
                }
            }
        }
    }
    let rex = 0x40 | (u8::from(op.wide) << 3) | ((op.reg >> 3) << 2) | (rex_x << 1) | rex_b;
    let needs_rex = rex != 0x40 || op.byte_rex;
    if op.high && needs_rex {
        return Err(RasError::EncodingError(
            "High byte registers cannot use REX".into(),
        ));
    }
    let mut bytes = Vec::new();
    if seg {
        bytes.push(0x64);
    }
    bytes.extend_from_slice(op.legacy);
    if needs_rex {
        bytes.push(rex);
    }
    bytes.extend_from_slice(op.opcode);
    let tail_start = bytes.len();
    bytes.extend(tail);
    let rip = rip_disp.map(|position| RipFixup {
        disp_offset: tail_start + position,
        trailing: op.imm.len(),
    });
    bytes.extend_from_slice(op.imm);
    Ok(Encoded { bytes, rip })
}

fn disp32(value: i64) -> Result<[u8; 4], RasError> {
    i32::try_from(value)
        .map(i32::to_le_bytes)
        .map_err(|_| RasError::EncodingError(format!("Displacement out of range: {}", value)))
}

fn parse_imm(text: &str) -> Result<i64, RasError> {
    let text = text.trim();
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let magnitude = match digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        Some(hex) => i64::from_str_radix(hex, 16),
        None => digits.parse::<i64>(),
    }
    .map_err(|_| RasError::EncodingError(format!("Invalid immediate: {}", text)))?;
    Ok(if negative { -magnitude } else { magnitude })
}

fn is_symbol(text: &str) -> bool {
    let mut chars = text.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '.')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$')
}

fn imm_bytes(value: i64, width: u8) -> Result<Vec<u8>, RasError> {
    let fits = match width {
        8 => (-128..=255).contains(&value),
        16 => (-32768..=65535).contains(&value),
        32 => (-(1i64 << 31)..=(1i64 << 32) - 1).contains(&value),
        _ => i32::try_from(value).is_ok(),
    };
    if !fits {
        return Err(RasError::EncodingError(format!(
            "Immediate {} out of range for {}-bit operand",
            value, width
        )));
    }
    Ok(match width {
        8 => vec![value as u8],
        16 => (value as u16).to_le_bytes().to_vec(),
        32 => (value as u32).to_le_bytes().to_vec(),
        _ => (value as i32).to_le_bytes().to_vec(),
    })
}

fn operand_width(width: Option<u8>, gprs: &[Option<Gpr>]) -> Result<u8, RasError> {
    let mut found = width;
    for gpr in gprs.iter().flatten() {
        match found {
            None => found = Some(gpr.width),
            Some(existing) if existing == gpr.width => {}
            Some(_) => {
                return Err(RasError::EncodingError(
                    "Operand widths do not match".into(),
                ));
            }
        }
    }
    found.ok_or_else(|| RasError::EncodingError("Cannot infer operand width".into()))
}

fn check_alu_width(width: u8) -> Result<(), RasError> {
    if matches!(width, 8 | 32 | 64) {
        Ok(())
    } else {
        Err(RasError::EncodingError(format!(
            "Unsupported operand width: {}",
            width
        )))
    }
}

fn byte_flags(gprs: &[Option<Gpr>]) -> (bool, bool) {
    gprs.iter()
        .flatten()
        .filter(|gpr| gpr.width == 8)
        .fold((false, false), |(rex, high), gpr| {
            if gpr.high {
                (rex, true)
            } else {
                (rex || (4..8).contains(&gpr.num), high)
            }
        })
}

const ALU_STEMS: [&str; 7] = ["add", "or", "and", "sub", "xor", "cmp", "test"];

fn alu_mnemonic(name: &str) -> Option<(&'static str, Option<u8>)> {
    if let Some(stem) = ALU_STEMS.iter().find(|candidate| **candidate == name) {
        return Some((*stem, None));
    }
    [('q', 64), ('l', 32), ('b', 8)]
        .into_iter()
        .find_map(|(suffix, width)| {
            let stem = name.strip_suffix(suffix)?;
            ALU_STEMS
                .iter()
                .find(|candidate| **candidate == stem)
                .map(|candidate| (*candidate, Some(width)))
        })
}

fn sized_stem(opcode: &str, stem: &str) -> Option<Option<u8>> {
    if opcode == stem {
        return Some(None);
    }
    match opcode.strip_prefix(stem)? {
        "b" => Some(Some(8)),
        "l" => Some(Some(32)),
        "q" => Some(Some(64)),
        _ => None,
    }
}

fn shift_mnemonic(opcode: &str) -> Option<(u8, Option<u8>)> {
    [("sar", 7), ("shl", 4), ("sal", 4), ("shr", 5)]
        .into_iter()
        .find_map(|(stem, digit)| sized_stem(opcode, stem).map(|width| (digit, width)))
}

fn unary_mnemonic(opcode: &str) -> Option<(u8, Option<u8>)> {
    [("neg", 3), ("mul", 4), ("idiv", 7), ("div", 6)]
        .into_iter()
        .find_map(|(stem, digit)| sized_stem(opcode, stem).map(|width| (digit, width)))
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
        let load = |mem: &str| {
            enc.encode_integer_move(Some(64), &[mem.to_string(), "%rax".to_string()])
                .expect("encode")
                .bytes
        };

        assert_eq!(load("8(%rsp)"), vec![0x48, 0x8b, 0x44, 0x24, 0x08]);
        assert_eq!(load("(%rsp)"), vec![0x48, 0x8b, 0x04, 0x24]);

        assert_eq!(load("(%r13)"), vec![0x49, 0x8b, 0x45, 0x00]);
        assert_eq!(load("8(%r13)"), vec![0x49, 0x8b, 0x45, 0x08]);

        assert_eq!(load("(%rbp)"), vec![0x48, 0x8b, 0x45, 0x00]);

        assert_eq!(load("(%r12)"), vec![0x49, 0x8b, 0x04, 0x24]);
    }

    fn encode_one(opcode: &str, operands: &[&str]) -> Vec<u8> {
        let inst = ParsedInstruction {
            opcode: opcode.to_string(),
            operands: operands.iter().map(|op| op.to_string()).collect(),
        };
        X86_64Encoder::new()
            .encode_instruction(&inst)
            .expect("encode")
    }

    #[test]
    fn movabsq_matches_gnu_as() {
        assert_eq!(
            encode_one("movabsq", &["$5", "%rax"]),
            vec![0x48, 0xb8, 5, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            encode_one("movabsq", &["$4294967296", "%r9"]),
            vec![0x49, 0xb9, 0, 0, 0, 0, 1, 0, 0, 0]
        );
    }

    #[test]
    fn sse_moves_match_gnu_as() {
        assert_eq!(
            encode_one("movaps", &["%xmm0", "112(%rsp)"]),
            vec![0x0f, 0x29, 0x44, 0x24, 0x70]
        );
        assert_eq!(
            encode_one("movaps", &["%xmm9", "%xmm1"]),
            vec![0x41, 0x0f, 0x28, 0xc9]
        );
        assert_eq!(
            encode_one("movups", &["%xmm0", "297(%rsp)"]),
            vec![0x0f, 0x11, 0x84, 0x24, 0x29, 0x01, 0x00, 0x00]
        );
        assert_eq!(
            encode_one("pxor", &["%xmm10", "%xmm3"]),
            vec![0x66, 0x41, 0x0f, 0xef, 0xda]
        );
    }

    #[test]
    fn shifts_and_unary_match_gnu_as() {
        assert_eq!(encode_one("sarl", &["$31", "%eax"]), vec![0xc1, 0xf8, 0x1f]);
        assert_eq!(encode_one("sarl", &["%ecx"]), vec![0xd1, 0xf9]);
        assert_eq!(
            encode_one("sarl", &["$1", "-4(%rbp)"]),
            vec![0xd1, 0x7d, 0xfc]
        );
        assert_eq!(encode_one("shll", &["%cl", "%r9d"]), vec![0x41, 0xd3, 0xe1]);
        assert_eq!(encode_one("sall", &["$2", "%eax"]), vec![0xc1, 0xe0, 0x02]);
        assert_eq!(encode_one("shrl", &["%eax"]), vec![0xd1, 0xe8]);
        assert_eq!(encode_one("negq", &["%rdx"]), vec![0x48, 0xf7, 0xda]);
        assert_eq!(encode_one("negl", &["%r13d"]), vec![0x41, 0xf7, 0xdd]);
        assert_eq!(encode_one("negl", &["-4(%rbp)"]), vec![0xf7, 0x5d, 0xfc]);
        assert_eq!(encode_one("mulq", &["%rcx"]), vec![0x48, 0xf7, 0xe1]);
        assert_eq!(
            encode_one("divq", &["-8(%rbp)"]),
            vec![0x48, 0xf7, 0x75, 0xf8]
        );
    }
}
