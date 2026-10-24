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
                Err(_) => {
                    let (label, addend) = rip_expression(disp_text).ok_or_else(invalid)?;
                    mem.label = Some(label.to_string());
                    mem.disp = addend;
                }
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
        (op.contains('(') && op.contains(')')) || op.starts_with("%fs:") || parse_imm(op).is_ok()
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
            "adc" => 2,
            "sbb" => 3,
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
            let imm = if width == 16 {
                i64::from(imm as u16 as i16)
            } else if width == 32 {
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
                if width == 16 {
                    bytes.push(0x66);
                }
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
                legacy: if width == 16 { &[0x66] } else { &[] },
                wide: width == 64,
                byte_rex,
                high,
                opcode: &[opcode],
                reg: digit,
                rm: dst_rm,
                imm: &imm_field,
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
            legacy: if width == 16 { &[0x66] } else { &[] },
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
            legacy: if width == 16 { &[0x66] } else { &[] },
            wide: width == 64,
            byte_rex,
            high,
            opcode: &opcode,
            reg: digit,
            rm,
            imm: &imm,
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
            legacy: if width == 16 { &[0x66] } else { &[] },
            wide: width == 64,
            byte_rex,
            high,
            opcode: &[if digit <= 1 {
                if width == 8 { 0xfe } else { 0xff }
            } else if width == 8 {
                0xf6
            } else {
                0xf7
            }],
            reg: digit,
            rm,
            ..Op::default()
        })
    }

    fn encode_cmov(
        &self,
        cc: u8,
        width: Option<u8>,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        let [src, dst] = operands else {
            return Err(RasError::EncodingError("cmov requires 2 operands".into()));
        };
        let (src_rm, src_gpr) = self.parse_rm(src, false)?;
        let (dst_rm, dst_gpr) = self.parse_rm(dst, false)?;
        let Rm::Reg(reg) = dst_rm else {
            return Err(RasError::EncodingError(
                "cmov: destination must be a register".into(),
            ));
        };
        let width = operand_width(width, &[src_gpr, dst_gpr])?;
        if !matches!(width, 16 | 32 | 64) {
            return Err(RasError::EncodingError(format!(
                "cmov: unsupported operand width: {}",
                width
            )));
        }
        encode_op(Op {
            legacy: if width == 16 { &[0x66] } else { &[] },
            wide: width == 64,
            opcode: &[0x0f, 0x40 | cc],
            reg,
            rm: src_rm,
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

    fn encode_x87(&self, opcode: &str, operands: &[String]) -> Result<Encoded, RasError> {
        let memory = match opcode {
            "flds" => Some((0xd9, 0)),
            "fldl" => Some((0xdd, 0)),
            "fldt" => Some((0xdb, 5)),
            "fstl" => Some((0xdd, 2)),
            "fstpl" => Some((0xdd, 3)),
            "fstpt" => Some((0xdb, 7)),
            "fildl" => Some((0xdb, 0)),
            "fildq" => Some((0xdf, 5)),
            "fistpl" => Some((0xdb, 3)),
            "fistpq" => Some((0xdf, 7)),
            "fldcw" => Some((0xd9, 5)),
            "fnstcw" => Some((0xd9, 7)),
            "fadds" => Some((0xd8, 0)),
            "faddl" => Some((0xdc, 0)),
            "fmull" => Some((0xdc, 1)),
            _ => None,
        };
        if let Some((byte, digit)) = memory {
            let [operand] = operands else {
                return Err(RasError::EncodingError(
                    "x87 memory operation requires 1 operand".into(),
                ));
            };
            return encode_op(Op {
                opcode: &[byte],
                reg: digit,
                rm: Rm::Mem(self.parse_mem(operand)?),
                ..Op::default()
            });
        }
        let stack = |operand: &str| -> Result<u8, RasError> {
            let text = operand.trim().trim_start_matches('%');
            if text == "st" {
                return Ok(0);
            }
            text.strip_prefix("st(")
                .and_then(|s| s.strip_suffix(')'))
                .and_then(|s| s.parse::<u8>().ok())
                .filter(|n| *n < 8)
                .ok_or_else(|| RasError::EncodingError("expected x87 stack register".into()))
        };
        let (byte, field) = match (opcode, operands) {
            ("fld" | "fxch" | "fstp", [operand]) => {
                let reg = stack(operand)?;
                match opcode {
                    "fld" => (0xd9, 0xc0 + reg),
                    "fxch" => (0xd9, 0xc8 + reg),
                    _ => (0xdd, 0xd8 + reg),
                }
            }
            (
                "fmul" | "fsub" | "fmulp" | "fdivp" | "fsubrp" | "fcomi" | "fcomip" | "fucomip",
                [src, dst],
            ) => {
                let src = stack(src)?;
                let dst = stack(dst)?;
                match opcode {
                    "fmul" if dst == 0 => (0xd8, 0xc8 + src),
                    "fmul" if src == 0 => (0xdc, 0xc8 + dst),
                    "fsub" if dst == 0 => (0xd8, 0xe0 + src),
                    "fsub" if src == 0 => (0xdc, 0xe0 + dst),
                    "fmulp" if src == 0 => (0xde, 0xc8 + dst),
                    "fdivp" if src == 0 => (0xde, 0xf0 + dst),
                    "fsubrp" if src == 0 => (0xde, 0xe8 + dst),
                    "fcomi" if dst == 0 => (0xdb, 0xf0 + src),
                    "fcomip" if dst == 0 => (0xdf, 0xf0 + src),
                    "fucomip" if dst == 0 => (0xdf, 0xe8 + src),
                    _ => {
                        return Err(RasError::EncodingError(
                            "unsupported x87 stack operands".into(),
                        ));
                    }
                }
            }
            _ => return Err(RasError::EncodingError("unsupported x87 operands".into())),
        };
        Ok(Encoded {
            bytes: vec![byte, field],
            rip: None,
        })
    }

    fn encode_cvtt_to_int(
        &self,
        legacy: u8,
        wide: bool,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError("cvtt requires 2 operands".into()));
        }
        let (src, _) = self.parse_rm(&operands[0], true)?;
        let (dst, dst_gpr) = self.parse_rm(&operands[1], false)?;
        let Rm::Reg(reg) = dst else {
            return Err(RasError::EncodingError(
                "cvtt: destination must be a register".into(),
            ));
        };
        let expected = if wide { 64 } else { 32 };
        if dst_gpr.is_none_or(|gpr| gpr.width != expected) {
            return Err(RasError::EncodingError(
                "cvtt: destination width does not match suffix".into(),
            ));
        }
        encode_op(Op {
            legacy: &[legacy],
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
            "movsbw" => (Some(8), Some(16)),
            "movzbq" | "movsbq" => (Some(8), Some(64)),
            "movzbl" | "movsbl" => (Some(8), Some(32)),
            "movzwq" | "movswq" => (Some(16), Some(64)),
            "movzwl" | "movswl" => (Some(16), Some(32)),
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
            opcode: &[
                0x0f,
                match (opcode.starts_with("movs"), src_bits == 8) {
                    (false, true) => 0xb6,
                    (false, false) => 0xb7,
                    (true, true) => 0xbe,
                    (true, false) => 0xbf,
                },
            ],
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
        if let Some((cc, width)) = cmov_mnemonic(&opcode) {
            let encoded = self.encode_cmov(cc, width, &inst.operands)?;
            let bytes = self.emit(encoded);
            self.position += bytes.len();
            return Ok(bytes);
        }
        if let Some(cc) = opcode.strip_prefix("set").and_then(condition_code) {
            let [operand] = inst.operands.as_slice() else {
                return Err(RasError::EncodingError("setcc requires 1 operand".into()));
            };
            let (rm, gpr) = self.parse_rm(operand, false)?;
            operand_width(Some(8), &[gpr])?;
            let (byte_rex, high) = byte_flags(&[gpr]);
            let enc = encode_op(Op {
                byte_rex,
                high,
                opcode: &[0x0f, 0x90 | cc],
                rm,
                ..Op::default()
            })?;
            let bytes = self.emit(enc);
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
                    let a_xmm = parse_xmm(a);
                    let b_xmm = parse_xmm(b);
                    let transfer = match (gpr(a), b_xmm, a_xmm, gpr(b)) {
                        (Some(reg), Some(xmm), _, _) => Some((reg, xmm, 0x6e)),
                        (_, _, Some(xmm), Some(reg)) => Some((reg, xmm, 0x7e)),
                        _ => None,
                    };
                    if let Some((reg, xmm, op)) = transfer {
                        operand_width(Some(64), &[Some(reg)])?;
                        code = self.emit(encode_op(Op {
                            legacy: &[0x66],
                            wide: true,
                            opcode: &[0x0f, op],
                            reg: xmm,
                            rm: Rm::Reg(reg.num),
                            ..Op::default()
                        })?);
                    } else if b_xmm.is_some() {
                        code = self.emit(self.encode_sse_rm(
                            &[0xf3],
                            &[0x0f, 0x7e],
                            &inst.operands,
                        )?);
                    } else if let Some(reg) = a_xmm {
                        code = self.emit(encode_op(Op {
                            legacy: &[0x66],
                            opcode: &[0x0f, 0xd6],
                            reg,
                            rm: Rm::Mem(self.parse_mem(b)?),
                            ..Op::default()
                        })?);
                    } else {
                        return Err(RasError::EncodingError("unsupported movq operands".into()));
                    }
                }
            }
            "movabsq" => {
                code = self.emit(self.encode_movabs(&inst.operands)?);
            }
            "call" | "jmp" => {
                let [target] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(format!(
                        "{} requires 1 operand",
                        opcode
                    )));
                };
                let target = target.trim().strip_prefix('*').ok_or_else(|| {
                    RasError::EncodingError(format!("{}: expected indirect operand", opcode))
                })?;
                let (rm, _) = self.parse_rm(target, false)?;
                let digit = if opcode == "call" { 2 } else { 4 };
                let enc = encode_op(Op {
                    opcode: &[0xff],
                    reg: digit,
                    rm,
                    ..Op::default()
                })?;
                code = self.emit(enc);
            }
            "flds" | "fldl" | "fldt" | "fstl" | "fstpl" | "fstpt" | "fildl" | "fildq"
            | "fistpl" | "fistpq" | "fldcw" | "fnstcw" | "fadds" | "faddl" | "fmull" | "fld"
            | "fxch" | "fstp" | "fmul" | "fsub" | "fmulp" | "fdivp" | "fsubrp" | "fcomi"
            | "fcomip" | "fucomip" => {
                code = self.emit(self.encode_x87(&opcode, &inst.operands)?);
            }
            "fldz" | "fchs" => {
                if !inst.operands.is_empty() {
                    return Err(RasError::EncodingError(
                        "x87 constant/sign operation takes no operands".into(),
                    ));
                }
                code.extend_from_slice(&[0xd9, if opcode == "fldz" { 0xee } else { 0xe0 }]);
            }
            "fld1" => code.extend_from_slice(&[0xd9, 0xe8]),
            "bswap" | "bswapl" | "bswapq" => {
                let [operand] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError("bswap requires 1 operand".into()));
                };
                let reg = gpr(operand)
                    .filter(|g| matches!(g.width, 32 | 64))
                    .ok_or_else(|| {
                        RasError::EncodingError("bswap requires 32/64-bit register".into())
                    })?;
                if (opcode == "bswapl" && reg.width != 32)
                    || (opcode == "bswapq" && reg.width != 64)
                {
                    return Err(RasError::EncodingError("bswap width mismatch".into()));
                }
                let rex = self.encode_rex(reg.width == 64, 0, 0, reg.num >> 3);
                if rex != 0x40 {
                    code.push(rex);
                }
                code.extend_from_slice(&[0x0f, 0xc8 | (reg.num & 7)]);
            }
            "btl" | "btq" | "btsl" | "btsq" | "btrl" | "btrq" | "btcl" | "btcq" | "bsrl"
            | "bsrq" => {
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "bit operation requires 2 operands".into(),
                    ));
                };
                let width = if opcode.ends_with('q') { 64 } else { 32 };
                let scan = opcode.starts_with("bsr");
                let (rm, rm_gpr) = self.parse_rm(if scan { src } else { dst }, false)?;
                let digit = if opcode.starts_with("bts") {
                    5
                } else if opcode.starts_with("btr") {
                    6
                } else if opcode.starts_with("btc") {
                    7
                } else {
                    4
                };
                let mut imm = Vec::new();
                let (op, reg) =
                    if !scan && let Some(value) = src.strip_prefix('$') {
                        imm.push(u8::try_from(parse_imm(value)?).map_err(|_| {
                            RasError::EncodingError("bit index out of range".into())
                        })?);
                        operand_width(Some(width), &[rm_gpr])?;
                        (0xba, digit)
                    } else {
                        let reg = gpr(if scan { dst } else { src }).ok_or_else(|| {
                            RasError::EncodingError("bit operation requires register".into())
                        })?;
                        operand_width(Some(width), &[rm_gpr, Some(reg)])?;
                        (if scan { 0xbd } else { 0xa3 + (digit - 4) * 8 }, reg.num)
                    };
                code = self.emit(encode_op(Op {
                    wide: width == 64,
                    opcode: &[0x0f, op],
                    reg,
                    rm,
                    imm: &imm,
                    ..Op::default()
                })?);
            }
            "ret" => code.push(0xC3),
            "endbr64" => code.extend_from_slice(&[0xf3, 0x0f, 0x1e, 0xfa]),
            "leave" => code.push(0xC9),
            "cvttsd2sil" | "cvttsd2siq" | "cvttss2sil" | "cvttss2siq" => {
                let legacy = if opcode.starts_with("cvttss") {
                    0xf3
                } else {
                    0xf2
                };
                let enc = self.encode_cvtt_to_int(legacy, opcode.ends_with('q'), &inst.operands)?;
                code = self.emit(enc);
            }
            "cvtsi2sdl" | "cvtsi2sdq" | "cvtsi2ssq" => {
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "cvtsi2sd requires 2 operands".into(),
                    ));
                };
                let (rm, src_reg) = self.parse_rm(src, false)?;
                let wide = opcode.ends_with('q');
                operand_width(Some(if wide { 64 } else { 32 }), &[src_reg])?;
                let reg = parse_xmm(dst).ok_or_else(|| {
                    RasError::EncodingError("cvtsi2sd requires xmm destination".into())
                })?;
                code = self.emit(encode_op(Op {
                    legacy: &[if opcode == "cvtsi2ssq" { 0xf3 } else { 0xf2 }],
                    wide,
                    opcode: &[0x0f, 0x2a],
                    reg,
                    rm,
                    ..Op::default()
                })?);
            }
            "movdqu" => {
                code = self.emit(self.encode_sse_move(
                    &[0xf3],
                    &[0x0f, 0x6f],
                    &[0x0f, 0x7f],
                    &inst.operands,
                )?);
            }
            "movhps" => {
                if inst.operands.len() != 2
                    || inst.operands.iter().filter(|op| self.is_memory(op)).count() != 1
                {
                    return Err(RasError::EncodingError(
                        "movhps requires memory and xmm operands".into(),
                    ));
                }
                code = self.emit(self.encode_sse_move(
                    &[],
                    &[0x0f, 0x16],
                    &[0x0f, 0x17],
                    &inst.operands,
                )?);
            }
            "punpcklqdq" | "punpckldq" | "pcmpeqd" | "comisd" | "ucomisd" | "paddb" | "paddw"
            | "psubd" | "pand" => {
                let op = match opcode.as_str() {
                    "punpcklqdq" => 0x6c,
                    "punpckldq" => 0x62,
                    "pcmpeqd" => 0x76,
                    "paddb" => 0xfc,
                    "paddw" => 0xfd,
                    "psubd" => 0xfa,
                    "pand" => 0xdb,
                    "comisd" => 0x2f,
                    _ => 0x2e,
                };
                code = self.emit(self.encode_sse_rm(&[0x66], &[0x0f, op], &inst.operands)?);
            }
            "pshufd" | "pshuflw" | "shufpd" => {
                let [imm, src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "shuffle requires 3 operands".into(),
                    ));
                };
                let imm = imm
                    .strip_prefix('$')
                    .ok_or_else(|| RasError::EncodingError("shuffle requires immediate".into()))?;
                let imm = u8::try_from(parse_imm(imm)?).map_err(|_| {
                    RasError::EncodingError("shuffle immediate out of range".into())
                })?;
                let (rm, _) = self.parse_rm(src, true)?;
                let reg = parse_xmm(dst).ok_or_else(|| {
                    RasError::EncodingError("shuffle requires xmm destination".into())
                })?;
                code = self.emit(encode_op(Op {
                    legacy: &[if opcode == "pshuflw" { 0xf2 } else { 0x66 }],
                    opcode: &[0x0f, if opcode == "shufpd" { 0xc6 } else { 0x70 }],
                    reg,
                    rm,
                    imm: &[imm],
                    ..Op::default()
                })?);
            }
            "movss" => {
                code = self.emit(self.encode_sse_move(
                    &[0xf3],
                    &[0x0f, 0x10],
                    &[0x0f, 0x11],
                    &inst.operands,
                )?);
            }
            "movlpd" => {
                if inst.operands.len() != 2
                    || inst.operands.iter().filter(|op| self.is_memory(op)).count() != 1
                {
                    return Err(RasError::EncodingError(
                        "movlpd requires memory and xmm operands".into(),
                    ));
                }
                code = self.emit(self.encode_sse_move(
                    &[0x66],
                    &[0x0f, 0x12],
                    &[0x0f, 0x13],
                    &inst.operands,
                )?);
            }
            "movhlps" => {
                if inst.operands.iter().any(|op| parse_xmm(op).is_none()) {
                    return Err(RasError::EncodingError(
                        "movhlps requires xmm registers".into(),
                    ));
                }
                code = self.emit(self.encode_sse_rm(&[], &[0x0f, 0x12], &inst.operands)?);
            }
            "cvtss2sd" | "minsd" => {
                code = self.emit(self.encode_sse_rm(
                    &[if opcode == "minsd" { 0xf2 } else { 0xf3 }],
                    &[0x0f, if opcode == "minsd" { 0x5d } else { 0x5a }],
                    &inst.operands,
                )?);
            }
            "pextrw" | "pinsrw" => {
                let [imm, src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "word transfer requires 3 operands".into(),
                    ));
                };
                let imm = imm.strip_prefix('$').ok_or_else(|| {
                    RasError::EncodingError("word transfer requires immediate".into())
                })?;
                let imm = u8::try_from(parse_imm(imm)?).map_err(|_| {
                    RasError::EncodingError("word transfer immediate out of range".into())
                })?;
                let (reg, rm) = if opcode == "pextrw" {
                    let reg = gpr(dst).filter(|g| g.width == 32).ok_or_else(|| {
                        RasError::EncodingError("pextrw requires 32-bit destination".into())
                    })?;
                    let src = parse_xmm(src).ok_or_else(|| {
                        RasError::EncodingError("pextrw requires xmm source".into())
                    })?;
                    (reg.num, Rm::Reg(src))
                } else {
                    let reg = parse_xmm(dst).ok_or_else(|| {
                        RasError::EncodingError("pinsrw requires xmm destination".into())
                    })?;
                    let (rm, gpr) = self.parse_rm(src, false)?;
                    operand_width(Some(32), &[gpr])?;
                    (reg, rm)
                };
                code = self.emit(encode_op(Op {
                    legacy: &[0x66],
                    opcode: &[0x0f, if opcode == "pextrw" { 0xc5 } else { 0xc4 }],
                    reg,
                    rm,
                    imm: &[imm],
                    ..Op::default()
                })?);
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
            "cwtl" => code.push(0x98),
            "ud2" => code.extend_from_slice(&[0x0f, 0x0b]),
            "cltq" => code.extend_from_slice(&[0x48, 0x98]),
            "movslq" => {
                let enc = self.encode_movsxd(&inst.operands)?;
                code = self.emit(enc);
            }
            "pushq" | "push" | "popq" | "pop" => {
                let [operand] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "push/pop requires 1 operand".into(),
                    ));
                };
                let push = opcode.starts_with("push");
                if let Some(imm) = operand.strip_prefix('$').filter(|_| push) {
                    let imm = parse_imm(imm)?;
                    let small = i8::try_from(imm).is_ok();
                    code.push(if small { 0x6a } else { 0x68 });
                    code.extend(imm_bytes(imm, if small { 8 } else { 64 })?);
                } else {
                    let (rm, reg) = self.parse_rm(operand, false)?;
                    operand_width(Some(64), &[reg])?;
                    if let Rm::Reg(reg) = rm {
                        if reg >= 8 {
                            code.push(0x41);
                        }
                        code.push((if push { 0x50 } else { 0x58 }) | (reg & 7));
                    } else {
                        code = self.emit(encode_op(Op {
                            opcode: &[if push { 0xff } else { 0x8f }],
                            reg: if push { 6 } else { 0 },
                            rm,
                            ..Op::default()
                        })?);
                    }
                }
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
                let suffix_width = match opcode.as_str() {
                    "imull" => Some(32),
                    "imulq" => Some(64),
                    _ => None,
                };
                if inst.operands.len() == 1 {
                    code = self.emit(self.encode_unary(5, suffix_width, &inst.operands)?);
                    self.position += code.len();
                    return Ok(code);
                }
                // Three-operand form: imul $imm, src, dst
                if inst.operands.len() == 3 {
                    let imm = parse_imm(inst.operands[0].trim_start_matches('$'))?;
                    let (rm, src_gpr) = self.parse_rm(&inst.operands[1], false)?;
                    let dst = gpr(&inst.operands[2]).ok_or_else(|| {
                        RasError::EncodingError("imul requires register destination".into())
                    })?;
                    let width = operand_width(suffix_width, &[src_gpr, Some(dst)])?;
                    if !matches!(width, 32 | 64) {
                        return Err(RasError::EncodingError("unsupported imul width".into()));
                    }
                    let small = (-128..=127).contains(&imm);
                    let immediate = imm_bytes(imm, if small { 8 } else { width })?;
                    code = self.emit(encode_op(Op {
                        wide: width == 64,
                        opcode: &[if small { 0x6b } else { 0x69 }],
                        reg: dst.num,
                        rm,
                        imm: &immediate,
                        ..Op::default()
                    })?);
                    self.position += code.len();
                    return Ok(code);
                }
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "imul requires 2 or 3 operands".to_string(),
                    ));
                }
                // AT&T order: `imul src, dst`. 0F AF is IMUL r, r/m where the
                // reg field is the destination and r/m is the source.
                let (src_rm, src_gpr) = self.parse_rm(&inst.operands[0], false)?;
                let (dst_rm, dst_gpr) = self.parse_rm(&inst.operands[1], false)?;
                let Rm::Reg(reg) = dst_rm else {
                    return Err(RasError::EncodingError(
                        "imul: destination must be a register".to_string(),
                    ));
                };
                let width = operand_width(suffix_width, &[src_gpr, dst_gpr])?;
                if width != 32 && width != 64 {
                    return Err(RasError::EncodingError(format!(
                        "imul: unsupported operand width: {}",
                        width
                    )));
                }
                let enc = encode_op(Op {
                    wide: width == 64,
                    opcode: &[0x0f, 0xaf],
                    reg,
                    rm: src_rm,
                    ..Op::default()
                })?;
                code = self.emit(enc);
            }
            "rep" | "repe" | "repz" | "repne" | "repnz" | "lock" | "notrack" => {
                let Some(first) = inst.operands.first() else {
                    return Err(RasError::EncodingError(
                        "prefix requires instruction".into(),
                    ));
                };
                let mut parts = first.splitn(2, char::is_whitespace);
                let nested_opcode = parts.next().unwrap().to_string();
                let mut operands = Vec::new();
                if let Some(operand) = parts.next() {
                    operands.push(operand.trim().to_string());
                }
                operands.extend_from_slice(&inst.operands[1..]);
                if opcode == "notrack"
                    && (!matches!(nested_opcode.as_str(), "call" | "jmp")
                        || operands.len() != 1
                        || !operands[0].starts_with('*'))
                {
                    return Err(RasError::EncodingError(
                        "notrack requires indirect branch".into(),
                    ));
                }
                if opcode == "lock"
                    && (alu_mnemonic(&nested_opcode)
                        .is_none_or(|(stem, _)| matches!(stem, "cmp" | "test"))
                        || !operands.last().is_some_and(|op| self.is_memory(op)))
                {
                    return Err(RasError::EncodingError(
                        "unsupported lock instruction".into(),
                    ));
                }
                if !matches!(opcode.as_str(), "lock" | "notrack")
                    && !matches!(nested_opcode.as_str(), "ret" | "nop")
                    && !["movs", "stos", "cmps", "scas", "lods"]
                        .iter()
                        .any(|stem| sized_stem(&nested_opcode, stem).is_some())
                {
                    return Err(RasError::EncodingError(
                        "unsupported repeat instruction".into(),
                    ));
                }
                let prefix = match opcode.as_str() {
                    "lock" => 0xf0,
                    "notrack" => 0x3e,
                    "repne" | "repnz" => 0xf2,
                    _ => 0xf3,
                };
                code.push(prefix);
                code.extend(self.encode_instruction(&ParsedInstruction {
                    opcode: nested_opcode,
                    operands,
                })?);
                let mut at = 1;
                while code.get(at).is_some_and(|byte| matches!(byte, 0x64 | 0x66)) {
                    code.swap(at - 1, at);
                    at += 1;
                }
                if let Some(fixup) = &mut self.rip_fixup {
                    fixup.disp_offset += 1;
                }
                self.position += 1;
                return Ok(code);
            }
            "movsb" | "movsw" | "movsl" | "movsq" | "stosb" | "stosw" | "stosl" | "stosq"
            | "cmpsb" | "cmpsw" | "cmpsl" | "cmpsq" | "scasb" | "scasw" | "scasl" | "scasq"
            | "lodsb" | "lodsw" | "lodsl" | "lodsq" => {
                if !inst.operands.is_empty() {
                    return Err(RasError::EncodingError(
                        "explicit string operands unsupported".into(),
                    ));
                }
                if opcode.ends_with('w') {
                    code.push(0x66);
                }
                if opcode.ends_with('q') {
                    code.push(0x48);
                }
                let base = if opcode.starts_with("movs") {
                    0xa4
                } else if opcode.starts_with("cmps") {
                    0xa6
                } else if opcode.starts_with("stos") {
                    0xaa
                } else if opcode.starts_with("lods") {
                    0xac
                } else {
                    0xae
                };
                code.push(base + u8::from(!opcode.ends_with('b')));
            }
            "nop" => {
                if !inst.operands.is_empty() {
                    return Err(RasError::EncodingError("nop takes no operands".to_string()));
                }
                code.push(0x90);
            }
            // REX.W 0x99: sign-extend rax into rdx:rax (needed before idivq)
            "cqto" | "cqo" => {
                code.push(0x48);
                code.push(0x99);
            }
            // REX.W 0x99 for 32-bit sign-extend eax into edx:eax
            "cdq" | "cltd" => {
                code.push(0x99);
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
            "movd" => {
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError("movd requires 2 operands".into()));
                };
                let (op, reg, rm, gpr) = if let Some(reg) = parse_xmm(dst) {
                    let (rm, gpr) = self.parse_rm(src, false)?;
                    (0x6e, reg, rm, gpr)
                } else if let Some(reg) = parse_xmm(src) {
                    let (rm, gpr) = self.parse_rm(dst, false)?;
                    (0x7e, reg, rm, gpr)
                } else {
                    return Err(RasError::EncodingError("movd requires xmm operand".into()));
                };
                let width = gpr.map_or(32, |reg| reg.width);
                if !matches!(width, 32 | 64) {
                    return Err(RasError::EncodingError(
                        "movd requires 32/64-bit GPR".into(),
                    ));
                }
                code = self.emit(encode_op(Op {
                    legacy: &[0x66],
                    wide: width == 64,
                    opcode: &[0x0f, op],
                    reg,
                    rm,
                    ..Op::default()
                })?);
            }
            "xorps" | "xorpd" => {
                code = self.emit(self.encode_sse_rm(
                    if opcode == "xorpd" { &[0x66] } else { &[] },
                    &[0x0f, 0x57],
                    &inst.operands,
                )?);
            }
            "movzbq" | "movzbl" | "movzx" | "movzwq" | "movzwl" | "movsbl" | "movsbq"
            | "movswl" | "movswq" | "movsbw" => {
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
        Some(hex) => i128::from_str_radix(hex, 16),
        None => digits.parse::<i128>(),
    }
    .map_err(|_| RasError::EncodingError(format!("Invalid immediate: {}", text)))?;
    i64::try_from(if negative { -magnitude } else { magnitude })
        .map_err(|_| RasError::EncodingError(format!("Immediate out of range: {}", text)))
}

pub(crate) fn rip_expression(text: &str) -> Option<(&str, i64)> {
    let symbol = |s: &str| is_symbol(s) || is_got_symbol(s);
    if symbol(text) {
        return Some((text, 0));
    }
    for (index, op) in text
        .char_indices()
        .filter(|(i, c)| *i > 0 && matches!(c, '+' | '-'))
    {
        let left = text[..index].trim();
        let right = text[index + 1..].trim();
        if symbol(left)
            && let Ok(value) = parse_imm(right)
        {
            return Some((
                left,
                if op == '-' {
                    value.checked_neg()?
                } else {
                    value
                },
            ));
        }
        if op == '+'
            && symbol(right)
            && let Ok(value) = parse_imm(left)
        {
            return Some((right, value));
        }
    }
    None
}

fn is_got_symbol(text: &str) -> bool {
    text.strip_suffix("@GOTPCREL").is_some_and(is_symbol)
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
    if matches!(width, 8 | 16 | 32 | 64) {
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

const ALU_STEMS: [&str; 9] = [
    "add", "or", "adc", "sbb", "and", "sub", "xor", "cmp", "test",
];

fn alu_mnemonic(name: &str) -> Option<(&'static str, Option<u8>)> {
    if let Some(stem) = ALU_STEMS.iter().find(|candidate| **candidate == name) {
        return Some((*stem, None));
    }
    [('q', 64), ('l', 32), ('w', 16), ('b', 8)]
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
        "w" => Some(Some(16)),
        "l" => Some(Some(32)),
        "q" => Some(Some(64)),
        _ => None,
    }
}

fn shift_mnemonic(opcode: &str) -> Option<(u8, Option<u8>)> {
    [
        ("rol", 0),
        ("ror", 1),
        ("sar", 7),
        ("shl", 4),
        ("sal", 4),
        ("shr", 5),
    ]
    .into_iter()
    .find_map(|(stem, digit)| sized_stem(opcode, stem).map(|width| (digit, width)))
}

fn condition_code(name: &str) -> Option<u8> {
    Some(match name {
        "o" => 0x0,
        "no" => 0x1,
        "b" | "c" | "nae" => 0x2,
        "ae" | "nb" | "nc" => 0x3,
        "e" | "z" => 0x4,
        "ne" | "nz" => 0x5,
        "be" | "na" => 0x6,
        "a" | "nbe" => 0x7,
        "s" => 0x8,
        "ns" => 0x9,
        "p" | "pe" => 0xa,
        "np" | "po" => 0xb,
        "l" | "nge" => 0xc,
        "ge" | "nl" => 0xd,
        "le" | "ng" => 0xe,
        "g" | "nle" => 0xf,
        _ => return None,
    })
}

fn cmov_mnemonic(opcode: &str) -> Option<(u8, Option<u8>)> {
    let rest = opcode.strip_prefix("cmov")?;
    if let Some(cc) = condition_code(rest) {
        return Some((cc, None));
    }
    let (name, width) = if let Some(name) = rest.strip_suffix('q') {
        (name, 64)
    } else if let Some(name) = rest.strip_suffix('w') {
        (name, 16)
    } else {
        (rest.strip_suffix('l')?, 32)
    };
    condition_code(name).map(|cc| (cc, Some(width)))
}

fn unary_mnemonic(opcode: &str) -> Option<(u8, Option<u8>)> {
    [
        ("inc", 0),
        ("dec", 1),
        ("not", 2),
        ("neg", 3),
        ("mul", 4),
        ("idiv", 7),
        ("div", 6),
    ]
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

    #[test]
    fn cmov_matches_gnu_as() {
        assert_eq!(
            encode_one("cmova", &["%rax", "%rcx"]),
            vec![0x48, 0x0f, 0x47, 0xc8]
        );
        assert_eq!(
            encode_one("cmovbe", &["%rax", "%r14"]),
            vec![0x4c, 0x0f, 0x46, 0xf0]
        );
        assert_eq!(
            encode_one("cmovl", &["%eax", "%ebx"]),
            vec![0x0f, 0x4c, 0xd8]
        );
        assert_eq!(
            encode_one("cmovnle", &["%rdx", "%rsi"]),
            vec![0x48, 0x0f, 0x4f, 0xf2]
        );
        assert_eq!(
            encode_one("cmovzl", &["%eax", "%edx"]),
            vec![0x0f, 0x44, 0xd0]
        );
        assert_eq!(
            encode_one("cmoveq", &["%r8", "%rdx"]),
            vec![0x49, 0x0f, 0x44, 0xd0]
        );
    }

    #[test]
    fn indirect_branches_match_gnu_as() {
        assert_eq!(encode_one("call", &["*%rdx"]), vec![0xff, 0xd2]);
        assert_eq!(encode_one("call", &["*16(%rax)"]), vec![0xff, 0x50, 0x10]);
        assert_eq!(encode_one("call", &["*%r12"]), vec![0x41, 0xff, 0xd4]);
        assert_eq!(encode_one("jmp", &["*%rax"]), vec![0xff, 0xe0]);
    }

    #[test]
    fn imul_movsx_match_gnu_as() {
        assert_eq!(
            encode_one("imull", &["-12(%rbp)", "%eax"]),
            vec![0x0f, 0xaf, 0x45, 0xf4]
        );
        assert_eq!(
            encode_one("imul", &["%ecx", "%eax"]),
            vec![0x0f, 0xaf, 0xc1]
        );
        assert_eq!(
            encode_one("movsbl", &["-36(%rbp)", "%edx"]),
            vec![0x0f, 0xbe, 0x55, 0xdc]
        );
    }

    #[test]
    fn rep_and_nop_match_gnu_as() {
        assert_eq!(encode_one("nop", &[]), vec![0x90]);
        assert_eq!(encode_one("rep", &["movsb"]), vec![0xf3, 0xa4]);
    }

    #[test]
    fn got_load_matches_gnu_as() {
        assert_eq!(
            encode_one("movq", &["foo@GOTPCREL(%rip)", "%rax"]),
            vec![0x48, 0x8b, 0x05, 0, 0, 0, 0]
        );
    }
}
