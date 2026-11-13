//! x86-64 binary instruction encoder

use crate::{
    encoder::traits::{InstructionEncoder, ParsedInstruction, RipFixup, TlsModel},
    error::RasError,
};

mod evex_table;
use evex_table::{BCST, EVEX_TABLE, KDST, MASK, MEMSRC, ROUND, SAE, STORE, ZERO};

/// Which x86 operating mode the encoder emits for. Protected mode is the
/// 32-bit i686 target: no REX prefixes, no RIP-relative addressing, 32-bit
/// address and operand defaults, and eight general registers.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Mode {
    #[default]
    Long,
    Protected,
}

pub struct X86_64Encoder {
    position: usize,
    rip_fixup: Option<RipFixup>,
    mode: Mode,
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
            mode: Mode::Long,
        }
    }

    /// An encoder for 32-bit protected mode, the i686 target.
    pub fn new_32() -> Self {
        Self {
            position: 0,
            rip_fixup: None,
            mode: Mode::Protected,
        }
    }

    fn protected(&self) -> bool {
        self.mode == Mode::Protected
    }

    /// The width a base or index register of a memory operand must have.
    fn address_width(&self) -> u8 {
        if self.protected() { 32 } else { 64 }
    }

    /// The width of a push, a pop and an indirect branch target.
    fn stack_width(&self) -> u8 {
        self.address_width()
    }

    fn encode_op(&self, mut op: Op<'_>) -> Result<Encoded, RasError> {
        op.mode = self.mode;
        encode_op(op)
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
        self.parse_mem_mode(op, false)
    }

    fn parse_mem_mode(&self, op: &str, vsib: bool) -> Result<Mem, RasError> {
        let invalid = || RasError::EncodingError(format!("Invalid memory operand: {}", op));
        let mut text = op.trim();
        let mut broadcast = None;
        if let Some(open) = text.rfind("{1to") {
            let count = text[open + 4..]
                .strip_suffix('}')
                .and_then(|digits| digits.parse::<u8>().ok())
                .filter(|count| matches!(count, 2 | 4 | 8 | 16 | 32 | 64))
                .ok_or_else(invalid)?;
            broadcast = Some(count);
            text = text[..open].trim_end();
        }
        if text.contains('{') {
            return Err(invalid());
        }
        let segment = match [("%fs:", 0x64u8), ("%gs:", 0x65u8)]
            .into_iter()
            .find_map(|(prefix, byte)| text.strip_prefix(prefix).map(|rest| (rest, byte)))
        {
            Some((rest, byte)) => {
                text = rest;
                Some(byte)
            }
            None => None,
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
            segment,
            disp: 0,
            label: None,
            base: None,
            index: None,
            vindex: None,
            broadcast: None,
            rip: false,
            tls: None,
        };
        if !disp_text.is_empty() {
            match parse_imm(disp_text) {
                Ok(value) => mem.disp = value,
                Err(_) => {
                    let (label, addend) = rip_expression(disp_text).ok_or_else(invalid)?;
                    mem.tls = tls_model(label);
                    mem.label = Some(label.to_string());
                    mem.disp = addend;
                }
            }
        }
        mem.broadcast = broadcast;
        if let Some(inner) = inner {
            let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
            let (base, index) = match parts.as_slice() {
                [base] => (*base, None),
                [base, index] => (*base, Some((*index, "1"))),
                [base, index, scale] => (*base, Some((*index, *scale))),
                _ => return Err(invalid()),
            };
            if base.trim_start_matches('%') == "rip" {
                if index.is_some() || segment.is_some() || self.protected() {
                    return Err(invalid());
                }
                mem.rip = true;
            } else if !base.is_empty() {
                let reg = gpr(base)
                    .filter(|g| g.width == self.address_width() && !g.high)
                    .ok_or_else(invalid)?;
                mem.base = Some(reg.num);
            }
            if let Some((index, scale)) = index {
                let scale_bits = match scale {
                    "1" => 0,
                    "2" => 1,
                    "4" => 2,
                    "8" => 3,
                    _ => return Err(invalid()),
                };
                match parse_vreg(index) {
                    Some((number, len)) if vsib => {
                        if mem.rip {
                            return Err(invalid());
                        }
                        mem.vindex = Some((number, scale_bits, len));
                    }
                    Some(_) => {
                        return Err(RasError::EncodingError(format!(
                            "Vector index is not allowed here: {}",
                            op
                        )));
                    }
                    None => {
                        let reg = gpr(index)
                            .filter(|g| g.width == self.address_width() && !g.high && g.num != 4)
                            .ok_or_else(invalid)?;
                        mem.index = Some((reg.num, scale_bits));
                    }
                }
            }
        }
        match mem.tls {
            Some(TlsModel::InitialExec) if !mem.rip => {
                return Err(RasError::EncodingError(
                    "@gottpoff requires %rip-relative addressing".into(),
                ));
            }
            Some(TlsModel::LocalExec) if mem.rip => {
                return Err(RasError::EncodingError(
                    "@tpoff cannot be %rip-relative; use a base register or %fs:".into(),
                ));
            }
            Some(TlsModel::LocalExec) if mem.base.is_none() && mem.segment != Some(0x64) => {
                return Err(RasError::EncodingError(
                    "@tpoff needs a base register or %fs:".into(),
                ));
            }
            Some(TlsModel::SecRel) if mem.rip || mem.base.is_none() || mem.segment.is_some() => {
                return Err(RasError::EncodingError(
                    "@SECREL32 needs a base register and no segment override or %rip".into(),
                ));
            }
            _ => {}
        }
        if mem.label.is_some() && mem.tls.is_none() && !mem.rip {
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

    /// Whether the operands force the AVX-512 encoding: a `zmm`, a register
    /// above 15, an opmask destination, any decorator, or a mnemonic with no
    /// VEX spelling for this operand shape.
    fn needs_evex(&self, name: &str, operands: &[String]) -> bool {
        let is_imm = operands
            .iter()
            .find(|op| rounding_decorator(op).is_none())
            .is_some_and(|op| op.trim().starts_with('$'));
        if !vex_form_exists(name, is_imm) {
            return true;
        }
        if operands
            .last()
            .and_then(|op| split_mask(op).ok())
            .is_some_and(|(base, _, _)| mask_register(base).is_some())
        {
            return true;
        }
        if broadcast_from_general(name).is_some()
            && operands.first().is_some_and(|op| gpr(op).is_some())
        {
            return true;
        }
        if packed_evex(name, is_imm).is_some_and(|entry| entry.form == VexForm::ShiftImm)
            && operands.get(1).is_some_and(|op| self.is_memory(op))
        {
            return true;
        }
        if let Some(entry) = packed_evex(name, is_imm)
            && entry.dst_shift > 0
            && operands.first().is_some_and(|op| self.is_memory(op))
            && operands
                .last()
                .and_then(|op| parse_vreg(op))
                .is_some_and(|(_, len)| len >= 1)
        {
            return true;
        }
        operands.iter().any(|op| {
            let op = op.trim();
            op.contains('{')
                || parse_vreg(op).is_some_and(|(number, len)| number >= 16 || len == 2)
                || self
                    .parse_mem_mode(op, true)
                    .is_ok_and(|mem| mem.vindex.is_some_and(|(number, _, _)| number >= 16))
        })
    }

    fn is_memory(&self, op: &str) -> bool {
        let op = op.trim();
        (op.contains('(') && op.contains(')'))
            || op.starts_with("%fs:")
            || op.starts_with("%gs:")
            || parse_imm(op).is_ok()
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
        if self.protected() && (gpr.num >= 8 || gpr.width == 64) {
            return Err(RasError::EncodingError(format!(
                "Register {} does not exist in 32-bit mode",
                op.trim()
            )));
        }
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
        let (src, dst) = (&operands[0], &operands[1]);
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
                return self.encode_op(Op {
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
        self.encode_op(Op {
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
            return self.encode_op(Op {
                legacy: if width == 16 { &[0x66] } else { &[] },
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
        self.encode_op(Op {
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
        self.encode_op(Op {
            legacy: if width == 16 { &[0x66] } else { &[] },
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
        self.encode_op(Op {
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
        self.encode_op(Op {
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
        self.encode_op(Op {
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
        self.encode_op(Op {
            legacy,
            opcode,
            reg,
            rm,
            ..Op::default()
        })
    }

    /// Parse a vector operand, reporting whether a register operand was 256-bit.
    fn parse_vector_rm(&self, op: &str) -> Result<(Rm, bool), RasError> {
        let op = op.trim();
        if self.is_memory(op) {
            return Ok((Rm::Mem(self.parse_mem(op)?), false));
        }
        parse_vector(op)
            .map(|(number, len256)| (Rm::Reg(number), len256))
            .ok_or_else(|| {
                RasError::EncodingError(format!("Expected vector or memory operand: {}", op))
            })
    }

    /// A VEX move between a vector register and memory or another register.
    /// `load` names the direction that writes the register.
    fn encode_vex_move(
        &self,
        pp: u8,
        load: u8,
        store: u8,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError(
                "VEX move requires 2 operands".into(),
            ));
        }
        let (src, src_wide) = self.parse_vector_rm(&operands[0])?;
        let (dst, dst_wide) = self.parse_vector_rm(&operands[1])?;
        let (opcode, reg, rm) = match (src, dst) {
            (Rm::Reg(src_reg), Rm::Mem(mem)) => (store, src_reg, Rm::Mem(mem)),
            // Only the high register needs an extension bit, and the two-byte
            // prefix carries one for the reg field alone, so between two
            // registers put whichever is high there and pick the direction to
            // match.
            (Rm::Reg(src_reg), Rm::Reg(dst_reg)) if src_reg >= 8 && dst_reg < 8 => {
                (store, src_reg, Rm::Reg(dst_reg))
            }
            (src, Rm::Reg(dst_reg)) => (load, dst_reg, src),
            (Rm::Mem(_), Rm::Mem(_)) => {
                return Err(RasError::EncodingError(
                    "VEX move: memory to memory is not encodable".into(),
                ));
            }
        };
        self.encode_op(Op {
            opcode: &[opcode],
            reg,
            rm,
            vex: Some(Vex {
                pp,
                len256: src_wide || dst_wide,
                ..Vex::default()
            }),
            ..Op::default()
        })
    }

    /// A three-operand VEX arithmetic instruction, written `op src2, src1, dst`.
    fn encode_vex_rrm(
        &self,
        map: u8,
        pp: u8,
        wide: bool,
        opcode: u8,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        if operands.len() != 3 {
            return Err(RasError::EncodingError(
                "VEX arithmetic requires 3 operands".into(),
            ));
        }
        let (rm, rm_wide) = self.parse_vector_rm(&operands[0])?;
        let (first, first_wide) = parse_vector(operands[1].trim()).ok_or_else(|| {
            RasError::EncodingError("VEX arithmetic: first source must be a vector register".into())
        })?;
        let (reg, reg_wide) = parse_vector(operands[2].trim()).ok_or_else(|| {
            RasError::EncodingError("VEX arithmetic: destination must be a vector register".into())
        })?;
        self.encode_op(Op {
            opcode: &[opcode],
            reg,
            rm,
            vex: Some(Vex {
                map,
                pp,
                wide,
                len256: rm_wide || first_wide || reg_wide,
                vvvv: first,
            }),
            ..Op::default()
        })
    }

    /// A BMI2 shift, written `op %count, %src, %dst`. SDM: SHLX is
    /// VEX.LZ.66.0F38.W0/W1 F7 /r, SARX is F3 and SHRX is F2. The count
    /// travels in vvvv, the shifted value in r/m and the result in reg.
    fn encode_bmi2_shift(&self, opcode: &str, operands: &[String]) -> Result<Encoded, RasError> {
        let [count, src, dst] = operands else {
            return Err(RasError::EncodingError(
                "BMI2 shift requires 3 operands".into(),
            ));
        };
        let pp = match opcode.get(..3) {
            Some("shl") => 1,
            Some("sar") => 2,
            _ => 3,
        };
        let wide = opcode.ends_with('q');
        let (rm, rm_gpr) = self.parse_rm(src, false)?;
        let reg = gpr(dst.trim()).ok_or_else(|| {
            RasError::EncodingError("BMI2 shift destination must be a general register".into())
        })?;
        let count = gpr(count.trim()).ok_or_else(|| {
            RasError::EncodingError("BMI2 shift count must be a general register".into())
        })?;
        operand_width(
            Some(if wide { 64 } else { 32 }),
            &[rm_gpr, Some(reg), Some(count)],
        )?;
        self.encode_op(Op {
            opcode: &[0xf7],
            reg: reg.num,
            rm,
            vex: Some(Vex {
                map: 2,
                pp,
                wide,
                vvvv: count.num,
                ..Vex::default()
            }),
            ..Op::default()
        })
    }

    /// An MMX operation. The encodings are the SSE ones without the `66`
    /// prefix, over the eight `%mm` registers, which carry no extension bit.
    fn encode_mmx(&self, name: &str, operands: &[String]) -> Result<Encoded, RasError> {
        let bad = |message: String| RasError::EncodingError(message);
        let mmx_rm = |op: &str| -> Result<Rm, RasError> {
            let op = op.trim();
            match mmx_register(op) {
                Some(number) => Ok(Rm::Reg(number)),
                None if self.is_memory(op) => Ok(Rm::Mem(self.parse_mem(op)?)),
                None => Err(bad(format!("Expected an MMX or memory operand: {}", op))),
            }
        };
        let mmx = |op: &str| -> Result<u8, RasError> {
            mmx_register(op).ok_or_else(|| bad(format!("Expected an MMX register: {}", op)))
        };
        let two = || -> Result<(&String, &String), RasError> {
            match operands {
                [source, destination] => Ok((source, destination)),
                _ => Err(bad(format!("{} requires 2 operands", name))),
            }
        };
        if matches!(name, "movq" | "movd") {
            let (source, destination) = two()?;
            let (source, destination) = (source.trim(), destination.trim());
            let wide = name == "movq";
            let load = mmx_register(destination).is_some();
            let (other, reg) = if load {
                (source, mmx(destination)?)
            } else {
                (destination, mmx(source)?)
            };
            if wide && (mmx_register(other).is_some() || self.is_memory(other)) {
                return self.encode_op(Op {
                    opcode: &[0x0f, if load { 0x6f } else { 0x7f }],
                    reg,
                    rm: mmx_rm(other)?,
                    ..Op::default()
                });
            }
            let (rm, gpr) = self.parse_rm(other, false)?;
            if let Some(gpr) = gpr
                && !matches!(gpr.width, 32 | 64)
            {
                return Err(bad(format!("{}: operand widths do not match", name)));
            }
            return self.encode_op(Op {
                opcode: &[0x0f, if load { 0x6e } else { 0x7e }],
                reg,
                rm,
                wide: gpr.is_some_and(|gpr| gpr.width == 64),
                ..Op::default()
            });
        }
        if let Some((prefix, opcode, to_mmx)) = mmx_convert(name) {
            let (source, destination) = two()?;
            let (reg, rm) = if to_mmx {
                (mmx(destination)?, self.parse_rm(source, true)?.0)
            } else {
                let reg = parse_xmm(destination)
                    .ok_or_else(|| bad(format!("{} writes an xmm register", name)))?;
                (reg, mmx_rm(source)?)
            };
            return self.encode_op(Op {
                legacy: prefix_bytes(prefix),
                opcode: &[0x0f, opcode],
                reg,
                rm,
                ..Op::default()
            });
        }
        match name {
            "pmovmskb" => {
                let (source, destination) = two()?;
                let reg = gpr(destination)
                    .filter(|reg| matches!(reg.width, 32 | 64) && !reg.high)
                    .ok_or_else(|| bad("pmovmskb writes a general register".into()))?;
                self.encode_op(Op {
                    opcode: &[0x0f, 0xd7],
                    reg: reg.num,
                    rm: Rm::Reg(mmx(source)?),
                    ..Op::default()
                })
            }
            "maskmovq" => {
                let (source, destination) = two()?;
                self.encode_op(Op {
                    opcode: &[0x0f, 0xf7],
                    reg: mmx(destination)?,
                    rm: Rm::Reg(mmx(source)?),
                    ..Op::default()
                })
            }
            "movntq" => {
                let (source, destination) = two()?;
                if !self.is_memory(destination) {
                    return Err(bad("movntq stores to memory".into()));
                }
                self.encode_op(Op {
                    opcode: &[0x0f, 0xe7],
                    reg: mmx(source)?,
                    rm: Rm::Mem(self.parse_mem(destination)?),
                    ..Op::default()
                })
            }
            "pextrw" | "pinsrw" => {
                let [immediate, other, vector] = operands else {
                    return Err(bad(format!("{} requires 3 operands", name)));
                };
                let immediate = immediate_byte(immediate)?;
                let insert = name == "pinsrw";
                let (reg, rm) = if insert {
                    let (rm, gpr) = self.parse_rm(other, false)?;
                    if gpr.is_some_and(|gpr| !matches!(gpr.width, 32 | 64) || gpr.high) {
                        return Err(bad("pinsrw reads a general register".into()));
                    }
                    (mmx(vector)?, rm)
                } else {
                    let reg = gpr(vector)
                        .filter(|reg| matches!(reg.width, 32 | 64) && !reg.high)
                        .ok_or_else(|| bad("pextrw writes a general register".into()))?;
                    (reg.num, Rm::Reg(mmx(other)?))
                };
                self.encode_op(Op {
                    opcode: &[0x0f, if insert { 0xc4 } else { 0xc5 }],
                    reg,
                    rm,
                    imm: &[immediate],
                    ..Op::default()
                })
            }
            _ => {
                if let [immediate, destination] = operands
                    && immediate.trim().starts_with('$')
                    && let Some((group, digit)) = packed_shift_imm(name)
                    && mmx_packed(name).is_some()
                {
                    return self.encode_op(Op {
                        opcode: &[0x0f, group],
                        reg: digit,
                        rm: Rm::Reg(mmx(destination)?),
                        imm: &[immediate_byte(immediate)?],
                        ..Op::default()
                    });
                }
                if let [immediate, source, destination] = operands
                    && let Some(entry) = (match name {
                        "pshufw" => Some((1, 0x70)),
                        "palignr" => packed_sse_imm(name).map(|(_, map, opcode)| (map, opcode)),
                        _ => None,
                    })
                {
                    return self.encode_op(Op {
                        opcode: &escape(entry.0, entry.1),
                        reg: mmx(destination)?,
                        rm: mmx_rm(source)?,
                        imm: &[immediate_byte(immediate)?],
                        ..Op::default()
                    });
                }
                let (map, opcode) = mmx_packed(name)
                    .ok_or_else(|| bad(format!("Unsupported MMX operation: {}", name)))?;
                let (source, destination) = two()?;
                self.encode_op(Op {
                    opcode: &escape(map, opcode),
                    reg: mmx(destination)?,
                    rm: mmx_rm(source)?,
                    ..Op::default()
                })
            }
        }
    }

    /// A VSIB gather. The AVX2 spelling carries its mask in a vector register
    /// named first; the AVX-512 spelling drops that operand and takes an
    /// opmask on the destination instead.
    fn encode_gather(&self, name: &str, operands: &[String]) -> Result<Encoded, RasError> {
        let bad = |message: String| RasError::EncodingError(message);
        let (index_wide, element_wide, opcode) =
            gather(name).ok_or_else(|| bad(format!("Unsupported instruction: {}", name)))?;
        let index_shift = u8::from(!index_wide && element_wide);
        let element_shift = u8::from(index_wide && !element_wide);
        let (mask_operand, memory, destination) = match operands {
            [memory, destination] => (None, memory, destination),
            [mask, memory, destination] => (Some(mask), memory, destination),
            _ => return Err(bad(format!("{} requires 2 or 3 operands", name))),
        };
        let memory = self.parse_mem_mode(memory, true)?;
        let (index, _, index_len) = memory
            .vindex
            .ok_or_else(|| bad(format!("{} requires a vector index", name)))?;
        if memory.broadcast.is_some() || memory.label.is_some() {
            return Err(bad(format!("{} takes a plain base and vector index", name)));
        }
        let (destination, mask, zeroing) = split_mask(destination)?;
        if zeroing {
            return Err(bad(format!("{} takes no zeroing mask", name)));
        }
        let (number, element_len) = parse_vreg(destination)
            .ok_or_else(|| bad(format!("{} writes a vector register", name)))?;
        let evex = mask_operand.is_none();
        if evex == (mask == 0) {
            return Err(bad(format!(
                "{} takes either a vector mask operand or an opmask",
                name
            )));
        }
        let vector_mask = mask_operand
            .map(|mask| {
                parse_vector(mask)
                    .ok_or_else(|| bad(format!("{} masks with a vector register", name)))
            })
            .transpose()?;
        let widest = if evex { 2 } else { 1 };
        let length = (0..=widest)
            .find(|length: &u8| {
                index_len == length.saturating_sub(index_shift)
                    && element_len == length.saturating_sub(element_shift)
                    && vector_mask.is_none_or(|(_, wide)| wide == (element_len == 1))
            })
            .ok_or_else(|| bad(format!("{}: operand sizes do not match", name)))?;
        if !evex && (index >= 16 || number >= 16) {
            return Err(bad(format!("{}: registers above 15 need an opmask", name)));
        }
        let rm = Rm::Mem(memory);
        if evex {
            self.encode_op(Op {
                opcode: &[opcode],
                reg: number,
                rm,
                evex: Some(Evex {
                    map: 2,
                    pp: 1,
                    wide: element_wide,
                    ll: length,
                    mask,
                    tuple: Tuple::Fixed(if element_wide { 8 } else { 4 }),
                    ..Evex::default()
                }),
                ..Op::default()
            })
        } else {
            self.encode_op(Op {
                opcode: &[opcode],
                reg: number,
                rm,
                vex: Some(Vex {
                    map: 2,
                    pp: 1,
                    wide: element_wide,
                    len256: length == 1,
                    vvvv: vector_mask.map_or(0, |(number, _)| number),
                }),
                ..Op::default()
            })
        }
    }

    /// Whether a scalar move names a register or a decorator that only the
    /// AVX-512 encoding can carry.
    fn scalar_move_needs_evex(&self, operands: &[String]) -> bool {
        operands.iter().any(|op| {
            let op = op.trim();
            op.contains('{') || parse_vreg(op).is_some_and(|(number, _)| number >= 16)
        })
    }

    /// `vmovss` and `vmovsd` under EVEX, which merge into the destination
    /// under an opmask and reach the whole register file.
    fn encode_evex_scalar_move(
        &self,
        name: &str,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        let bad = |message: String| RasError::EncodingError(message);
        let wide = name == "vmovsd";
        let pp = if wide { 3 } else { 2 };
        let low = |text: &str| -> Result<u8, RasError> {
            match parse_vreg(text) {
                Some((number, 0)) => Ok(number),
                _ => Err(bad(format!("{} takes xmm registers: {}", name, text))),
            }
        };
        let prefix = |mask, zeroing, vvvv| Evex {
            map: 1,
            pp,
            wide,
            vvvv,
            mask,
            zeroing,
            tuple: Tuple::Fixed(if wide { 8 } else { 4 }),
            ..Evex::default()
        };
        match operands {
            [source, destination] => {
                let (destination, mask, zeroing) = split_mask(destination)?;
                if self.is_memory(destination) {
                    if zeroing {
                        return Err(bad("{z} requires a register destination".into()));
                    }
                    return self.encode_op(Op {
                        opcode: &[0x11],
                        reg: low(source)?,
                        rm: Rm::Mem(self.parse_mem(destination)?),
                        evex: Some(prefix(mask, false, 0)),
                        ..Op::default()
                    });
                }
                if !self.is_memory(source) {
                    return Err(bad(format!(
                        "{} between registers names a merge source",
                        name
                    )));
                }
                self.encode_op(Op {
                    opcode: &[0x10],
                    reg: low(destination)?,
                    rm: Rm::Mem(self.parse_mem(source)?),
                    evex: Some(prefix(mask, zeroing, 0)),
                    ..Op::default()
                })
            }
            [source, first, destination] => {
                let (destination, mask, zeroing) = split_mask(destination)?;
                self.encode_op(Op {
                    opcode: &[0x10],
                    reg: low(destination)?,
                    rm: Rm::Reg(low(source)?),
                    evex: Some(prefix(mask, zeroing, low(first)?)),
                    ..Op::default()
                })
            }
            _ => Err(bad(format!("{} requires 2 or 3 operands", name))),
        }
    }

    /// A convert whose mnemonic spells its source width with a trailing `x`
    /// or `y`, which pins the vector length the unsuffixed spelling would
    /// have to read off an operand.
    fn encode_sized_convert(&self, name: &str, operands: &[String]) -> Result<Encoded, RasError> {
        let bad = |message: String| RasError::EncodingError(message);
        let (base, length) =
            sized_convert(name).ok_or_else(|| bad(format!("Unsupported instruction: {}", name)))?;
        let [source, destination] = operands else {
            return Err(bad(format!("{} requires 2 operands", name)));
        };
        if self.needs_evex(base, operands) {
            let entry = packed_evex(base, false)
                .ok_or_else(|| bad(format!("Unsupported instruction: {}", name)))?;
            let narrowed = EvexEntry {
                lens: entry.lens & (1 << length),
                ..*entry
            };
            return self.encode_evex_packed(&narrowed, operands);
        }
        let (map, pp, wide, opcode, _) =
            packed_vex(base).ok_or_else(|| bad(format!("Unsupported instruction: {}", name)))?;
        let (rm, rm_wide) = self.parse_vector_rm(source)?;
        if matches!(rm, Rm::Reg(_)) && rm_wide != (length == 1) {
            return Err(bad(format!("{}: wrong source register width", name)));
        }
        let reg = parse_xmm(destination.trim())
            .ok_or_else(|| bad(format!("{} writes an xmm register", name)))?;
        self.encode_op(Op {
            opcode: &[opcode],
            reg,
            rm,
            vex: Some(Vex {
                map,
                pp,
                wide,
                len256: length == 1,
                vvvv: 0,
            }),
            ..Op::default()
        })
    }

    /// An opmask instruction. These are VEX-encoded and move between opmask
    /// registers, general registers and memory.
    fn encode_kmask(&self, name: &str, operands: &[String]) -> Result<Encoded, RasError> {
        let bad = |message: String| RasError::EncodingError(message);
        let (map, pp, wide, opcode, form) =
            kmask(name).ok_or_else(|| bad(format!("Unsupported instruction: {}", name)))?;
        let kreg = |text: &str| -> Result<u8, RasError> {
            mask_register(text).ok_or_else(|| bad(format!("Expected an opmask register: {}", text)))
        };
        let expected = match form {
            KForm::Unary | KForm::Move => 2,
            KForm::Binary | KForm::Shift => 3,
        };
        if operands.len() != expected {
            return Err(bad(format!("{} requires {} operands", name, expected)));
        }
        let plain = |vvvv, len256, reg, rm, imm: &[u8]| {
            self.encode_op(Op {
                opcode: &[opcode],
                reg,
                rm,
                imm,
                vex: Some(Vex {
                    map,
                    pp,
                    wide,
                    len256,
                    vvvv,
                }),
                ..Op::default()
            })
        };
        match form {
            KForm::Unary => plain(
                0,
                false,
                kreg(&operands[1])?,
                Rm::Reg(kreg(&operands[0])?),
                &[],
            ),
            KForm::Binary => plain(
                kreg(&operands[1])?,
                true,
                kreg(&operands[2])?,
                Rm::Reg(kreg(&operands[0])?),
                &[],
            ),
            KForm::Shift => {
                let count = operands[0]
                    .trim()
                    .strip_prefix('$')
                    .map(parse_imm)
                    .ok_or_else(|| bad(format!("{} takes an immediate count", name)))??;
                let count = u8::try_from(count)
                    .map_err(|_| bad(format!("Shift count out of range: {}", count)))?;
                plain(
                    0,
                    false,
                    kreg(&operands[2])?,
                    Rm::Reg(kreg(&operands[1])?),
                    &[count],
                )
            }
            KForm::Move => {
                let suffix = &name[name.len() - 1..];
                let (source, destination) = (operands[0].trim(), operands[1].trim());
                let (transfer_wide, transfer_pp, width) = kmov_transfer(suffix)
                    .ok_or_else(|| bad(format!("Unsupported instruction: {}", name)))?;
                let general = |register: Gpr, k: u8, load: bool| {
                    if register.width != width || register.high {
                        return Err(bad(format!("{} transfers a {}-bit register", name, width)));
                    }
                    let (reg, rm) = if load {
                        (k, Rm::Reg(register.num))
                    } else {
                        (register.num, Rm::Reg(k))
                    };
                    self.encode_op(Op {
                        opcode: &[opcode + if load { 2 } else { 3 }],
                        reg,
                        rm,
                        vex: Some(Vex {
                            map,
                            pp: transfer_pp,
                            wide: transfer_wide,
                            len256: false,
                            vvvv: 0,
                        }),
                        ..Op::default()
                    })
                };
                match (mask_register(source), mask_register(destination)) {
                    (Some(source), Some(destination)) => {
                        plain(0, false, destination, Rm::Reg(source), &[])
                    }
                    (None, Some(destination)) => match gpr(source) {
                        Some(register) => general(register, destination, true),
                        None if self.is_memory(source) => {
                            plain(0, false, destination, Rm::Mem(self.parse_mem(source)?), &[])
                        }
                        None => Err(bad(format!("{}: unsupported source: {}", name, source))),
                    },
                    (Some(source), None) => match gpr(destination) {
                        Some(register) => general(register, source, false),
                        None if self.is_memory(destination) => self.encode_op(Op {
                            opcode: &[opcode + 1],
                            reg: source,
                            rm: Rm::Mem(self.parse_mem(destination)?),
                            vex: Some(Vex {
                                map,
                                pp,
                                wide,
                                len256: false,
                                vvvv: 0,
                            }),
                            ..Op::default()
                        }),
                        None => Err(bad(format!(
                            "{}: unsupported destination: {}",
                            name, destination
                        ))),
                    },
                    (None, None) => Err(bad(format!("{} names at least one opmask", name))),
                }
            }
        }
    }

    /// A VSIB scatter, which names the scattered register first and the
    /// vector-indexed destination second. The opmask is not optional.
    fn encode_scatter(&self, name: &str, operands: &[String]) -> Result<Encoded, RasError> {
        let bad = |message: String| RasError::EncodingError(message);
        let (index_wide, element_wide, opcode) =
            scatter(name).ok_or_else(|| bad(format!("Unsupported instruction: {}", name)))?;
        let index_shift = u8::from(!index_wide && element_wide);
        let element_shift = u8::from(index_wide && !element_wide);
        let [source, destination] = operands else {
            return Err(bad(format!("{} requires 2 operands", name)));
        };
        let (destination, mask, zeroing) = split_mask(destination)?;
        if zeroing {
            return Err(bad(format!("{} takes no zeroing mask", name)));
        }
        if mask == 0 {
            return Err(bad(format!("{} requires a write mask", name)));
        }
        let memory = self.parse_mem_mode(destination, true)?;
        let (_, _, index_len) = memory
            .vindex
            .ok_or_else(|| bad(format!("{} requires a vector index", name)))?;
        if memory.broadcast.is_some() || memory.label.is_some() {
            return Err(bad(format!("{} takes a plain base and vector index", name)));
        }
        let (number, element_len) =
            parse_vreg(source).ok_or_else(|| bad(format!("{} reads a vector register", name)))?;
        let length = (0..3u8)
            .find(|length| {
                index_len == length.saturating_sub(index_shift)
                    && element_len == length.saturating_sub(element_shift)
            })
            .ok_or_else(|| bad(format!("{}: operand sizes do not match", name)))?;
        self.encode_op(Op {
            opcode: &[opcode],
            reg: number,
            rm: Rm::Mem(memory),
            evex: Some(Evex {
                map: 2,
                pp: 1,
                wide: element_wide,
                ll: length,
                mask,
                tuple: Tuple::Fixed(if element_wide { 8 } else { 4 }),
                ..Evex::default()
            }),
            ..Op::default()
        })
    }

    /// A VSIB gather or scatter prefetch. These name only the memory operand
    /// and are always 512 bits wide; the hint sits in the `reg` field.
    fn encode_vsib_prefetch(&self, name: &str, operands: &[String]) -> Result<Encoded, RasError> {
        let bad = |message: String| RasError::EncodingError(message);
        let (index_wide, element_wide, opcode, digit) =
            vsib_prefetch(name).ok_or_else(|| bad(format!("Unsupported instruction: {}", name)))?;
        let [operand] = operands else {
            return Err(bad(format!("{} requires 1 operand", name)));
        };
        let (operand, mask, zeroing) = split_mask(operand)?;
        if zeroing {
            return Err(bad(format!("{} takes no zeroing mask", name)));
        }
        if mask == 0 {
            return Err(bad(format!("{} requires a write mask", name)));
        }
        let memory = self.parse_mem_mode(operand, true)?;
        let (_, _, index_len) = memory
            .vindex
            .ok_or_else(|| bad(format!("{} requires a vector index", name)))?;
        if memory.broadcast.is_some() || memory.label.is_some() {
            return Err(bad(format!("{} takes a plain base and vector index", name)));
        }
        if index_len != 2 - u8::from(!index_wide && element_wide) {
            return Err(bad(format!("{}: wrong index register width", name)));
        }
        self.encode_op(Op {
            opcode: &[opcode],
            reg: digit,
            rm: Rm::Mem(memory),
            evex: Some(Evex {
                map: 2,
                pp: 1,
                wide: element_wide,
                ll: 2,
                mask,
                tuple: Tuple::Fixed(if element_wide { 8 } else { 4 }),
                ..Evex::default()
            }),
            ..Op::default()
        })
    }

    /// An AVX-512 operation. The table entry fixes the prefix fields and the
    /// operand shape; the operands supply the vector length, the opmask and
    /// the broadcast, rounding and zeroing decorators.
    fn encode_evex_packed(
        &self,
        entry: &EvexEntry,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        let bad = |message: String| RasError::EncodingError(message);
        let mut rest: Vec<&str> = operands.iter().map(|op| op.trim()).collect();
        let mut rounding = None;
        if let Some(index) = rest.iter().position(|op| rounding_decorator(op).is_some()) {
            let (bits, sae_only) = rounding_decorator(rest[index]).unwrap_or((0, false));
            let allowed = if sae_only {
                entry.has(SAE) || entry.has(ROUND)
            } else {
                entry.has(ROUND)
            };
            if !allowed {
                return Err(bad(format!("{} takes no rounding control", entry.name)));
            }
            rounding = Some(bits);
            rest.remove(index);
        }
        let expected = match entry.form {
            VexForm::Rm => 2,
            VexForm::Nds | VexForm::RmImm | VexForm::ExtractImm | VexForm::ShiftImm => 3,
            VexForm::NdsImm => 4,
            VexForm::Is4 => return Err(bad(format!("Unsupported form for {}", entry.name))),
        };
        if rest.len() != expected {
            return Err(bad(format!(
                "{} requires {} operands",
                entry.name, expected
            )));
        }
        let mut imm = Vec::new();
        if entry.takes_imm() {
            imm.push(immediate_byte(rest[0])?);
            rest.remove(0);
        }
        let shift_group = entry.form == VexForm::ShiftImm;
        let nds = matches!(entry.form, VexForm::Nds | VexForm::NdsImm);
        let source = rest[0];
        let destination = *rest.last().unwrap_or(&source);
        let (destination, mask, zeroing) = split_mask(destination)?;
        if mask != 0 && !entry.has(MASK) {
            return Err(bad(format!("{} takes no write mask", entry.name)));
        }
        if zeroing && !entry.has(ZERO) {
            return Err(bad(format!("{} takes no zeroing mask", entry.name)));
        }
        let vector = |text: &str| -> Result<(u8, u8), RasError> {
            parse_vreg(text).ok_or_else(|| bad(format!("Expected a vector register: {}", text)))
        };
        if let Some((opcode, width, wide)) = broadcast_from_general(entry.name)
            && let Some(register) = gpr(source)
        {
            if register.width != width || register.high {
                return Err(bad(format!(
                    "{} reads a {}-bit register",
                    entry.name, width
                )));
            }
            let (number, length) = vector(destination)?;
            return self.encode_op(Op {
                opcode: &[opcode],
                reg: number,
                rm: Rm::Reg(register.num),
                evex: Some(Evex {
                    map: 2,
                    pp: 1,
                    wide,
                    ll: length,
                    mask,
                    zeroing,
                    tuple: entry.tuple,
                    ..Evex::default()
                }),
                ..Op::default()
            });
        }
        let source_is_memory = self.is_memory(source);
        let destination_is_memory = self.is_memory(destination);
        if entry.has(MEMSRC) && !source_is_memory {
            return Err(bad(format!("{} requires a memory source", entry.name)));
        }
        let reversed = entry.has(STORE);
        if reversed && source_is_memory {
            return Err(bad(format!("{} requires a register source", entry.name)));
        }
        let store = (destination_is_memory && entry.form == VexForm::Rm)
            .then(|| evex_store(entry.name))
            .flatten();
        if destination_is_memory
            && entry.form != VexForm::ExtractImm
            && store.is_none()
            && !reversed
        {
            return Err(bad(format!(
                "{} requires a register destination",
                entry.name
            )));
        }
        if destination_is_memory && zeroing {
            return Err(bad("{z} requires a register destination".into()));
        }
        let source_reg = (!source_is_memory).then(|| vector(source)).transpose()?;
        let destination_reg = if entry.has(KDST) || destination_is_memory {
            None
        } else {
            Some(vector(destination)?)
        };
        let first = if shift_group {
            destination_reg
        } else {
            nds.then(|| vector(rest[1])).transpose()?
        };
        let mask_dst = entry
            .has(KDST)
            .then(|| {
                mask_register(destination)
                    .ok_or_else(|| bad(format!("{} writes an opmask register", entry.name)))
            })
            .transpose()?;
        let count = source_is_memory
            .then(|| self.parse_mem(source).ok().and_then(|mem| mem.broadcast))
            .flatten();
        let element: u16 = if entry.wide { 8 } else { 4 };
        let fits = |length: u8| {
            let want_src = length.saturating_sub(entry.src_shift);
            let want_dst = length.saturating_sub(entry.dst_shift);
            count.is_none_or(|count| u16::from(count) == (16u16 << length) / element)
                && source_reg.is_none_or(|(_, len)| len == want_src)
                && destination_reg.is_none_or(|(_, len)| len == want_dst)
                && first.is_none_or(|(_, len)| len == want_dst)
        };
        let length = (0..3)
            .find(|length| entry.lens & (1 << length) != 0 && fits(*length))
            .ok_or_else(|| bad(format!("{}: operand sizes do not match", entry.name)))?;
        let mut rm = if source_is_memory {
            Rm::Mem(self.parse_mem(source)?)
        } else {
            Rm::Reg(source_reg.map_or(0, |(number, _)| number))
        };
        let mut reg = if shift_group {
            entry.digit
        } else {
            destination_reg
                .map(|(number, _)| number)
                .or(mask_dst)
                .unwrap_or(0)
        };
        if entry.form == VexForm::ExtractImm || store.is_some() || reversed {
            reg = source_reg.map_or(0, |(number, _)| number);
            rm = if destination_is_memory {
                Rm::Mem(self.parse_mem(destination)?)
            } else {
                Rm::Reg(vector(destination)?.0)
            };
        }
        if let Rm::Mem(memory) = &rm
            && let Some(count) = memory.broadcast
        {
            if !entry.has(BCST) {
                return Err(bad(format!("{} takes no broadcast", entry.name)));
            }
            let want = (16u16 << length) / if entry.wide { 8 } else { 4 };
            if u16::from(count) != want {
                return Err(bad(format!(
                    "{} broadcasts {} elements here",
                    entry.name, want
                )));
            }
        }
        let broadcast_memory = matches!(&rm, Rm::Mem(memory) if memory.broadcast.is_some());
        if rounding.is_some() && (source_is_memory || destination_is_memory) {
            return Err(bad("Rounding control requires register operands".into()));
        }
        let widest = 7 - entry.lens.leading_zeros() as u8;
        if rounding.is_some() && length != widest {
            return Err(bad(format!(
                "{} takes rounding control only at its widest form",
                entry.name
            )));
        }
        self.encode_op(Op {
            opcode: &[store.unwrap_or(entry.opcode)],
            reg,
            rm,
            imm: &imm,
            evex: Some(Evex {
                map: entry.map,
                pp: entry.pp,
                wide: entry.wide,
                vvvv: first.map_or(0, |(number, _)| number),
                ll: rounding.unwrap_or(length),
                mask,
                zeroing,
                broadcast: rounding.is_some() || broadcast_memory,
                tuple: entry.tuple,
            }),
            ..Op::default()
        })
    }

    /// A VEX shift by an immediate, written `op $count, src, dst`, which keeps
    /// the shifted register in the r/m field and the destination in `vvvv`.
    /// `vmovd` between a general register or memory and the low lane of a
    /// vector register. `6e` writes the vector, `7e` reads it.
    fn encode_vmovd(&self, operands: &[String]) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError("vmovd requires 2 operands".into()));
        }
        let src = operands[0].trim();
        let dst = operands[1].trim();
        let (opcode, reg, rm) = match (parse_vector(src), parse_vector(dst)) {
            (None, Some((reg, _))) => (0x6e, reg, self.parse_rm(src, false)?.0),
            (Some((reg, _)), None) => (0x7e, reg, self.parse_rm(dst, false)?.0),
            _ => {
                return Err(RasError::EncodingError(
                    "vmovd moves between a vector register and a general register or memory".into(),
                ));
            }
        };
        self.encode_op(Op {
            opcode: &[opcode],
            reg,
            rm,
            vex: Some(Vex {
                pp: 1,
                ..Vex::default()
            }),
            ..Op::default()
        })
    }

    fn encode_vex_shift(
        &self,
        group: u8,
        digit: u8,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        if operands.len() != 3 {
            return Err(RasError::EncodingError(
                "VEX shift requires 3 operands".into(),
            ));
        }
        let count = immediate_byte(&operands[0])?;
        let (rm, rm_wide) = parse_vector(operands[1].trim()).ok_or_else(|| {
            RasError::EncodingError("VEX shift: source must be a vector register".into())
        })?;
        let (dst, dst_wide) = parse_vector(operands[2].trim()).ok_or_else(|| {
            RasError::EncodingError("VEX shift: destination must be a vector register".into())
        })?;
        self.encode_op(Op {
            opcode: &[group],
            reg: digit,
            rm: Rm::Reg(rm),
            imm: &[count],
            vex: Some(Vex {
                pp: 1,
                len256: rm_wide || dst_wide,
                vvvv: dst,
                ..Vex::default()
            }),
            ..Op::default()
        })
    }

    /// A VEX-encoded packed operation, in the operand shape `form` names.
    fn encode_vex_packed(
        &self,
        map: u8,
        pp: u8,
        wide: bool,
        opcode: u8,
        form: VexForm,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        let expected = match form {
            VexForm::Rm => 2,
            VexForm::Nds | VexForm::RmImm | VexForm::ExtractImm => 3,
            VexForm::NdsImm | VexForm::Is4 => 4,
            VexForm::ShiftImm => {
                return Err(RasError::EncodingError(
                    "VEX shift groups are encoded separately".into(),
                ));
            }
        };
        if operands.len() != expected {
            return Err(RasError::EncodingError(format!(
                "VEX packed operation requires {} operands",
                expected
            )));
        }
        let takes_imm = matches!(form, VexForm::NdsImm | VexForm::RmImm | VexForm::ExtractImm);
        let mut imm = Vec::new();
        let mut rest = operands;
        if takes_imm {
            imm.push(immediate_byte(&operands[0])?);
            rest = &operands[1..];
        }
        if form == VexForm::Is4 {
            let (mask, _) = parse_vector(operands[0].trim()).ok_or_else(|| {
                RasError::EncodingError("VEX blend mask must be a vector register".into())
            })?;
            imm.push(mask << 4);
            rest = &operands[1..];
        }
        // An extract names the source in `reg` and the destination in `r/m`,
        // the reverse of every other shape here.
        let (source, destination) = if form == VexForm::ExtractImm {
            (&rest[1], &rest[0])
        } else {
            (&rest[0], rest.last().unwrap_or(&rest[0]))
        };
        let (rm, rm_wide) = self.parse_vector_rm(source)?;
        let (reg, reg_wide) = parse_vector(destination.trim()).ok_or_else(|| {
            RasError::EncodingError("VEX packed: destination must be a vector register".into())
        })?;
        let (vvvv, first_wide) = match form {
            VexForm::Nds | VexForm::NdsImm | VexForm::Is4 => parse_vector(rest[1].trim())
                .ok_or_else(|| {
                    RasError::EncodingError(
                        "VEX packed: first source must be a vector register".into(),
                    )
                })?,
            _ => (0, false),
        };
        self.encode_op(Op {
            opcode: &[opcode],
            reg,
            rm,
            imm: &imm,
            vex: Some(Vex {
                map,
                pp,
                wide,
                len256: rm_wide || first_wide || reg_wide,
                vvvv,
            }),
            ..Op::default()
        })
    }

    fn encode_sse_shift(
        &self,
        group: u8,
        digit: u8,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
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
        self.encode_op(Op {
            legacy: &[0x66],
            opcode: &[0x0f, group],
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
            "fsts" => Some((0xd9, 2)),
            "fstps" => Some((0xd9, 3)),
            "fildll" => Some((0xdf, 5)),
            "fadds" => Some((0xd8, 0)),
            "faddl" => Some((0xdc, 0)),
            "fmuls" => Some((0xd8, 1)),
            "fmull" => Some((0xdc, 1)),
            "fsubs" => Some((0xd8, 4)),
            "fsubl" => Some((0xdc, 4)),
            "fdivs" => Some((0xd8, 6)),
            "fdivl" => Some((0xdc, 6)),
            _ => None,
        };
        if let Some((byte, digit)) = memory {
            let [operand] = operands else {
                return Err(RasError::EncodingError(
                    "x87 memory operation requires 1 operand".into(),
                ));
            };
            return self.encode_op(Op {
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

    /// `cvt[t]ss2si` and `cvt[t]sd2si`. `wide` is the destination width where
    /// the mnemonic carries a suffix; without one the register decides.
    fn encode_cvt_to_int(
        &self,
        legacy: u8,
        opcode: u8,
        wide: Option<bool>,
        operands: &[String],
    ) -> Result<Encoded, RasError> {
        if operands.len() != 2 {
            return Err(RasError::EncodingError("cvt requires 2 operands".into()));
        }
        let (src, _) = self.parse_rm(&operands[0], true)?;
        let (dst, dst_gpr) = self.parse_rm(&operands[1], false)?;
        let Rm::Reg(reg) = dst else {
            return Err(RasError::EncodingError(
                "cvt: destination must be a register".into(),
            ));
        };
        let Some(dst_gpr) = dst_gpr.filter(|gpr| matches!(gpr.width, 32 | 64)) else {
            return Err(RasError::EncodingError(
                "cvt: destination must be a 32- or 64-bit register".into(),
            ));
        };
        let wide = match wide {
            Some(wide) if wide != (dst_gpr.width == 64) => {
                return Err(RasError::EncodingError(
                    "cvt: destination width does not match suffix".into(),
                ));
            }
            _ => dst_gpr.width == 64,
        };
        self.encode_op(Op {
            legacy: &[legacy],
            wide,
            opcode: &[0x0f, opcode],
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
        self.encode_op(Op {
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
        self.encode_op(Op {
            wide: width == 64,
            opcode: &[0x8d],
            reg,
            rm: src,
            ..Op::default()
        })
    }

    fn encode_exchange(&self, opcode: &str, operands: &[String]) -> Result<Encoded, RasError> {
        let Some((stem, width)) = exchange_mnemonic(opcode) else {
            return Err(RasError::EncodingError(format!(
                "Unsupported instruction: {opcode}"
            )));
        };
        let [src, dst] = operands else {
            return Err(RasError::EncodingError(format!(
                "{stem} requires 2 operands"
            )));
        };
        let (src_rm, src_gpr) = self.parse_rm(src, false)?;
        let (dst_rm, dst_gpr) = self.parse_rm(dst, false)?;
        let Rm::Reg(reg) = src_rm else {
            return Err(RasError::EncodingError(format!(
                "{stem} requires a register source"
            )));
        };
        let width = operand_width(Some(width), &[src_gpr, dst_gpr])?;
        check_alu_width(width)?;
        let (byte_rex, high) = byte_flags(&[src_gpr, dst_gpr]);
        let (escape, byte_op, wide_op) = match stem {
            "xchg" => (false, 0x86, 0x87),
            "xadd" => (true, 0xc0, 0xc1),
            _ => (true, 0xb0, 0xb1),
        };
        let mut bytes = Vec::new();
        if escape {
            bytes.push(0x0f);
        }
        bytes.push(if width == 8 { byte_op } else { wide_op });
        self.encode_op(Op {
            legacy: if width == 16 { &[0x66] } else { &[] },
            wide: width == 64,
            byte_rex,
            high,
            opcode: &bytes,
            reg,
            rm: dst_rm,
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
            (Some(bits), Some(gpr)) if gpr.width != bits => {
                return Err(RasError::EncodingError(
                    "movzx: source register width does not match mnemonic".into(),
                ));
            }
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
        self.encode_op(Op {
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
        if self.protected() && !protected_mode_supported(&inst.opcode.to_lowercase()) {
            return Err(RasError::EncodingError(format!(
                "{} is not implemented by the 32-bit (i686) encoder",
                inst.opcode
            )));
        }
        if let Some(suffix) = inst
            .operands
            .iter()
            .find_map(|op| unsupported_tls_suffix(op))
        {
            return Err(unsupported_tls_error(suffix));
        }
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
            let enc = self.encode_op(Op {
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
        if !inst.operands.is_empty()
            && matches!(
                opcode.as_str(),
                "fld1"
                    | "endbr64"
                    | "leave"
                    | "cwtl"
                    | "cltq"
                    | "cqto"
                    | "cqo"
                    | "cdq"
                    | "cltd"
                    | "ud2"
                    | "hlt"
                    | "cli"
                    | "sti"
                    | "int3"
                    | "cpuid"
                    | "syscall"
                    | "mfence"
                    | "rdtsc"
            )
        {
            return Err(RasError::EncodingError(format!(
                "{opcode} takes no operands"
            )));
        }

        match opcode.as_str() {
            "emms" | "femms" => {
                if !inst.operands.is_empty() {
                    return Err(RasError::EncodingError(format!(
                        "{} takes no operands",
                        opcode
                    )));
                }
                code.extend_from_slice(if opcode == "emms" {
                    &[0x0f, 0x77]
                } else {
                    &[0x0f, 0x0e]
                });
            }
            name if mmx_convert(name).is_some()
                || (mmx_mnemonic(name)
                    && inst.operands.iter().any(|op| mmx_register(op).is_some())) =>
            {
                let enc = self.encode_mmx(name, &inst.operands)?;
                code = self.emit(enc);
            }
            name if gather(name).is_some() => {
                let enc = self.encode_gather(name, &inst.operands)?;
                code = self.emit(enc);
            }
            name if scatter(name).is_some() => {
                let enc = self.encode_scatter(name, &inst.operands)?;
                code = self.emit(enc);
            }
            name if vsib_prefetch(name).is_some() => {
                let enc = self.encode_vsib_prefetch(name, &inst.operands)?;
                code = self.emit(enc);
            }
            name if kmask(name).is_some() => {
                let enc = self.encode_kmask(name, &inst.operands)?;
                code = self.emit(enc);
            }
            name if sized_convert(name).is_some() => {
                let enc = self.encode_sized_convert(name, &inst.operands)?;
                code = self.emit(enc);
            }
            name if is_evex_mnemonic(name) && self.needs_evex(name, &inst.operands) => {
                let is_imm = inst
                    .operands
                    .iter()
                    .find(|op| rounding_decorator(op).is_none())
                    .is_some_and(|op| op.trim().starts_with('$'));
                let entry = packed_evex(name, is_imm).ok_or_else(|| {
                    RasError::EncodingError(format!("Unsupported form of {}", name))
                })?;
                let enc = self.encode_evex_packed(entry, &inst.operands)?;
                code = self.emit(enc);
            }
            "movq" | "mov" => {
                if inst.operands.len() != 2 {
                    return Err(RasError::EncodingError(
                        "mov requires 2 operands".to_string(),
                    ));
                }
                let a = &inst.operands[0];
                let b = &inst.operands[1];
                if opcode == "mov" && (parse_xmm(a).is_some() || parse_xmm(b).is_some()) {
                    return Err(RasError::EncodingError(
                        "mov: xmm operands require movq or movd".into(),
                    ));
                }

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
                        code = self.emit(self.encode_op(Op {
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
                        code = self.emit(self.encode_op(Op {
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
            "movabsq" | "movabs" => {
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
                let (rm, gpr) = self.parse_rm(target, false)?;
                let width = self.stack_width();
                if gpr.is_some_and(|gpr| gpr.width != width) {
                    return Err(RasError::EncodingError(format!(
                        "{opcode}: indirect target must be a {width}-bit register"
                    )));
                }
                let digit = if opcode == "call" { 2 } else { 4 };
                let enc = self.encode_op(Op {
                    opcode: &[0xff],
                    reg: digit,
                    rm,
                    ..Op::default()
                })?;
                code = self.emit(enc);
            }
            "flds" | "fldl" | "fldt" | "fsts" | "fstl" | "fstps" | "fstpl" | "fstpt" | "fildl"
            | "fildq" | "fildll" | "fistpl" | "fistpq" | "fldcw" | "fnstcw" | "fadds" | "faddl"
            | "fmuls" | "fmull" | "fsubs" | "fsubl" | "fdivs" | "fdivl" | "fld" | "fxch"
            | "fstp" | "fmul" | "fsub" | "fmulp" | "fdivp" | "fsubrp" | "fcomi" | "fcomip"
            | "fucomip" => {
                code = self.emit(self.encode_x87(&opcode, &inst.operands)?);
            }
            "fldz" | "fchs" | "fabs" | "fsqrt" => {
                if !inst.operands.is_empty() {
                    return Err(RasError::EncodingError(
                        "x87 constant/sign operation takes no operands".into(),
                    ));
                }
                let second = match opcode.as_str() {
                    "fldz" => 0xee,
                    "fchs" => 0xe0,
                    "fabs" => 0xe1,
                    _ => 0xfa,
                };
                code.extend_from_slice(&[0xd9, second]);
            }
            "fld1" => code.extend_from_slice(&[0xd9, 0xe8]),
            "bswap" | "bswapl" | "bswapq" => {
                let [operand] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError("bswap requires 1 operand".into()));
                };
                let reg = gpr(operand)
                    .filter(|g| matches!(g.width, 32 | 64))
                    .filter(|g| !self.protected() || (g.width == 32 && g.num < 8))
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
            "btl" | "btq" | "btsl" | "btsq" | "btrl" | "btrq" | "btcl" | "btcq" | "bsfl"
            | "bsfq" | "bsrl" | "bsrq" => {
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "bit operation requires 2 operands".into(),
                    ));
                };
                let width = if opcode.ends_with('q') { 64 } else { 32 };
                let scan = opcode.starts_with("bs");
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
                        let op = if opcode.starts_with("bsf") {
                            0xbc
                        } else if scan {
                            0xbd
                        } else {
                            0xa3 + (digit - 4) * 8
                        };
                        (op, reg.num)
                    };
                code = self.emit(self.encode_op(Op {
                    wide: width == 64,
                    opcode: &[0x0f, op],
                    reg,
                    rm,
                    imm: &imm,
                    ..Op::default()
                })?);
            }
            "popcntw" | "popcntl" | "popcntq" | "lzcntw" | "lzcntl" | "lzcntq" | "tzcntw"
            | "tzcntl" | "tzcntq" => {
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "bit count requires 2 operands".into(),
                    ));
                };
                let count_opcode = if opcode.starts_with("popcnt") {
                    0xb8
                } else if opcode.starts_with("lzcnt") {
                    0xbd
                } else {
                    0xbc
                };
                let width = if opcode.ends_with('q') {
                    64
                } else if opcode.ends_with('w') {
                    16
                } else {
                    32
                };
                let (rm, rm_gpr) = self.parse_rm(src, false)?;
                let reg = gpr(dst).ok_or_else(|| {
                    RasError::EncodingError("bit count requires register destination".into())
                })?;
                operand_width(Some(width), &[rm_gpr, Some(reg)])?;
                code = self.emit(self.encode_op(Op {
                    legacy: if width == 16 { &[0x66, 0xf3] } else { &[0xf3] },
                    wide: width == 64,
                    opcode: &[0x0f, count_opcode],
                    reg: reg.num,
                    rm,
                    ..Op::default()
                })?);
            }
            "xchgb" | "xchgw" | "xchgl" | "xchgq" | "xaddb" | "xaddw" | "xaddl" | "xaddq"
            | "cmpxchgb" | "cmpxchgw" | "cmpxchgl" | "cmpxchgq" => {
                code = self.emit(self.encode_exchange(&opcode, &inst.operands)?);
            }
            "mfence" => code.extend_from_slice(&[0x0f, 0xae, 0xf0]),
            "rdtsc" => code.extend_from_slice(&[0x0f, 0x31]),
            "ret" => match inst.operands.as_slice() {
                [] => code.push(0xC3),
                [imm] => {
                    let value = imm.strip_prefix('$').ok_or_else(|| {
                        RasError::EncodingError("ret requires an immediate".into())
                    })?;
                    let value = u16::try_from(parse_imm(value)?).map_err(|_| {
                        RasError::EncodingError("ret immediate out of range".into())
                    })?;
                    code.push(0xC2);
                    code.extend_from_slice(&value.to_le_bytes());
                }
                _ => {
                    return Err(RasError::EncodingError(
                        "ret takes at most one operand".into(),
                    ));
                }
            },
            "int" => {
                let [operand] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError("int requires 1 operand".into()));
                };
                code.push(0xcd);
                code.push(immediate_byte(operand)?);
            }
            "endbr64" => code.extend_from_slice(&[0xf3, 0x0f, 0x1e, 0xfa]),
            "leave" => code.push(0xC9),
            "cvttsd2si" | "cvttsd2sil" | "cvttsd2siq" | "cvttss2si" | "cvttss2sil"
            | "cvttss2siq" | "cvtsd2si" | "cvtsd2sil" | "cvtsd2siq" | "cvtss2si" | "cvtss2sil"
            | "cvtss2siq" => {
                let single = opcode.starts_with("cvttss") || opcode.starts_with("cvtss");
                let legacy = if single { 0xf3 } else { 0xf2 };
                let truncate = opcode.starts_with("cvtt");
                let wide = match opcode.as_bytes().last() {
                    Some(b'l') => Some(false),
                    Some(b'q') => Some(true),
                    _ => None,
                };
                let enc = self.encode_cvt_to_int(
                    legacy,
                    if truncate { 0x2c } else { 0x2d },
                    wide,
                    &inst.operands,
                )?;
                code = self.emit(enc);
            }
            "cvtsi2sdl" | "cvtsi2sdq" | "cvtsi2ssl" | "cvtsi2ssq" => {
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
                code = self.emit(self.encode_op(Op {
                    legacy: &[if opcode.starts_with("cvtsi2ss") {
                        0xf3
                    } else {
                        0xf2
                    }],
                    wide,
                    opcode: &[0x0f, 0x2a],
                    reg,
                    rm,
                    ..Op::default()
                })?);
            }
            name if packed_sse(name).is_some() => {
                let is_imm = inst
                    .operands
                    .first()
                    .is_some_and(|op| op.trim().starts_with('$'));
                let enc = match packed_shift_imm(name).filter(|_| is_imm) {
                    Some((group, digit)) => self.encode_sse_shift(group, digit, &inst.operands)?,
                    None => {
                        let (prefix, map, op) = packed_sse(name).unwrap_or((0, 1, 0));
                        if matches!(name, "lddqu" | "movntdqa")
                            && !inst.operands.first().is_some_and(|op| self.is_memory(op))
                        {
                            return Err(RasError::EncodingError(format!(
                                "{} requires a memory source",
                                name
                            )));
                        }
                        let legacy = prefix_bytes(prefix);
                        self.encode_sse_rm(legacy, &escape(map, op), &inst.operands)?
                    }
                };
                code = self.emit(enc);
            }
            name if packed_sse_imm(name).is_some() && inst.operands.len() == 3 => {
                let (prefix, map, op) = packed_sse_imm(name).unwrap_or((0, 1, 0));
                let [imm, src, dst] = inst.operands.as_slice() else {
                    unreachable!()
                };
                let imm = immediate_byte(imm)?;
                let (rm, _) = self.parse_rm(src, true)?;
                let reg = parse_xmm(dst).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires an xmm destination", name))
                })?;
                code = self.emit(self.encode_op(Op {
                    legacy: prefix_bytes(prefix),
                    opcode: &escape(map, op),
                    reg,
                    rm,
                    imm: &[imm],
                    ..Op::default()
                })?);
            }
            name if packed_blendv(name).is_some() => {
                let op = packed_blendv(name).unwrap_or(0);
                let [mask, src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(format!(
                        "{} requires 3 operands",
                        name
                    )));
                };
                if parse_xmm(mask) != Some(0) {
                    return Err(RasError::EncodingError(format!(
                        "{} takes its mask in %xmm0",
                        name
                    )));
                }
                let (rm, _) = self.parse_rm(src, true)?;
                let reg = parse_xmm(dst).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires an xmm destination", name))
                })?;
                code = self.emit(self.encode_op(Op {
                    legacy: &[0x66],
                    opcode: &escape(2, op),
                    reg,
                    rm,
                    ..Op::default()
                })?);
            }
            name if packed_store(name).is_some() => {
                let (prefix, op) = packed_store(name).unwrap_or((0, 0));
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(format!(
                        "{} requires 2 operands",
                        name
                    )));
                };
                let reg = parse_xmm(src).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires an xmm source", name))
                })?;
                if !self.is_memory(dst) {
                    return Err(RasError::EncodingError(format!(
                        "{} requires a memory destination",
                        name
                    )));
                }
                let (rm, _) = self.parse_rm(dst, true)?;
                code = self.emit(self.encode_op(Op {
                    legacy: prefix_bytes(prefix),
                    opcode: &escape(1, op),
                    reg,
                    rm,
                    ..Op::default()
                })?);
            }
            name if packed_mask_move(name).is_some() => {
                let (prefix, op) = packed_mask_move(name).unwrap_or((0, 0));
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(format!(
                        "{} requires 2 operands",
                        name
                    )));
                };
                let rm = parse_xmm(src).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires an xmm source", name))
                })?;
                let reg = gpr(dst.trim()).filter(|g| g.width == 32).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires a 32-bit destination", name))
                })?;
                code = self.emit(self.encode_op(Op {
                    legacy: prefix_bytes(prefix),
                    opcode: &escape(1, op),
                    reg: reg.num,
                    rm: Rm::Reg(rm),
                    ..Op::default()
                })?);
            }
            name if lane_transfer(name).is_some() => {
                let (op, wide, width) = lane_transfer(name).unwrap_or((0, false, 32));
                let [imm, src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "lane transfer requires 3 operands".into(),
                    ));
                };
                let imm = immediate_byte(imm)?;
                let extract = name.starts_with("pextr") || name == "extractps";
                let (xmm, other) = if extract { (src, dst) } else { (dst, src) };
                let reg = parse_xmm(xmm).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires an xmm operand", name))
                })?;
                let (rm, gpr) = self.parse_rm(other, false)?;
                if gpr.is_some() {
                    operand_width(Some(width), &[gpr])?;
                }
                code = self.emit(self.encode_op(Op {
                    legacy: &[0x66],
                    wide,
                    opcode: &[0x0f, 0x3a, op],
                    reg,
                    rm,
                    imm: &[imm],
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
            "movhps" | "movlps" | "movhpd" => {
                let (prefix, load) = match opcode.as_str() {
                    "movlps" => (0u8, 0x12u8),
                    "movhpd" => (0x66, 0x16),
                    _ => (0, 0x16),
                };
                if inst.operands.len() != 2
                    || inst.operands.iter().filter(|op| self.is_memory(op)).count() != 1
                {
                    return Err(RasError::EncodingError(format!(
                        "{} requires memory and xmm operands",
                        opcode
                    )));
                }
                code = self.emit(self.encode_sse_move(
                    prefix_bytes(prefix),
                    &[0x0f, load],
                    &[0x0f, load + 1],
                    &inst.operands,
                )?);
            }
            "comisd" | "ucomisd" => {
                let op = if opcode.as_str() == "comisd" {
                    0x2f
                } else {
                    0x2e
                };
                code = self.emit(self.encode_sse_rm(&[0x66], &[0x0f, op], &inst.operands)?);
            }
            "comiss" | "ucomiss" => {
                let op = if opcode.as_str() == "comiss" {
                    0x2f
                } else {
                    0x2e
                };
                code = self.emit(self.encode_sse_rm(&[], &[0x0f, op], &inst.operands)?);
            }
            "pshufd" | "pshuflw" | "pshufhw" | "shufpd" | "shufps" => {
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
                code = self.emit(self.encode_op(Op {
                    legacy: match opcode.as_str() {
                        "pshuflw" => &[0xf2],
                        "pshufhw" => &[0xf3],
                        "shufps" => &[],
                        _ => &[0x66],
                    },
                    opcode: &[
                        0x0f,
                        if matches!(opcode.as_str(), "shufpd" | "shufps") {
                            0xc6
                        } else {
                            0x70
                        },
                    ],
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
            "movhlps" | "movlhps" => {
                if inst.operands.iter().any(|op| parse_xmm(op).is_none()) {
                    return Err(RasError::EncodingError(format!(
                        "{} requires xmm registers",
                        opcode
                    )));
                }
                let op = if opcode == "movhlps" { 0x12 } else { 0x16 };
                code = self.emit(self.encode_sse_rm(&[], &[0x0f, op], &inst.operands)?);
            }
            "cvtss2sd" | "cvtsd2ss" | "minsd" => {
                let (prefix, op) = match opcode.as_str() {
                    "cvtss2sd" => (0xf3, 0x5a),
                    "cvtsd2ss" => (0xf2, 0x5a),
                    _ => (0xf2, 0x5d),
                };
                code = self.emit(self.encode_sse_rm(&[prefix], &[0x0f, op], &inst.operands)?);
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
                // Extracting to memory is only encodable in the 0f 3a map,
                // while the register form keeps the older 0f c5 opcode.
                let to_memory = opcode == "pextrw" && self.is_memory(dst);
                let (reg, rm) = if to_memory {
                    let reg = parse_xmm(src).ok_or_else(|| {
                        RasError::EncodingError("pextrw requires xmm source".into())
                    })?;
                    (reg, self.parse_rm(dst, true)?.0)
                } else if opcode == "pextrw" {
                    let reg = gpr(dst)
                        .filter(|g| matches!(g.width, 32 | 64))
                        .ok_or_else(|| {
                            RasError::EncodingError(
                                "pextrw requires a 32- or 64-bit destination".into(),
                            )
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
                let escaped: &[u8] = if to_memory {
                    &[0x0f, 0x3a, 0x15]
                } else if opcode == "pextrw" {
                    &[0x0f, 0xc5]
                } else {
                    &[0x0f, 0xc4]
                };
                code = self.emit(self.encode_op(Op {
                    legacy: &[0x66],
                    opcode: escaped,
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
            "movupd" => {
                let enc =
                    self.encode_sse_move(&[0x66], &[0x0f, 0x10], &[0x0f, 0x11], &inst.operands)?;
                code = self.emit(enc);
            }
            "sfence" | "lfence" | "pause" => {
                if !inst.operands.is_empty() {
                    return Err(RasError::EncodingError(format!(
                        "{} takes no operands",
                        opcode
                    )));
                }
                code.extend_from_slice(match opcode.as_str() {
                    "sfence" => &[0x0f, 0xae, 0xf8],
                    "lfence" => &[0x0f, 0xae, 0xe8],
                    _ => &[0xf3, 0x90],
                });
            }
            "ldmxcsr" | "stmxcsr" => {
                let [target] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(format!(
                        "{} requires 1 operand",
                        opcode
                    )));
                };
                if !self.is_memory(target) {
                    return Err(RasError::EncodingError(format!(
                        "{} requires a memory operand",
                        opcode
                    )));
                }
                code = self.emit(self.encode_op(Op {
                    opcode: &[0x0f, 0xae],
                    reg: if opcode == "ldmxcsr" { 2 } else { 3 },
                    rm: self.parse_rm(target, true)?.0,
                    ..Op::default()
                })?);
            }
            "prefetchnta" | "prefetcht0" | "prefetcht1" | "prefetcht2" => {
                let digit = match opcode.as_str() {
                    "prefetcht0" => 1,
                    "prefetcht1" => 2,
                    "prefetcht2" => 3,
                    _ => 0,
                };
                let [target] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(format!(
                        "{} requires 1 operand",
                        opcode
                    )));
                };
                if !self.is_memory(target) {
                    return Err(RasError::EncodingError(format!(
                        "{} requires a memory operand",
                        opcode
                    )));
                }
                code = self.emit(self.encode_op(Op {
                    opcode: &[0x0f, 0x18],
                    reg: digit,
                    rm: self.parse_rm(target, true)?.0,
                    ..Op::default()
                })?);
            }
            "movntil" | "movntiq" => {
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError("movnti requires 2 operands".into()));
                };
                let wide = opcode.ends_with('q');
                let reg = gpr(src.trim())
                    .filter(|g| g.width == if wide { 64 } else { 32 })
                    .ok_or_else(|| {
                        RasError::EncodingError("movnti source width disagrees".into())
                    })?;
                if !self.is_memory(dst) {
                    return Err(RasError::EncodingError(
                        "movnti requires a memory destination".into(),
                    ));
                }
                code = self.emit(self.encode_op(Op {
                    wide,
                    opcode: &[0x0f, 0xc3],
                    reg: reg.num,
                    rm: self.parse_rm(dst, true)?.0,
                    ..Op::default()
                })?);
            }
            // SSE4.2 CRC32: the suffix names the source width, the result is
            // always a 32- or 64-bit general register.
            "crc32b" | "crc32w" | "crc32l" | "crc32q" => {
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError("crc32 requires 2 operands".into()));
                };
                let source = match opcode.as_str() {
                    "crc32b" => 8,
                    "crc32w" => 16,
                    "crc32l" => 32,
                    _ => 64,
                };
                let wide = source == 64;
                let reg = gpr(dst.trim())
                    .filter(|g| g.width == if wide { 64 } else { 32 })
                    .ok_or_else(|| {
                        RasError::EncodingError("crc32 destination width disagrees".into())
                    })?;
                let (rm, source_gpr) = self.parse_rm(src, false)?;
                operand_width(Some(source), &[source_gpr])?;
                let (byte_rex, high) = byte_flags(&[source_gpr]);
                let legacy: &[u8] = if source == 16 { &[0x66, 0xf2] } else { &[0xf2] };
                code = self.emit(self.encode_op(Op {
                    legacy,
                    wide,
                    byte_rex,
                    high,
                    opcode: &[0x0f, 0x38, 0xf0 + u8::from(source != 8)],
                    reg: reg.num,
                    rm,
                    ..Op::default()
                })?);
            }
            "movdqa" => {
                let enc =
                    self.encode_sse_move(&[0x66], &[0x0f, 0x6f], &[0x0f, 0x7f], &inst.operands)?;
                code = self.emit(enc);
            }
            "addsd" | "addss" | "subsd" | "subss" | "mulsd" | "mulss" | "divsd" | "divss"
            | "sqrtsd" | "sqrtss" | "minss" | "maxsd" | "maxss" => {
                let legacy = match opcode.as_str() {
                    "addss" | "subss" | "mulss" | "divss" | "sqrtss" | "minss" | "maxss" => 0xf3,
                    _ => 0xf2,
                };
                let op = match opcode.as_str() {
                    "addsd" | "addss" => 0x58,
                    "mulsd" | "mulss" => 0x59,
                    "sqrtsd" | "sqrtss" => 0x51,
                    "minss" => 0x5d,
                    "maxsd" | "maxss" => 0x5f,
                    "subsd" | "subss" => 0x5c,
                    _ => 0x5e,
                };
                let enc = self.encode_sse_rm(&[legacy], &[0x0f, op], &inst.operands)?;
                code = self.emit(enc);
            }
            "pslldq" | "psrldq" => {
                let (group, digit) = packed_shift_imm(opcode.as_str())
                    .ok_or_else(|| RasError::EncodingError(format!("Unknown shift: {}", opcode)))?;
                let enc = self.encode_sse_shift(group, digit, &inst.operands)?;
                code = self.emit(enc);
            }
            "cwtl" => code.push(0x98),
            "ud2" => code.extend_from_slice(&[0x0f, 0x0b]),
            // The instructions that stop, wait or trap, which is what a naked body
            // and a startup stub reach for when there is nowhere left to go.
            "hlt" => code.push(0xf4),
            "cli" => code.push(0xfa),
            "sti" => code.push(0xfb),
            "int3" => code.push(0xcc),
            "cpuid" => code.extend_from_slice(&[0x0f, 0xa2]),
            "cltq" => code.extend_from_slice(&[0x48, 0x98]),
            "movslq" => {
                let enc = self.encode_movsxd(&inst.operands)?;
                code = self.emit(enc);
            }
            "pushq" | "pushl" | "push" | "popq" | "popl" | "pop" => {
                let [operand] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "push/pop requires 1 operand".into(),
                    ));
                };
                let width = match (opcode.as_bytes().last(), self.protected()) {
                    (Some(b'q'), false) => 64,
                    (Some(b'q'), true) | (Some(b'l'), false) => {
                        return Err(RasError::EncodingError(format!(
                            "{opcode} is not available in this mode"
                        )));
                    }
                    _ => self.stack_width(),
                };
                let push = opcode.starts_with("push");
                if let Some(imm) = operand.strip_prefix('$').filter(|_| push) {
                    let imm = parse_imm(imm)?;
                    let small = i8::try_from(imm).is_ok();
                    code.push(if small { 0x6a } else { 0x68 });
                    code.extend(imm_bytes(imm, if small { 8 } else { width })?);
                } else {
                    let (rm, reg) = self.parse_rm(operand, false)?;
                    operand_width(Some(width), &[reg])?;
                    if let Rm::Reg(reg) = rm {
                        if reg >= 8 {
                            code.push(0x41);
                        }
                        code.push((if push { 0x50 } else { 0x58 }) | (reg & 7));
                    } else {
                        code = self.emit(self.encode_op(Op {
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
                    code = self.emit(self.encode_op(Op {
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
                let enc = self.encode_op(Op {
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
                let lockable = exchange_mnemonic(&nested_opcode).is_some()
                    || alu_mnemonic(&nested_opcode)
                        .is_some_and(|(stem, _)| !matches!(stem, "cmp" | "test"));
                if opcode == "lock"
                    && (!lockable || !operands.last().is_some_and(|op| self.is_memory(op)))
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
            name if packed_vex(name).is_some()
                || vex_shift_imm(name).is_some()
                || packed_vex_permil(name, false).is_some() =>
            {
                let is_imm = inst
                    .operands
                    .first()
                    .is_some_and(|op| op.trim().starts_with('$'));
                let shift = vex_shift_imm(name).filter(|_| is_imm && inst.operands.len() == 3);
                let enc = match shift {
                    Some((group, digit)) => self.encode_vex_shift(group, digit, &inst.operands)?,
                    None => {
                        let (map, pp, wide, opcode, form) = packed_vex_permil(name, is_imm)
                            .or_else(|| packed_vex(name))
                            .ok_or_else(|| {
                                RasError::EncodingError(format!(
                                    "Unsupported instruction: {}",
                                    name
                                ))
                            })?;
                        // A variable shift takes its count from the low
                        // 128 bits, so the count operand is never a ymm.
                        if matches!(
                            name,
                            "vpsllw"
                                | "vpslld"
                                | "vpsllq"
                                | "vpsrlw"
                                | "vpsrld"
                                | "vpsrlq"
                                | "vpsraw"
                                | "vpsrad"
                        ) && inst
                            .operands
                            .first()
                            .and_then(|op| parse_vector(op.trim()))
                            .is_some_and(|(_, wide)| wide)
                        {
                            return Err(RasError::EncodingError(format!(
                                "{} takes its count in an xmm register",
                                name
                            )));
                        }
                        if matches!(name, "vcvtpd2ps" | "vcvtpd2dq" | "vcvttpd2dq")
                            && inst.operands.first().is_some_and(|op| self.is_memory(op))
                        {
                            return Err(RasError::EncodingError(format!(
                                "{} needs an explicit operand size for a memory source",
                                name
                            )));
                        }
                        if matches!(name, "vlddqu" | "vmovntdqa")
                            && !inst.operands.first().is_some_and(|op| self.is_memory(op))
                        {
                            return Err(RasError::EncodingError(format!(
                                "{} requires a memory source",
                                name
                            )));
                        }
                        self.encode_vex_packed(map, pp, wide, opcode, form, &inst.operands)?
                    }
                };
                code = self.emit(enc);
            }
            // VEX lane transfers. Extracts leave vvvv unused; inserts take
            // the untouched lanes from a second source register.
            name if name.strip_prefix('v').is_some_and(|stem| {
                lane_transfer(stem).is_some() || matches!(stem, "pextrw" | "pinsrw")
            }) =>
            {
                let stem = name.strip_prefix('v').unwrap_or(name);
                let insert = stem.starts_with("pinsr");
                let (op, wide, width) = match stem {
                    "pextrw" | "pinsrw" => (if insert { 0xc4 } else { 0xc5 }, false, 32),
                    _ => lane_transfer(stem).unwrap_or((0, false, 32)),
                };
                let expected = if insert { 4 } else { 3 };
                if inst.operands.len() != expected {
                    return Err(RasError::EncodingError(format!(
                        "{} requires {} operands",
                        name, expected
                    )));
                }
                let imm = immediate_byte(&inst.operands[0])?;
                let other = &inst.operands[1];
                let vector = &inst.operands[if insert { 3 } else { 2 }];
                let (reg, rm, vvvv) = if insert {
                    let reg = parse_xmm(vector).ok_or_else(|| {
                        RasError::EncodingError(format!("{} requires an xmm destination", name))
                    })?;
                    let (rm, gpr) = self.parse_rm(other, false)?;
                    if gpr.is_some() {
                        operand_width(Some(width), &[gpr])?;
                    }
                    let vvvv = parse_xmm(&inst.operands[2]).ok_or_else(|| {
                        RasError::EncodingError(format!("{} requires an xmm source", name))
                    })?;
                    (reg, rm, vvvv)
                } else {
                    let reg = parse_xmm(&inst.operands[1]).ok_or_else(|| {
                        RasError::EncodingError(format!("{} requires an xmm source", name))
                    })?;
                    let (rm, gpr) = self.parse_rm(&inst.operands[2], false)?;
                    if gpr.is_some() {
                        operand_width(Some(width), &[gpr])?;
                    }
                    (reg, rm, 0)
                };
                let _ = other;
                // The word forms keep the older one-byte opcodes in the 0f
                // map; extracting a word to memory is only in 0f 3a.
                let map = if stem == "pinsrw" || (stem == "pextrw" && matches!(rm, Rm::Reg(_))) {
                    1
                } else {
                    3
                };
                let op = if stem == "pextrw" && map == 3 {
                    0x15
                } else {
                    op
                };
                // `0f c5` reads the lane into reg from an xmm in r/m, the
                // reverse of every other extract here.
                let (reg, rm) = if stem == "pextrw" && map == 1 {
                    match rm {
                        Rm::Reg(number) => (number, Rm::Reg(reg)),
                        rm => (reg, rm),
                    }
                } else {
                    (reg, rm)
                };
                code = self.emit(self.encode_op(Op {
                    opcode: &[op],
                    reg,
                    rm,
                    imm: &[imm],
                    vex: Some(Vex {
                        map,
                        pp: 1,
                        wide,
                        vvvv,
                        ..Vex::default()
                    }),
                    ..Op::default()
                })?);
            }
            "vpmovmskb" | "vmovmskps" | "vmovmskpd" => {
                let (pp, op) = match opcode.as_str() {
                    "vmovmskps" => (0u8, 0x50u8),
                    "vmovmskpd" => (1, 0x50),
                    _ => (1, 0xd7),
                };
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(format!(
                        "{} requires 2 operands",
                        opcode
                    )));
                };
                let (rm, len256) = parse_vector(src.trim()).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires a vector source", opcode))
                })?;
                let reg = gpr(dst.trim()).filter(|g| g.width == 32).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires a 32-bit destination", opcode))
                })?;
                code = self.emit(self.encode_op(Op {
                    opcode: &[op],
                    reg: reg.num,
                    rm: Rm::Reg(rm),
                    vex: Some(Vex {
                        pp,
                        len256,
                        ..Vex::default()
                    }),
                    ..Op::default()
                })?);
            }
            "vmovq" => {
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError("vmovq requires 2 operands".into()));
                };
                let (op, pp, wide, reg, rm) = match (parse_xmm(src), parse_xmm(dst)) {
                    // Between a general register and a vector register the
                    // 66 map carries the transfer with REX.W set.
                    (None, Some(reg)) if !self.is_memory(src) => {
                        (0x6e, 1, true, reg, self.parse_rm(src, false)?.0)
                    }
                    (Some(reg), None) if !self.is_memory(dst) => {
                        (0x7e, 1, true, reg, self.parse_rm(dst, false)?.0)
                    }
                    (Some(reg), None) => (0xd6, 1, false, reg, self.parse_rm(dst, true)?.0),
                    // Between two registers the two-byte prefix extends the
                    // reg field alone, so put the high register there and
                    // pick the direction to match.
                    (Some(src_reg), Some(dst_reg)) if src_reg >= 8 && dst_reg < 8 => {
                        (0xd6, 1, false, src_reg, Rm::Reg(dst_reg))
                    }
                    (_, Some(reg)) => (0x7e, 2, false, reg, self.parse_rm(src, true)?.0),
                    _ => {
                        return Err(RasError::EncodingError(
                            "vmovq: memory to memory is not encodable".into(),
                        ));
                    }
                };
                if wide {
                    let width = gpr(if op == 0x6e { src } else { dst }.trim())
                        .map(|g| g.width)
                        .unwrap_or(0);
                    if width != 64 {
                        return Err(RasError::EncodingError(
                            "vmovq requires a 64-bit general register".into(),
                        ));
                    }
                }
                code = self.emit(self.encode_op(Op {
                    opcode: &[op],
                    reg,
                    rm,
                    vex: Some(Vex {
                        pp,
                        wide,
                        ..Vex::default()
                    }),
                    ..Op::default()
                })?);
            }
            // The scalar moves merge from a second source between registers
            // and address memory in two operands.
            "vmovss" | "vmovsd" | "vmovhps" | "vmovlps" | "vmovhpd" | "vmovlpd" => {
                let scalar = matches!(opcode.as_str(), "vmovss" | "vmovsd");
                if scalar && self.scalar_move_needs_evex(&inst.operands) {
                    let enc = self.encode_evex_scalar_move(&opcode, &inst.operands)?;
                    code = self.emit(enc);
                    self.position += code.len();
                    return Ok(code);
                }
                if scalar
                    && inst.operands.len() == 2
                    && !inst.operands.iter().any(|op| self.is_memory(op))
                {
                    return Err(RasError::EncodingError(format!(
                        "{} between registers names a merge source",
                        opcode
                    )));
                }
                let (pp, load) = match opcode.as_str() {
                    "vmovss" => (2u8, 0x10u8),
                    "vmovsd" => (3, 0x10),
                    "vmovlps" => (0, 0x12),
                    "vmovlpd" => (1, 0x12),
                    "vmovhpd" => (1, 0x16),
                    _ => (0, 0x16),
                };
                match inst.operands.as_slice() {
                    [src, dst] => {
                        let (opcode_byte, reg, rm) = if self.is_memory(src) {
                            let reg = parse_xmm(dst).ok_or_else(|| {
                                RasError::EncodingError(format!(
                                    "{} requires an xmm destination",
                                    opcode
                                ))
                            })?;
                            (load, reg, self.parse_rm(src, true)?.0)
                        } else {
                            let reg = parse_xmm(src).ok_or_else(|| {
                                RasError::EncodingError(format!(
                                    "{} requires an xmm source",
                                    opcode
                                ))
                            })?;
                            (load + 1, reg, self.parse_rm(dst, true)?.0)
                        };
                        code = self.emit(self.encode_op(Op {
                            opcode: &[opcode_byte],
                            reg,
                            rm,
                            vex: Some(Vex {
                                pp,
                                ..Vex::default()
                            }),
                            ..Op::default()
                        })?);
                    }
                    [src, first, dst] => {
                        let destination = parse_xmm(dst).ok_or_else(|| {
                            RasError::EncodingError(format!(
                                "{} requires an xmm destination",
                                opcode
                            ))
                        })?;
                        let vvvv = parse_xmm(first).ok_or_else(|| {
                            RasError::EncodingError(format!("{} requires an xmm source", opcode))
                        })?;
                        // Keep the high register in the reg field so the
                        // two-byte prefix still fits.
                        let (op, reg, rm) = match parse_xmm(src) {
                            Some(source) if source >= 8 && destination < 8 => {
                                (load + 1, source, Rm::Reg(destination))
                            }
                            _ => (load, destination, self.parse_rm(src, true)?.0),
                        };
                        code = self.emit(self.encode_op(Op {
                            opcode: &[op],
                            reg,
                            rm,
                            vex: Some(Vex {
                                pp,
                                vvvv,
                                ..Vex::default()
                            }),
                            ..Op::default()
                        })?);
                    }
                    _ => {
                        return Err(RasError::EncodingError(format!(
                            "{} requires 2 or 3 operands",
                            opcode
                        )));
                    }
                }
            }
            "vcvtsi2ssl" | "vcvtsi2ssq" | "vcvtsi2sdl" | "vcvtsi2sdq" => {
                let [src, first, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(
                        "vcvtsi2 requires 3 operands".into(),
                    ));
                };
                let wide = opcode.ends_with('q');
                let (rm, src_gpr) = self.parse_rm(src, false)?;
                operand_width(Some(if wide { 64 } else { 32 }), &[src_gpr])?;
                let vvvv = parse_xmm(first).ok_or_else(|| {
                    RasError::EncodingError("vcvtsi2 requires an xmm source".into())
                })?;
                let reg = parse_xmm(dst).ok_or_else(|| {
                    RasError::EncodingError("vcvtsi2 requires an xmm destination".into())
                })?;
                code = self.emit(self.encode_op(Op {
                    opcode: &[0x2a],
                    reg,
                    rm,
                    vex: Some(Vex {
                        pp: if opcode.starts_with("vcvtsi2ss") {
                            2
                        } else {
                            3
                        },
                        wide,
                        vvvv,
                        ..Vex::default()
                    }),
                    ..Op::default()
                })?);
            }
            "vcvtss2si" | "vcvtss2sil" | "vcvtss2siq" | "vcvtsd2si" | "vcvtsd2sil"
            | "vcvtsd2siq" | "vcvttss2si" | "vcvttss2sil" | "vcvttss2siq" | "vcvttsd2si"
            | "vcvttsd2sil" | "vcvttsd2siq" => {
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError("vcvt requires 2 operands".into()));
                };
                let single = opcode.starts_with("vcvtss") || opcode.starts_with("vcvttss");
                let truncate = opcode.starts_with("vcvtt");
                let reg = gpr(dst.trim())
                    .filter(|g| matches!(g.width, 32 | 64))
                    .ok_or_else(|| {
                        RasError::EncodingError(
                            "vcvt: destination must be a 32- or 64-bit register".into(),
                        )
                    })?;
                let wide = reg.width == 64;
                if matches!(opcode.as_bytes().last(), Some(b'l')) && wide
                    || matches!(opcode.as_bytes().last(), Some(b'q')) && !wide
                {
                    return Err(RasError::EncodingError(
                        "vcvt: destination width does not match suffix".into(),
                    ));
                }
                code = self.emit(self.encode_op(Op {
                    opcode: &[if truncate { 0x2c } else { 0x2d }],
                    reg: reg.num,
                    rm: self.parse_rm(src, true)?.0,
                    vex: Some(Vex {
                        pp: if single { 2 } else { 3 },
                        wide,
                        ..Vex::default()
                    }),
                    ..Op::default()
                })?);
            }
            name if fma(name).is_some() => {
                let (wide, op) = fma(name).unwrap_or((false, 0));
                let enc = self.encode_vex_rrm(2, 1, wide, op, &inst.operands)?;
                code = self.emit(enc);
            }
            name if mask_move(name).is_some() => {
                let (wide, load) = mask_move(name).unwrap_or((false, 0));
                let [src, first, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(format!(
                        "{} requires 3 operands",
                        name
                    )));
                };
                let store = self.is_memory(dst);
                let (vector, memory) = if store { (src, dst) } else { (dst, src) };
                let (reg, reg_wide) = parse_vector(vector.trim()).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires a vector operand", name))
                })?;
                let (vvvv, mask_wide) = parse_vector(first.trim()).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires a vector mask", name))
                })?;
                if !self.is_memory(memory) {
                    return Err(RasError::EncodingError(format!(
                        "{} requires a memory operand",
                        name
                    )));
                }
                code = self.emit(self.encode_op(Op {
                    opcode: &[load + u8::from(store) * 2],
                    reg,
                    rm: self.parse_rm(memory, true)?.0,
                    vex: Some(Vex {
                        map: 2,
                        pp: 1,
                        wide,
                        len256: reg_wide || mask_wide,
                        vvvv,
                    }),
                    ..Op::default()
                })?);
            }
            "vmovaps" | "vmovapd" | "vmovups" | "vmovupd" => {
                let pp = u8::from(opcode.ends_with("pd"));
                let load = if opcode.starts_with("vmova") {
                    0x28
                } else {
                    0x10
                };
                let enc = self.encode_vex_move(pp, load, load + 1, &inst.operands)?;
                code = self.emit(enc);
            }
            "vmovntdq" | "vmovntps" | "vmovntpd" => {
                let (pp, op) = match opcode.as_str() {
                    "vmovntps" => (0u8, 0x2bu8),
                    "vmovntpd" => (1, 0x2b),
                    _ => (1, 0xe7),
                };
                let [src, dst] = inst.operands.as_slice() else {
                    return Err(RasError::EncodingError(format!(
                        "{} requires 2 operands",
                        opcode
                    )));
                };
                let (reg, wide) = parse_vector(src.trim()).ok_or_else(|| {
                    RasError::EncodingError(format!("{} requires a vector source", opcode))
                })?;
                if !self.is_memory(dst) {
                    return Err(RasError::EncodingError(format!(
                        "{} requires a memory destination",
                        opcode
                    )));
                }
                code = self.emit(self.encode_op(Op {
                    opcode: &[op],
                    reg,
                    rm: self.parse_rm(dst, true)?.0,
                    vex: Some(Vex {
                        pp,
                        len256: wide,
                        ..Vex::default()
                    }),
                    ..Op::default()
                })?);
            }
            "vmovdqa" => {
                let enc = self.encode_vex_move(1, 0x6f, 0x7f, &inst.operands)?;
                code = self.emit(enc);
            }
            "vmovdqu" => {
                let enc = self.encode_vex_move(2, 0x6f, 0x7f, &inst.operands)?;
                code = self.emit(enc);
            }
            "vmovd" => {
                let enc = self.encode_vmovd(&inst.operands)?;
                code = self.emit(enc);
            }
            "shlxl" | "shlxq" | "shrxl" | "shrxq" | "sarxl" | "sarxq" => {
                let enc = self.encode_bmi2_shift(&opcode, &inst.operands)?;
                code = self.emit(enc);
            }
            // AVX2 per-lane shifts: VEX.128/256.66.0F38.W0/W1 45/46/47 /r.
            "vpsllvd" | "vpsllvq" | "vpsrlvd" | "vpsrlvq" | "vpsravd" => {
                let op = match opcode.as_str() {
                    "vpsrlvd" | "vpsrlvq" => 0x45,
                    "vpsravd" => 0x46,
                    _ => 0x47,
                };
                let enc = self.encode_vex_rrm(2, 1, opcode.ends_with('q'), op, &inst.operands)?;
                code = self.emit(enc);
            }
            "vzeroupper" | "vzeroall" => {
                if !inst.operands.is_empty() {
                    return Err(RasError::EncodingError(format!(
                        "{} takes no operands",
                        opcode
                    )));
                }
                code.extend_from_slice(if opcode == "vzeroupper" {
                    &[0xc5, 0xf8, 0x77]
                } else {
                    &[0xc5, 0xfc, 0x77]
                });
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
                code = self.emit(self.encode_op(Op {
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
    segment: Option<u8>,
    disp: i64,
    label: Option<String>,
    base: Option<u8>,
    index: Option<(u8, u8)>,
    vindex: Option<(u8, u8, u8)>,
    broadcast: Option<u8>,
    rip: bool,
    tls: Option<TlsModel>,
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
    vex: Option<Vex>,
    evex: Option<Evex>,
    mode: Mode,
}

/// The parts of a VEX prefix that vary between instructions.
///
/// `map` selects the opcode map the escape bytes would otherwise name: 1 for
/// `0f`, 2 for `0f 38`, 3 for `0f 3a`. `pp` carries the legacy prefix in the
/// same way: 0 for none, 1 for `66`, 2 for `f3`, 3 for `f2`. `vvvv` names the
/// extra source operand by its plain register number; the prefix carries it
/// inverted, so an instruction without one leaves it at zero.
#[derive(Clone, Copy)]
struct Vex {
    map: u8,
    pp: u8,
    len256: bool,
    wide: bool,
    vvvv: u8,
}

impl Default for Vex {
    fn default() -> Self {
        Self {
            map: 1,
            pp: 0,
            len256: false,
            wide: false,
            vvvv: 0,
        }
    }
}

impl Vex {
    /// Two-byte form when the fields it cannot express are all at their
    /// defaults, otherwise the three-byte form.
    fn emit(self, rex_r: u8, rex_x: u8, rex_b: u8, bytes: &mut Vec<u8>) {
        let tail = ((!self.vvvv & 15) << 3) | (u8::from(self.len256) << 2) | self.pp;
        if self.map == 1 && !self.wide && rex_x == 0 && rex_b == 0 {
            bytes.push(0xc5);
            bytes.push(((1 - rex_r) << 7) | tail);
            return;
        }
        bytes.push(0xc4);
        bytes.push(((1 - rex_r) << 7) | ((1 - rex_x) << 6) | ((1 - rex_b) << 5) | self.map);
        bytes.push((u8::from(self.wide) << 7) | tail);
    }
}

/// One AVX-512 form. `lens` is a bit per legal vector length and `flags`
/// records which decorators the form accepts.
#[derive(Clone, Copy)]
struct EvexEntry {
    name: &'static str,
    map: u8,
    pp: u8,
    wide: bool,
    opcode: u8,
    digit: u8,
    form: VexForm,
    tuple: Tuple,
    flags: u8,
    lens: u8,
    src_shift: u8,
    dst_shift: u8,
}

impl EvexEntry {
    fn has(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    fn takes_imm(&self) -> bool {
        matches!(
            self.form,
            VexForm::RmImm | VexForm::NdsImm | VexForm::ExtractImm | VexForm::ShiftImm
        )
    }
}

/// The AVX-512 form of a mnemonic. A handful of mnemonics name both an
/// immediate and a register-controlled operation, so the spelling of the
/// first operand picks between them.
fn packed_evex(name: &str, is_imm: bool) -> Option<&'static EvexEntry> {
    let start = EVEX_TABLE.partition_point(|entry| entry.name < name);
    EVEX_TABLE[start..]
        .iter()
        .take_while(|entry| entry.name == name)
        .find(|entry| entry.takes_imm() == is_imm)
}

/// Whether a mnemonic has any AVX-512 form at all.
fn is_evex_mnemonic(name: &str) -> bool {
    let start = EVEX_TABLE.partition_point(|entry| entry.name < name);
    EVEX_TABLE
        .get(start)
        .is_some_and(|entry| entry.name == name)
}

/// The MMX operations written `op mm/m64, mm`. These keep the SSE opcode and
/// drop the `66` prefix; the operations that only ever existed at 128 bits
/// are absent.
const MMX_PACKED: &[&str] = &[
    "packssdw",
    "packsswb",
    "packuswb",
    "paddb",
    "paddd",
    "paddq",
    "paddsb",
    "paddsw",
    "paddusb",
    "paddusw",
    "paddw",
    "pand",
    "pandn",
    "pavgb",
    "pavgw",
    "pcmpeqb",
    "pcmpeqd",
    "pcmpeqw",
    "pcmpgtb",
    "pcmpgtd",
    "pcmpgtw",
    "pmaddwd",
    "pmaxsw",
    "pmaxub",
    "pminsw",
    "pminub",
    "pmulhuw",
    "pmulhw",
    "pmullw",
    "pmuludq",
    "por",
    "psadbw",
    "pslld",
    "psllq",
    "psllw",
    "psrad",
    "psraw",
    "psrld",
    "psrlq",
    "psrlw",
    "psubb",
    "psubd",
    "psubq",
    "psubsb",
    "psubsw",
    "psubusb",
    "psubusw",
    "psubw",
    "punpckhbw",
    "punpckhdq",
    "punpckhwd",
    "punpcklbw",
    "punpckldq",
    "punpcklwd",
    "pxor",
    "pabsb",
    "pabsd",
    "pabsw",
    "phaddd",
    "phaddsw",
    "phaddw",
    "phsubd",
    "phsubsw",
    "phsubw",
    "pmaddubsw",
    "pmulhrsw",
    "pshufb",
    "psignb",
    "psignd",
    "psignw",
];

/// An MMX operation written `op mm/m64, mm`, as `(opcode map, opcode byte)`.
fn mmx_packed(name: &str) -> Option<(u8, u8)> {
    if !MMX_PACKED.contains(&name) {
        return None;
    }
    packed_sse(name).map(|(_, map, opcode)| (map, opcode))
}

/// The MMX conversions between a vector register and an MMX register, as
/// `(legacy prefix, opcode, the MMX register is the destination)`.
fn mmx_convert(name: &str) -> Option<(u8, u8, bool)> {
    match name {
        "cvtpi2ps" => Some((0, 0x2a, false)),
        "cvtps2pi" => Some((0, 0x2d, true)),
        "cvttps2pi" => Some((0, 0x2c, true)),
        "cvtpi2pd" => Some((0x66, 0x2a, false)),
        "cvtpd2pi" => Some((0x66, 0x2d, true)),
        "cvttpd2pi" => Some((0x66, 0x2c, true)),
        _ => None,
    }
}

/// Whether a mnemonic has any MMX form.
fn mmx_mnemonic(name: &str) -> bool {
    mmx_packed(name).is_some()
        || mmx_convert(name).is_some()
        || matches!(
            name,
            "movq"
                | "movd"
                | "pshufw"
                | "palignr"
                | "pmovmskb"
                | "maskmovq"
                | "movntq"
                | "pextrw"
                | "pinsrw"
        )
}

/// The AVX-512 broadcasts that take their source in a general register, as
/// `(opcode, register width, W)`.
fn broadcast_from_general(name: &str) -> Option<(u8, u8, bool)> {
    match name {
        "vpbroadcastb" => Some((0x7a, 32, false)),
        "vpbroadcastw" => Some((0x7b, 32, false)),
        "vpbroadcastd" => Some((0x7c, 32, false)),
        "vpbroadcastq" => Some((0x7c, 64, true)),
        _ => None,
    }
}

/// A VSIB gather, as `(64-bit index, 64-bit element, opcode byte)`. The
/// mnemonic spells the index size first and the gathered element second.
fn gather(name: &str) -> Option<(bool, bool, u8)> {
    let rest = name.strip_prefix('v')?;
    let (integer, rest) = match rest.strip_prefix("pgather") {
        Some(rest) => (true, rest),
        None => (false, rest.strip_prefix("gather")?),
    };
    let (index_wide, rest) = match rest.strip_prefix('d') {
        Some(rest) => (false, rest),
        None => (true, rest.strip_prefix('q')?),
    };
    let element_wide = match (integer, rest) {
        (true, "d") | (false, "ps") => false,
        (true, "q") | (false, "pd") => true,
        _ => return None,
    };
    let opcode = if integer { 0x90 } else { 0x92 } + u8::from(index_wide);
    Some((index_wide, element_wide, opcode))
}

/// The operand shape of an opmask instruction.
#[derive(Clone, Copy, PartialEq)]
enum KForm {
    /// `op k, k`, with the source in `r/m`.
    Unary,
    /// `op k, k, k`, with the middle operand in `vvvv` and `L` set.
    Binary,
    /// `op $imm, k, k`.
    Shift,
    /// `kmov`, which also reaches general registers and memory.
    Move,
}

/// `W` and `pp` for an opmask element-size suffix.
fn kmask_width(suffix: &str) -> Option<(bool, u8)> {
    match suffix {
        "b" => Some((false, 1)),
        "w" => Some((false, 0)),
        "d" => Some((true, 1)),
        "q" => Some((true, 0)),
        _ => None,
    }
}

/// An opmask instruction, as `(map, pp, W, opcode, form)`.
fn kmask(name: &str) -> Option<(u8, u8, bool, u8, KForm)> {
    let rest = name.strip_prefix('k')?;
    if let Some(suffix) = rest.strip_prefix("unpck") {
        let (wide, pp) = match suffix {
            "bw" => (false, 1),
            "wd" => (false, 0),
            "dq" => (true, 0),
            _ => return None,
        };
        return Some((1, pp, wide, 0x4b, KForm::Binary));
    }
    for (stem, opcode) in [("shiftr", 0x30u8), ("shiftl", 0x32)] {
        if let Some(suffix) = rest.strip_prefix(stem) {
            let (wide, high) = match suffix {
                "b" => (false, false),
                "w" => (true, false),
                "d" => (false, true),
                "q" => (true, true),
                _ => return None,
            };
            return Some((3, 1, wide, opcode + u8::from(high), KForm::Shift));
        }
    }
    let (stem, suffix) = rest.split_at(rest.len().checked_sub(1)?);
    let (wide, pp) = kmask_width(suffix)?;
    let (opcode, form) = match stem {
        "mov" => (0x90, KForm::Move),
        "and" => (0x41, KForm::Binary),
        "andn" => (0x42, KForm::Binary),
        "not" => (0x44, KForm::Unary),
        "or" => (0x45, KForm::Binary),
        "xnor" => (0x46, KForm::Binary),
        "xor" => (0x47, KForm::Binary),
        "add" => (0x4a, KForm::Binary),
        "ortest" => (0x98, KForm::Unary),
        "test" => (0x99, KForm::Unary),
        _ => return None,
    };
    Some((1, pp, wide, opcode, form))
}

/// `W`, `pp` and the general-register width `kmov` uses when it crosses
/// between an opmask and a general register, which differ from the fields the
/// register and memory forms use.
fn kmov_transfer(suffix: &str) -> Option<(bool, u8, u8)> {
    match suffix {
        "b" => Some((false, 1, 32)),
        "w" => Some((false, 0, 32)),
        "d" => Some((false, 3, 32)),
        "q" => Some((true, 3, 64)),
        _ => None,
    }
}

/// A VSIB scatter, as `(64-bit index, 64-bit element, opcode byte)`. The
/// mnemonic spells the index size first and the scattered element second.
fn scatter(name: &str) -> Option<(bool, bool, u8)> {
    let rest = name.strip_prefix('v')?;
    let (integer, rest) = match rest.strip_prefix("pscatter") {
        Some(rest) => (true, rest),
        None => (false, rest.strip_prefix("scatter")?),
    };
    let (index_wide, rest) = match rest.strip_prefix('d') {
        Some(rest) => (false, rest),
        None => (true, rest.strip_prefix('q')?),
    };
    let element_wide = match (integer, rest) {
        (true, "d") | (false, "ps") => false,
        (true, "q") | (false, "pd") => true,
        _ => return None,
    };
    let opcode = if integer { 0xa0 } else { 0xa2 } + u8::from(index_wide);
    Some((index_wide, element_wide, opcode))
}

/// A VSIB prefetch, as `(64-bit index, 64-bit element, opcode byte, modrm
/// digit)`. These touch no vector register, so the hint rides in `reg`.
fn vsib_prefetch(name: &str) -> Option<(bool, bool, u8, u8)> {
    let rest = name.strip_prefix('v')?;
    let (digit, rest) = match rest.strip_prefix("gatherpf") {
        Some(rest) => (1u8, rest),
        None => (5, rest.strip_prefix("scatterpf")?),
    };
    let (hint, rest) = match rest.strip_prefix('0') {
        Some(rest) => (0u8, rest),
        None => (1, rest.strip_prefix('1')?),
    };
    let (index_wide, rest) = match rest.strip_prefix('d') {
        Some(rest) => (false, rest),
        None => (true, rest.strip_prefix('q')?),
    };
    let element_wide = match rest {
        "ps" => false,
        "pd" => true,
        _ => return None,
    };
    Some((
        index_wide,
        element_wide,
        0xc6 + u8::from(index_wide),
        digit + hint,
    ))
}

/// A `vcvtpd2ps`-style convert whose mnemonic spells its source width, as the
/// unsuffixed mnemonic and the vector length that spelling pins.
fn sized_convert(name: &str) -> Option<(&str, u8)> {
    let (base, length) = match name.strip_suffix('x') {
        Some(base) => (base, 0),
        None => (name.strip_suffix('y')?, 1),
    };
    matches!(base, "vcvtpd2ps" | "vcvtpd2dq" | "vcvttpd2dq").then_some((base, length))
}

/// Whether a mnemonic has a VEX spelling for this operand shape, which
/// decides whether operands that do not demand EVEX may still take it.
fn vex_form_exists(name: &str, is_imm: bool) -> bool {
    if packed_vex_permil(name, is_imm).is_some() {
        return true;
    }
    if let Some((_, _, _, _, form)) = packed_vex(name)
        && matches!(form, VexForm::RmImm | VexForm::NdsImm | VexForm::ExtractImm) == is_imm
    {
        return true;
    }
    if is_imm {
        return vex_shift_imm(name).is_some();
    }
    fma(name).is_some()
        || mask_move(name).is_some()
        || matches!(
            name,
            "vmovaps"
                | "vmovups"
                | "vmovapd"
                | "vmovupd"
                | "vmovdqa"
                | "vmovdqu"
                | "vpsllvd"
                | "vpsllvq"
                | "vpsrlvd"
                | "vpsrlvq"
                | "vpsravd"
        )
}

/// The store opcode of an AVX-512 move, for the spelling that names memory
/// as the destination.
fn evex_store(name: &str) -> Option<u8> {
    match name {
        "vmovaps" | "vmovapd" => Some(0x29),
        "vmovups" | "vmovupd" => Some(0x11),
        "vmovdqa32" | "vmovdqa64" | "vmovdqu8" | "vmovdqu16" | "vmovdqu32" | "vmovdqu64" => {
            Some(0x7f)
        }
        _ => None,
    }
}

/// An `{er}` or `{sae}` operand, as the `L'L` bits it writes and whether it
/// suppresses exceptions only.
fn rounding_decorator(op: &str) -> Option<(u8, bool)> {
    match op.trim() {
        "{rn-sae}" => Some((0, false)),
        "{rd-sae}" => Some((1, false)),
        "{ru-sae}" => Some((2, false)),
        "{rz-sae}" => Some((3, false)),
        "{sae}" => Some((0, true)),
        _ => None,
    }
}

/// A destination split from its `{%k}` and `{z}` decorators.
fn split_mask(op: &str) -> Result<(&str, u8, bool), RasError> {
    let mut base = op.trim();
    let mut zeroing = false;
    if let Some(rest) = base.strip_suffix("{z}") {
        zeroing = true;
        base = rest.trim_end();
    }
    let mut mask = 0;
    if base.ends_with('}') {
        let open = base
            .rfind('{')
            .ok_or_else(|| RasError::EncodingError(format!("Malformed decorator: {}", op)))?;
        mask = mask_register(&base[open + 1..base.len() - 1]).ok_or_else(|| {
            RasError::EncodingError(format!("Expected an opmask register: {}", op))
        })?;
        if mask == 0 {
            return Err(RasError::EncodingError(
                "%k0 cannot be used as a write mask".into(),
            ));
        }
        base = base[..open].trim_end();
    }
    if zeroing && mask == 0 {
        return Err(RasError::EncodingError("{z} requires a write mask".into()));
    }
    if base.contains('{') && !base.contains("{1to") {
        return Err(RasError::EncodingError(format!(
            "Malformed decorator: {}",
            op
        )));
    }
    Ok((base, mask, zeroing))
}

/// How an EVEX instruction scales a one-byte displacement. The scale is the
/// size of the memory the instruction actually touches, which depends on the
/// tuple type, the vector length and, for the broadcast forms, on W.
#[derive(Clone, Copy, PartialEq)]
enum Tuple {
    Scaled(u8),
    Fixed(u8),
    Movddup,
}

/// The EVEX prefix fields. `ll` is the vector length, 0 for 128 bits through
/// 2 for 512, or the rounding mode when `broadcast` carries `{er}`/`{sae}`.
#[derive(Clone, Copy)]
struct Evex {
    map: u8,
    pp: u8,
    wide: bool,
    vvvv: u8,
    ll: u8,
    mask: u8,
    zeroing: bool,
    broadcast: bool,
    tuple: Tuple,
}

impl Default for Evex {
    fn default() -> Self {
        Self {
            map: 1,
            pp: 0,
            wide: false,
            vvvv: 0,
            ll: 0,
            mask: 0,
            zeroing: false,
            broadcast: false,
            tuple: Tuple::Scaled(1),
        }
    }
}

impl Evex {
    /// The displacement scale, `N` in the SDM's disp8*N.
    fn disp_scale(self, broadcast: bool) -> i64 {
        let vl: i64 = 16 << self.ll;
        if broadcast {
            return if self.wide { 8 } else { 4 };
        }
        match self.tuple {
            Tuple::Scaled(divisor) => vl / i64::from(divisor),
            Tuple::Fixed(size) => i64::from(size),
            Tuple::Movddup => {
                if self.ll == 0 {
                    8
                } else {
                    vl
                }
            }
        }
    }

    fn emit(self, reg: u8, rex_x: u8, rex_b: u8, v_high: u8, bytes: &mut Vec<u8>) {
        let rex_r = (reg >> 3) & 1;
        let r_prime = (reg >> 4) & 1;
        let v_prime = ((self.vvvv >> 4) & 1) | v_high;
        bytes.push(0x62);
        bytes.push(
            ((1 - rex_r) << 7)
                | ((1 - rex_x) << 6)
                | ((1 - rex_b) << 5)
                | ((1 - r_prime) << 4)
                | self.map,
        );
        bytes.push((u8::from(self.wide) << 7) | ((!self.vvvv & 15) << 3) | 0b100 | self.pp);
        bytes.push(
            (u8::from(self.zeroing) << 7)
                | (self.ll << 5)
                | (u8::from(self.broadcast) << 4)
                | ((1 - v_prime) << 3)
                | self.mask,
        );
    }
}

fn encode_op(op: Op<'_>) -> Result<Encoded, RasError> {
    if op.mode == Mode::Protected && (op.vex.is_some() || op.evex.is_some()) {
        return Err(RasError::EncodingError(
            "VEX and EVEX encodings are not implemented in 32-bit mode".into(),
        ));
    }
    let mut tail = Vec::new();
    let mut rex_x = 0;
    let mut rex_b = 0;
    let mut v_high = 0;
    let reg_field = (op.reg & 7) << 3;
    let mut segment_prefix = None;
    let mut disp_field = None;
    let mut tls = None;
    let mut broadcast = false;
    match &op.rm {
        Rm::Reg(number) => {
            tail.push(0xc0 | reg_field | (number & 7));
            rex_b = (number >> 3) & 1;
            if op.evex.is_some() {
                rex_x = (number >> 4) & 1;
            } else if number >> 3 > 1 {
                return Err(RasError::EncodingError(
                    "Register requires an EVEX encoding".into(),
                ));
            }
        }
        Rm::Mem(mem) if mem.rip => {
            tail.push(reg_field | 5);
            disp_field = Some(tail.len());
            tls = mem.tls;
            tail.extend_from_slice(&disp32(mem.disp)?);
        }
        Rm::Mem(mem) => {
            segment_prefix = mem.segment;
            tls = mem.tls;
            broadcast = mem.broadcast.is_some();
            if let Some((index, _, _)) = mem.vindex {
                rex_x = (index >> 3) & 1;
                v_high = (index >> 4) & 1;
                if v_high == 1 && op.evex.is_none() {
                    return Err(RasError::EncodingError(
                        "Vector index requires an EVEX encoding".into(),
                    ));
                }
            } else if let Some((index, _)) = mem.index {
                rex_x = index >> 3;
            }
            let index_field = match (mem.vindex, mem.index) {
                (Some((index, _, _)), _) => index & 7,
                (None, Some((index, _))) => index & 7,
                _ => 4,
            };
            let scale = match (mem.vindex, mem.index) {
                (Some((_, bits, _)), _) => bits,
                (None, Some((_, bits))) => bits,
                _ => 0,
            };
            let scale_n = op.evex.map_or(1, |evex| evex.disp_scale(broadcast)).max(1);
            match mem.base {
                None => {
                    tail.push(reg_field | 4);
                    tail.push((scale << 6) | (index_field << 3) | 5);
                    disp_field = tls.map(|_| tail.len());
                    tail.extend_from_slice(&disp32(mem.disp)?);
                }
                Some(base) => {
                    rex_b = base >> 3;
                    let need_sib = mem.index.is_some() || mem.vindex.is_some() || base & 7 == 4;
                    let scaled = (mem.disp % scale_n == 0).then(|| mem.disp / scale_n);
                    let mode = if mem.tls.is_some() {
                        2
                    } else if mem.disp == 0 && base & 7 != 5 {
                        0
                    } else if scaled.is_some_and(|value| i8::try_from(value).is_ok()) {
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
                        1 => tail.push(scaled.unwrap_or(mem.disp) as u8),
                        2 => {
                            disp_field = tls.map(|_| tail.len());
                            tail.extend_from_slice(&disp32(mem.disp)?);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    let mut bytes = Vec::new();
    if let Some(prefix) = segment_prefix {
        bytes.push(prefix);
    }
    if let Some(mut evex) = op.evex {
        evex.broadcast |= broadcast;
        evex.emit(op.reg, rex_x, rex_b, v_high, &mut bytes);
    } else if let Some(vex) = op.vex {
        vex.emit(op.reg >> 3, rex_x, rex_b, &mut bytes);
    } else {
        let rex = 0x40 | (u8::from(op.wide) << 3) | ((op.reg >> 3) << 2) | (rex_x << 1) | rex_b;
        let needs_rex = rex != 0x40 || op.byte_rex;
        if op.mode == Mode::Protected {
            if needs_rex {
                return Err(RasError::EncodingError(
                    "32-bit mode has no REX prefix: 64-bit operands and the registers above %edi are unavailable".into(),
                ));
            }
            bytes.extend_from_slice(op.legacy);
        } else {
            if op.high && needs_rex {
                return Err(RasError::EncodingError(
                    "High byte registers cannot use REX".into(),
                ));
            }
            bytes.extend_from_slice(op.legacy);
            if needs_rex {
                bytes.push(rex);
            }
        }
    }
    bytes.extend_from_slice(op.opcode);
    let tail_start = bytes.len();
    bytes.extend(tail);
    let rip = disp_field.map(|position| RipFixup {
        disp_offset: tail_start + position,
        trailing: op.imm.len(),
        tls,
    });
    bytes.extend_from_slice(op.imm);
    Ok(Encoded { bytes, rip })
}

/// An `$imm` operand narrowed to the single byte the encoding carries.
fn immediate_byte(operand: &str) -> Result<u8, RasError> {
    let value = operand
        .trim()
        .strip_prefix('$')
        .map(parse_imm)
        .ok_or_else(|| RasError::EncodingError("Expected an immediate operand".into()))??;
    u8::try_from(value)
        .or_else(|_| i8::try_from(value).map(|narrow| narrow as u8))
        .map_err(|_| RasError::EncodingError(format!("Immediate out of range: {}", value)))
}

/// The operand shape of a VEX-encoded packed operation.
#[derive(Clone, Copy, PartialEq)]
enum VexForm {
    /// `op src, dst`.
    Rm,
    /// `op src2, src1, dst`.
    Nds,
    /// `op $imm, src, dst`.
    RmImm,
    /// `op $imm, src2, src1, dst`.
    NdsImm,
    /// `op mask, src2, src1, dst`, the mask in the high nibble of `imm8`.
    Is4,
    /// `op $imm, src, dst` with the source in `reg` and the destination in
    /// `r/m`, as the 128-bit lane extracts are written.
    ExtractImm,
    /// `op $imm, src, dst` with the source in `r/m`, the destination in
    /// `vvvv` and a fixed digit in `reg`, as the shift groups are written.
    ShiftImm,
}

/// Forms whose VEX encoding leaves `vvvv` unused, so the AT&T spelling keeps
/// the two operands the SSE form had.
const VEX_TWO_OPERAND: &[&str] = &[
    "sqrtps",
    "sqrtpd",
    "rcpps",
    "rsqrtps",
    "movshdup",
    "movsldup",
    "movddup",
    "lddqu",
    "movntdqa",
    "maskmovdqu",
    "cvtdq2ps",
    "cvtps2dq",
    "cvttps2dq",
    "cvtdq2pd",
    "cvtpd2dq",
    "cvttpd2dq",
    "cvtps2pd",
    "cvtpd2ps",
    "ptest",
    "pabsb",
    "pabsw",
    "pabsd",
    "phminposuw",
    "aesimc",
    "pmovsxbw",
    "pmovsxbd",
    "pmovsxbq",
    "pmovsxwd",
    "pmovsxwq",
    "pmovsxdq",
    "pmovzxbw",
    "pmovzxbd",
    "pmovzxbq",
    "pmovzxwd",
    "pmovzxwq",
    "pmovzxdq",
];

/// Immediate forms whose VEX encoding leaves `vvvv` unused.
const VEX_TWO_OPERAND_IMM: &[&str] = &[
    "roundps",
    "roundpd",
    "aeskeygenassist",
    "pcmpistri",
    "pcmpistrm",
    "pcmpestri",
    "pcmpestrm",
];

/// The VEX counterpart of a packed SSE form, as
/// `(opcode map, pp, W, opcode byte, operand shape)`.
fn packed_vex(name: &str) -> Option<(u8, u8, bool, u8, VexForm)> {
    let stem = name.strip_prefix('v')?;
    let pp = |prefix| match prefix {
        0x66 => 1,
        0xf3 => 2,
        0xf2 => 3,
        _ => 0,
    };
    // The variable blends move from `0f 38` to `0f 3a` and take their mask in
    // an immediate rather than in the fixed %xmm0.
    if packed_blendv(stem).is_some() {
        let opcode = match stem {
            "blendvps" => 0x4a,
            "blendvpd" => 0x4b,
            _ => 0x4c,
        };
        return Some((3, 1, false, opcode, VexForm::Is4));
    }
    if let Some((prefix, map, opcode)) = packed_sse_imm(stem) {
        let form = if VEX_TWO_OPERAND_IMM.contains(&stem) {
            VexForm::RmImm
        } else {
            VexForm::NdsImm
        };
        return Some((map, pp(prefix), false, opcode, form));
    }
    if let Some((prefix, map, opcode)) = packed_sse(stem) {
        let form = if VEX_TWO_OPERAND.contains(&stem) {
            VexForm::Rm
        } else {
            VexForm::Nds
        };
        return Some((map, pp(prefix), false, opcode, form));
    }
    let entry = match stem {
        "pshufd" => (1, 1, false, 0x70, VexForm::RmImm),
        "pshufhw" => (1, 2, false, 0x70, VexForm::RmImm),
        "pshuflw" => (1, 3, false, 0x70, VexForm::RmImm),
        "shufps" => (1, 0, false, 0xc6, VexForm::NdsImm),
        "shufpd" => (1, 1, false, 0xc6, VexForm::NdsImm),
        "addss" => (1, 2, false, 0x58, VexForm::Nds),
        "addsd" => (1, 3, false, 0x58, VexForm::Nds),
        "subss" => (1, 2, false, 0x5c, VexForm::Nds),
        "subsd" => (1, 3, false, 0x5c, VexForm::Nds),
        "mulss" => (1, 2, false, 0x59, VexForm::Nds),
        "mulsd" => (1, 3, false, 0x59, VexForm::Nds),
        "divss" => (1, 2, false, 0x5e, VexForm::Nds),
        "divsd" => (1, 3, false, 0x5e, VexForm::Nds),
        "minss" => (1, 2, false, 0x5d, VexForm::Nds),
        "minsd" => (1, 3, false, 0x5d, VexForm::Nds),
        "maxss" => (1, 2, false, 0x5f, VexForm::Nds),
        "maxsd" => (1, 3, false, 0x5f, VexForm::Nds),
        "sqrtss" => (1, 2, false, 0x51, VexForm::Nds),
        "sqrtsd" => (1, 3, false, 0x51, VexForm::Nds),
        "cvtss2sd" => (1, 2, false, 0x5a, VexForm::Nds),
        "cvtsd2ss" => (1, 3, false, 0x5a, VexForm::Nds),
        "cmpss" => (1, 2, false, 0xc2, VexForm::NdsImm),
        "cmpsd" => (1, 3, false, 0xc2, VexForm::NdsImm),
        "movhlps" => (1, 0, false, 0x12, VexForm::Nds),
        "movlhps" => (1, 0, false, 0x16, VexForm::Nds),
        "ucomiss" => (1, 0, false, 0x2e, VexForm::Rm),
        "comiss" => (1, 0, false, 0x2f, VexForm::Rm),
        "ucomisd" => (1, 1, false, 0x2e, VexForm::Rm),
        "comisd" => (1, 1, false, 0x2f, VexForm::Rm),
        "broadcastss" => (2, 1, false, 0x18, VexForm::Rm),
        "broadcastsd" => (2, 1, false, 0x19, VexForm::Rm),
        "broadcastf128" => (2, 1, false, 0x1a, VexForm::Rm),
        "broadcasti128" => (2, 1, false, 0x5a, VexForm::Rm),
        "pbroadcastb" => (2, 1, false, 0x78, VexForm::Rm),
        "pbroadcastw" => (2, 1, false, 0x79, VexForm::Rm),
        "pbroadcastd" => (2, 1, false, 0x58, VexForm::Rm),
        "pbroadcastq" => (2, 1, false, 0x59, VexForm::Rm),
        "insertf128" => (3, 1, false, 0x18, VexForm::NdsImm),
        "inserti128" => (3, 1, false, 0x38, VexForm::NdsImm),
        "extractf128" => (3, 1, false, 0x19, VexForm::ExtractImm),
        "extracti128" => (3, 1, false, 0x39, VexForm::ExtractImm),
        "perm2f128" => (3, 1, false, 0x06, VexForm::NdsImm),
        "perm2i128" => (3, 1, false, 0x46, VexForm::NdsImm),
        "permd" => (2, 1, false, 0x36, VexForm::Nds),
        "permps" => (2, 1, false, 0x16, VexForm::Nds),
        "permq" => (3, 1, true, 0x00, VexForm::RmImm),
        "permpd" => (3, 1, true, 0x01, VexForm::RmImm),
        _ => return None,
    };
    Some(entry)
}

/// `vpermilps` and `vpermilpd` have both an immediate and a vector control
/// form, under different opcodes; `is_imm` picks between them.
fn packed_vex_permil(name: &str, is_imm: bool) -> Option<(u8, u8, bool, u8, VexForm)> {
    let single = match name {
        "vpermilps" => true,
        "vpermilpd" => false,
        _ => return None,
    };
    Some(if is_imm {
        (
            3,
            1,
            false,
            if single { 0x04 } else { 0x05 },
            VexForm::RmImm,
        )
    } else {
        (2, 1, false, if single { 0x0c } else { 0x0d }, VexForm::Nds)
    })
}

/// An FMA3 mnemonic, as `(W, opcode byte)`. The operation picks a base in the
/// `132` block and the operand order shifts it by one block per step.
fn fma(name: &str) -> Option<(bool, u8)> {
    let rest = name.strip_prefix("vf")?;
    let (negate, rest) = match rest.strip_prefix('n') {
        Some(rest) => (true, rest),
        None => (false, rest),
    };
    let (kind, rest) = ["maddsub", "msubadd", "madd", "msub"]
        .into_iter()
        .find_map(|kind| rest.strip_prefix(kind).map(|rest| (kind, rest)))?;
    let (order, kind_suffix) = rest.split_at_checked(3)?;
    let scalar = match kind_suffix {
        "ps" | "pd" => false,
        "ss" | "sd" => true,
        _ => return None,
    };
    if scalar && matches!(kind, "maddsub" | "msubadd") {
        return None;
    }
    if negate && matches!(kind, "maddsub" | "msubadd") {
        return None;
    }
    let base = match (kind, negate) {
        ("maddsub", _) => 0x96,
        ("msubadd", _) => 0x97,
        ("madd", false) => 0x98,
        ("msub", false) => 0x9a,
        ("madd", true) => 0x9c,
        _ => 0x9e,
    } + u8::from(scalar);
    let step = match order {
        "132" => 0x00,
        "213" => 0x10,
        "231" => 0x20,
        _ => return None,
    };
    Some((kind_suffix.ends_with('d'), base + step))
}

/// AVX2 conditional vector moves, as `(W, load opcode)`; the store opcode is
/// two higher.
fn mask_move(name: &str) -> Option<(bool, u8)> {
    match name {
        "vmaskmovps" => Some((false, 0x2c)),
        "vmaskmovpd" => Some((false, 0x2d)),
        "vpmaskmovd" => Some((false, 0x8c)),
        "vpmaskmovq" => Some((true, 0x8c)),
        _ => None,
    }
}

/// VEX shift-by-immediate forms, as `(group opcode, /digit)`.
fn vex_shift_imm(name: &str) -> Option<(u8, u8)> {
    packed_shift_imm(name.strip_prefix('v')?)
}

/// The legacy prefix bytes a packed form carries, where 0 is none.
fn prefix_bytes(prefix: u8) -> &'static [u8] {
    match prefix {
        0x66 => &[0x66],
        0xf2 => &[0xf2],
        0xf3 => &[0xf3],
        _ => &[],
    }
}

/// The escape bytes for an opcode map, ahead of the opcode itself.
fn escape(map: u8, opcode: u8) -> Vec<u8> {
    match map {
        2 => vec![0x0f, 0x38, opcode],
        3 => vec![0x0f, 0x3a, opcode],
        _ => vec![0x0f, opcode],
    }
}

/// Packed SSE operations of the form `op xmm/m128, xmm`, as
/// `(legacy prefix, opcode map, opcode byte)`; a prefix of 0 is none, and
/// map 1 is `0f`, map 2 is `0f 38`.
fn packed_sse(name: &str) -> Option<(u8, u8, u8)> {
    let entry = match name {
        "paddb" => (0x66, 1, 0xfc),
        "paddw" => (0x66, 1, 0xfd),
        "paddd" => (0x66, 1, 0xfe),
        "paddq" => (0x66, 1, 0xd4),
        "paddsb" => (0x66, 1, 0xec),
        "paddsw" => (0x66, 1, 0xed),
        "paddusb" => (0x66, 1, 0xdc),
        "paddusw" => (0x66, 1, 0xdd),
        "psubb" => (0x66, 1, 0xf8),
        "psubw" => (0x66, 1, 0xf9),
        "psubd" => (0x66, 1, 0xfa),
        "psubq" => (0x66, 1, 0xfb),
        "psubsb" => (0x66, 1, 0xe8),
        "psubsw" => (0x66, 1, 0xe9),
        "psubusb" => (0x66, 1, 0xd8),
        "psubusw" => (0x66, 1, 0xd9),
        "pmullw" => (0x66, 1, 0xd5),
        "pmulhw" => (0x66, 1, 0xe5),
        "pmulhuw" => (0x66, 1, 0xe4),
        "pmuludq" => (0x66, 1, 0xf4),
        "pmaddwd" => (0x66, 1, 0xf5),
        "psadbw" => (0x66, 1, 0xf6),
        "pavgb" => (0x66, 1, 0xe0),
        "pavgw" => (0x66, 1, 0xe3),
        "packsswb" => (0x66, 1, 0x63),
        "packssdw" => (0x66, 1, 0x6b),
        "packuswb" => (0x66, 1, 0x67),
        "pminub" => (0x66, 1, 0xda),
        "pminsw" => (0x66, 1, 0xea),
        "pmaxub" => (0x66, 1, 0xde),
        "pmaxsw" => (0x66, 1, 0xee),
        "pand" => (0x66, 1, 0xdb),
        "pandn" => (0x66, 1, 0xdf),
        "por" => (0x66, 1, 0xeb),
        "pxor" => (0x66, 1, 0xef),
        "psllw" => (0x66, 1, 0xf1),
        "pslld" => (0x66, 1, 0xf2),
        "psllq" => (0x66, 1, 0xf3),
        "psrlw" => (0x66, 1, 0xd1),
        "psrld" => (0x66, 1, 0xd2),
        "psrlq" => (0x66, 1, 0xd3),
        "psraw" => (0x66, 1, 0xe1),
        "psrad" => (0x66, 1, 0xe2),
        "punpcklbw" => (0x66, 1, 0x60),
        "punpcklwd" => (0x66, 1, 0x61),
        "punpckldq" => (0x66, 1, 0x62),
        "punpcklqdq" => (0x66, 1, 0x6c),
        "punpckhbw" => (0x66, 1, 0x68),
        "punpckhwd" => (0x66, 1, 0x69),
        "punpckhdq" => (0x66, 1, 0x6a),
        "punpckhqdq" => (0x66, 1, 0x6d),
        "pcmpeqb" => (0x66, 1, 0x74),
        "pcmpeqw" => (0x66, 1, 0x75),
        "pcmpeqd" => (0x66, 1, 0x76),
        "pcmpgtb" => (0x66, 1, 0x64),
        "pcmpgtw" => (0x66, 1, 0x65),
        "pcmpgtd" => (0x66, 1, 0x66),
        "addps" => (0, 1, 0x58),
        "subps" => (0, 1, 0x5c),
        "mulps" => (0, 1, 0x59),
        "divps" => (0, 1, 0x5e),
        "minps" => (0, 1, 0x5d),
        "maxps" => (0, 1, 0x5f),
        "sqrtps" => (0, 1, 0x51),
        "rcpps" => (0, 1, 0x53),
        "rsqrtps" => (0, 1, 0x52),
        "rcpss" => (0xf3, 1, 0x53),
        "rsqrtss" => (0xf3, 1, 0x52),
        "andps" => (0, 1, 0x54),
        "andnps" => (0, 1, 0x55),
        "orps" => (0, 1, 0x56),
        "xorps" => (0, 1, 0x57),
        "unpcklps" => (0, 1, 0x14),
        "unpckhps" => (0, 1, 0x15),
        "addpd" => (0x66, 1, 0x58),
        "subpd" => (0x66, 1, 0x5c),
        "mulpd" => (0x66, 1, 0x59),
        "divpd" => (0x66, 1, 0x5e),
        "minpd" => (0x66, 1, 0x5d),
        "maxpd" => (0x66, 1, 0x5f),
        "sqrtpd" => (0x66, 1, 0x51),
        "andpd" => (0x66, 1, 0x54),
        "andnpd" => (0x66, 1, 0x55),
        "orpd" => (0x66, 1, 0x56),
        "xorpd" => (0x66, 1, 0x57),
        "unpcklpd" => (0x66, 1, 0x14),
        "unpckhpd" => (0x66, 1, 0x15),
        "addsubps" => (0xf2, 1, 0xd0),
        "addsubpd" => (0x66, 1, 0xd0),
        "haddps" => (0xf2, 1, 0x7c),
        "haddpd" => (0x66, 1, 0x7c),
        "hsubps" => (0xf2, 1, 0x7d),
        "hsubpd" => (0x66, 1, 0x7d),
        "movshdup" => (0xf3, 1, 0x16),
        "movsldup" => (0xf3, 1, 0x12),
        "movddup" => (0xf2, 1, 0x12),
        "lddqu" => (0xf2, 1, 0xf0),
        "cvtdq2ps" => (0, 1, 0x5b),
        "cvtps2dq" => (0x66, 1, 0x5b),
        "cvttps2dq" => (0xf3, 1, 0x5b),
        "cvtdq2pd" => (0xf3, 1, 0xe6),
        "cvtpd2dq" => (0xf2, 1, 0xe6),
        "cvttpd2dq" => (0x66, 1, 0xe6),
        "cvtps2pd" => (0, 1, 0x5a),
        "cvtpd2ps" => (0x66, 1, 0x5a),
        "pshufb" => (0x66, 2, 0x00),
        "phaddw" => (0x66, 2, 0x01),
        "phaddd" => (0x66, 2, 0x02),
        "phaddsw" => (0x66, 2, 0x03),
        "pmaddubsw" => (0x66, 2, 0x04),
        "phsubw" => (0x66, 2, 0x05),
        "phsubd" => (0x66, 2, 0x06),
        "phsubsw" => (0x66, 2, 0x07),
        "psignb" => (0x66, 2, 0x08),
        "psignw" => (0x66, 2, 0x09),
        "psignd" => (0x66, 2, 0x0a),
        "pmulhrsw" => (0x66, 2, 0x0b),
        "ptest" => (0x66, 2, 0x17),
        "pabsb" => (0x66, 2, 0x1c),
        "pabsw" => (0x66, 2, 0x1d),
        "pabsd" => (0x66, 2, 0x1e),
        "pmovsxbw" => (0x66, 2, 0x20),
        "pmovsxbd" => (0x66, 2, 0x21),
        "pmovsxbq" => (0x66, 2, 0x22),
        "pmovsxwd" => (0x66, 2, 0x23),
        "pmovsxwq" => (0x66, 2, 0x24),
        "pmovsxdq" => (0x66, 2, 0x25),
        "pmuldq" => (0x66, 2, 0x28),
        "pcmpeqq" => (0x66, 2, 0x29),
        "movntdqa" => (0x66, 2, 0x2a),
        "packusdw" => (0x66, 2, 0x2b),
        "pmovzxbw" => (0x66, 2, 0x30),
        "pmovzxbd" => (0x66, 2, 0x31),
        "pmovzxbq" => (0x66, 2, 0x32),
        "pmovzxwd" => (0x66, 2, 0x33),
        "pmovzxwq" => (0x66, 2, 0x34),
        "pmovzxdq" => (0x66, 2, 0x35),
        "pcmpgtq" => (0x66, 2, 0x37),
        "pminsb" => (0x66, 2, 0x38),
        "pminsd" => (0x66, 2, 0x39),
        "pminuw" => (0x66, 2, 0x3a),
        "pminud" => (0x66, 2, 0x3b),
        "pmaxsb" => (0x66, 2, 0x3c),
        "pmaxsd" => (0x66, 2, 0x3d),
        "pmaxuw" => (0x66, 2, 0x3e),
        "pmaxud" => (0x66, 2, 0x3f),
        "pmulld" => (0x66, 2, 0x40),
        "phminposuw" => (0x66, 2, 0x41),
        "maskmovdqu" => (0x66, 1, 0xf7),
        "aesimc" => (0x66, 2, 0xdb),
        "aesenc" => (0x66, 2, 0xdc),
        "aesenclast" => (0x66, 2, 0xdd),
        "aesdec" => (0x66, 2, 0xde),
        "aesdeclast" => (0x66, 2, 0xdf),
        _ => return None,
    };
    Some(entry)
}

/// Packed operations taking an `imm8`, written `op $imm, xmm/m128, xmm`, as
/// `(legacy prefix, opcode map, opcode byte)`; map 3 is `0f 3a`.
fn packed_sse_imm(name: &str) -> Option<(u8, u8, u8)> {
    let entry = match name {
        "cmpps" => (0, 1, 0xc2),
        "cmppd" => (0x66, 1, 0xc2),
        "roundps" => (0x66, 3, 0x08),
        "roundpd" => (0x66, 3, 0x09),
        "roundss" => (0x66, 3, 0x0a),
        "roundsd" => (0x66, 3, 0x0b),
        "blendps" => (0x66, 3, 0x0c),
        "blendpd" => (0x66, 3, 0x0d),
        "pblendw" => (0x66, 3, 0x0e),
        "palignr" => (0x66, 3, 0x0f),
        "insertps" => (0x66, 3, 0x21),
        "dpps" => (0x66, 3, 0x40),
        "dppd" => (0x66, 3, 0x41),
        "mpsadbw" => (0x66, 3, 0x42),
        "pclmulqdq" => (0x66, 3, 0x44),
        "pcmpestrm" => (0x66, 3, 0x60),
        "pcmpestri" => (0x66, 3, 0x61),
        "pcmpistrm" => (0x66, 3, 0x62),
        "pcmpistri" => (0x66, 3, 0x63),
        "aeskeygenassist" => (0x66, 3, 0xdf),
        _ => return None,
    };
    Some(entry)
}

/// SSE4.1 variable blends, written `op %xmm0, xmm/m128, xmm`, with the mask
/// register fixed at `%xmm0`.
fn packed_blendv(name: &str) -> Option<u8> {
    match name {
        "pblendvb" => Some(0x10),
        "blendvps" => Some(0x14),
        "blendvpd" => Some(0x15),
        _ => None,
    }
}

/// Non-temporal stores, written `op xmm, m128`, as `(legacy prefix, opcode)`.
fn packed_store(name: &str) -> Option<(u8, u8)> {
    match name {
        "movntps" => Some((0, 0x2b)),
        "movntpd" => Some((0x66, 0x2b)),
        "movntdq" => Some((0x66, 0xe7)),
        _ => None,
    }
}

/// Sign-mask extractions, written `op xmm, r32`, as `(legacy prefix, opcode)`.
fn packed_mask_move(name: &str) -> Option<(u8, u8)> {
    match name {
        "movmskps" => Some((0, 0x50)),
        "movmskpd" => Some((0x66, 0x50)),
        "pmovmskb" => Some((0x66, 0xd7)),
        _ => None,
    }
}

/// Shift-by-immediate forms, as `(group opcode, /digit)`.
fn packed_shift_imm(name: &str) -> Option<(u8, u8)> {
    let entry = match name {
        "psrlw" => (0x71, 2),
        "psraw" => (0x71, 4),
        "psllw" => (0x71, 6),
        "psrld" => (0x72, 2),
        "psrad" => (0x72, 4),
        "pslld" => (0x72, 6),
        "psrlq" => (0x73, 2),
        "psrldq" => (0x73, 3),
        "psllq" => (0x73, 6),
        "pslldq" => (0x73, 7),
        _ => return None,
    };
    Some(entry)
}

/// SSE4.1 lane transfers, as `(opcode byte, REX.W, gpr width)`.
fn lane_transfer(name: &str) -> Option<(u8, bool, u8)> {
    let entry = match name {
        "extractps" => (0x17, false, 32),
        "pextrb" => (0x14, false, 32),
        "pextrd" => (0x16, false, 32),
        "pextrq" => (0x16, true, 64),
        "pinsrb" => (0x20, false, 32),
        "pinsrd" => (0x22, false, 32),
        "pinsrq" => (0x22, true, 64),
        _ => return None,
    };
    Some(entry)
}

/// The mnemonics the encoder emits correct 32-bit code for. Protected mode
/// shares every encoding path with long mode, so a mnemonic outside this set
/// could otherwise reach a 32-bit object with a long-mode encoding; it refuses
/// by name instead.
fn protected_mode_supported(opcode: &str) -> bool {
    const PLAIN: &[&str] = &[
        "bswap", "bswapl", "call", "cdq", "cltd", "cwtl", "endbr32", "int", "jmp", "lea", "leal",
        "leave", "mov", "movb", "movl", "movw", "movsbl", "movsbw", "movswl", "movzbl", "movzwl",
        "movsx", "movzx", "nop", "pop", "popl", "push", "pushl", "ret", "ud2", "hlt", "cli",
        "sti", "int3", "imul", "imull",
        "btl", "btsl", "btrl", "btcl", "bsfl", "bsrl", "xchgb", "xchgw", "xchgl", "xaddb", "xaddw",
        "xaddl", "cmpxchgb", "cmpxchgw", "cmpxchgl", "lock",
    ];
    const X87: &[&str] = &[
        "fabs", "fadds", "faddl", "fchs", "fcomi", "fcomip", "fdivs", "fdivl", "fdivp", "fild",
        "fildl", "fildll", "fistpl", "fistpq", "fld", "fld1", "fldcw", "fldl", "flds", "fldt",
        "fldz", "fmul", "fmuls", "fmull", "fmulp", "fnstcw", "fsqrt", "fst", "fstl", "fsts",
        "fstp", "fstpl", "fstps", "fstpt", "fsub", "fsubs", "fsubl", "fsubrp", "fucomip", "fxch",
    ];
    if PLAIN.contains(&opcode) || X87.contains(&opcode) {
        return true;
    }
    if let Some(cc) = opcode.strip_prefix("set") {
        return condition_code(cc).is_some();
    }
    if let Some(cc) = opcode.strip_prefix('j') {
        return condition_code(cc).is_some();
    }
    let narrow = |width: Option<u8>| width != Some(64);
    alu_mnemonic(opcode).is_some_and(|(_, width)| narrow(width))
        || shift_mnemonic(opcode).is_some_and(|(_, width)| narrow(width))
        || unary_mnemonic(opcode).is_some_and(|(_, width)| narrow(width))
        || cmov_mnemonic(opcode).is_some_and(|(_, width)| narrow(width))
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
    let value = if negative { -magnitude } else { magnitude };
    // A bit pattern above i64::MAX, such as a sign mask, is written unsigned.
    i64::try_from(value)
        .or_else(|_| u64::try_from(value).map(|bits| bits as i64))
        .map_err(|_| RasError::EncodingError(format!("Immediate out of range: {}", text)))
}

pub(crate) fn rip_expression(text: &str) -> Option<(&str, i64)> {
    let symbol = |s: &str| is_symbol(s) || is_got_symbol(s) || tls_model(s).is_some();
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

/// A 32-bit general register that exists in 32-bit mode, by its number.
pub(crate) fn gpr32_number(name: &str) -> Option<u8> {
    gpr(name)
        .filter(|reg| reg.width == 32 && reg.num < 8 && !reg.high)
        .map(|reg| reg.num)
}

fn is_got_symbol(text: &str) -> bool {
    text.strip_suffix("@GOTPCREL").is_some_and(is_symbol)
}

/// The thread-local model a `sym@gottpoff` or `sym@tpoff` reference selects.
pub(crate) fn tls_model(text: &str) -> Option<TlsModel> {
    let (name, suffix) = text.rsplit_once('@')?;
    if !is_symbol(name) {
        return None;
    }
    match suffix {
        "gottpoff" => Some(TlsModel::InitialExec),
        "tpoff" => Some(TlsModel::LocalExec),
        "SECREL32" => Some(TlsModel::SecRel),
        _ => None,
    }
}

/// The TLS relocation suffix of an operand that this assembler does not implement.
pub(crate) fn unsupported_tls_suffix(operand: &str) -> Option<&str> {
    let (_, after) = operand.split_once('@')?;
    let suffix = after
        .split(|c: char| !c.is_ascii_alphanumeric())
        .next()
        .unwrap_or_default();
    const UNSUPPORTED: [&str; 7] = [
        "tlsgd",
        "tlsld",
        "dtpoff",
        "gotntpoff",
        "indntpoff",
        "tlsdesc",
        "tlscall",
    ];
    UNSUPPORTED.contains(&suffix).then_some(suffix)
}

pub(crate) fn unsupported_tls_error(suffix: &str) -> RasError {
    RasError::EncodingError(format!(
        "unsupported TLS model @{suffix}: only initial-exec (@gottpoff) and local-exec (@tpoff) are implemented"
    ))
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

fn exchange_mnemonic(opcode: &str) -> Option<(&'static str, u8)> {
    ["xchg", "xadd", "cmpxchg"]
        .into_iter()
        .find_map(|stem| sized_stem(opcode, stem)?.map(|width| (stem, width)))
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

/// A vector register of either width, as its number and whether it is 256-bit.
fn parse_vector(s: &str) -> Option<(u8, bool)> {
    match parse_vreg(s) {
        Some((n, len)) if n < 16 && len < 2 => Some((n, len == 1)),
        _ => None,
    }
}

/// Any vector register, as its number and its length class: 0 for `xmm`,
/// 1 for `ymm`, 2 for `zmm`.
fn parse_vreg(s: &str) -> Option<(u8, u8)> {
    let s = s.trim().trim_start_matches('%');
    let (rest, len) = if let Some(rest) = s.strip_prefix("zmm") {
        (rest, 2u8)
    } else if let Some(rest) = s.strip_prefix("ymm") {
        (rest, 1)
    } else {
        (s.strip_prefix("xmm")?, 0)
    };
    if rest.len() > 1 && rest.starts_with('0') {
        return None;
    }
    let n: u8 = rest.parse().ok()?;
    if n < 32 { Some((n, len)) } else { None }
}

/// An opmask register `%k0` through `%k7`.
fn mask_register(s: &str) -> Option<u8> {
    let rest = s.trim().trim_start_matches('%').strip_prefix('k')?;
    if rest.len() != 1 {
        return None;
    }
    let n: u8 = rest.parse().ok()?;
    if n < 8 { Some(n) } else { None }
}

/// An MMX register `%mm0` through `%mm7`.
fn mmx_register(s: &str) -> Option<u8> {
    let rest = s.trim().trim_start_matches('%').strip_prefix("mm")?;
    if rest.len() != 1 {
        return None;
    }
    let n: u8 = rest.parse().ok()?;
    if n < 8 { Some(n) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evex_table_is_searchable() {
        assert!(
            EVEX_TABLE
                .windows(2)
                .all(|pair| pair[0].name <= pair[1].name),
            "EVEX_TABLE must be sorted by mnemonic"
        );
        for (index, entry) in EVEX_TABLE.iter().enumerate() {
            assert!(entry.lens != 0, "{} has no vector length", entry.name);
            assert!(
                !EVEX_TABLE[index + 1..]
                    .iter()
                    .take_while(|other| other.name == entry.name)
                    .any(|other| other.takes_imm() == entry.takes_imm()),
                "{} is listed twice for the same operand shape",
                entry.name
            );
            assert_eq!(
                packed_evex(entry.name, entry.takes_imm()).map(|found| found.opcode),
                Some(entry.opcode),
                "{} is not reachable",
                entry.name
            );
        }
    }

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

    /// Expected bytes come from GNU as.
    #[test]
    fn avx_encodings_match_gnu_as() {
        let cases: &[(&str, &[&str], &[u8])] = &[
            ("vmovdqa", &["%xmm0", "%xmm1"], &[0xc5, 0xf9, 0x6f, 0xc8]),
            ("vmovdqa", &["%xmm9", "%xmm2"], &[0xc5, 0x79, 0x7f, 0xca]),
            ("vmovdqa", &["%xmm3", "%xmm11"], &[0xc5, 0x79, 0x6f, 0xdb]),
            ("vmovdqa", &["%ymm0", "%ymm1"], &[0xc5, 0xfd, 0x6f, 0xc8]),
            (
                "vmovdqa",
                &["%ymm12", "%ymm13"],
                &[0xc4, 0x41, 0x7d, 0x6f, 0xec],
            ),
            (
                "vmovdqa",
                &["16(%rsp)", "%xmm0"],
                &[0xc5, 0xf9, 0x6f, 0x44, 0x24, 0x10],
            ),
            (
                "vmovdqa",
                &["%ymm15", "128(%r13)"],
                &[0xc4, 0x41, 0x7d, 0x7f, 0xbd, 0x80, 0x00, 0x00, 0x00],
            ),
            (
                "vmovdqu",
                &["%ymm8", "%ymm9"],
                &[0xc4, 0x41, 0x7e, 0x6f, 0xc8],
            ),
            (
                "vmovdqu",
                &["%ymm3", "100(%rax,%rbx,4)"],
                &[0xc5, 0xfe, 0x7f, 0x5c, 0x98, 0x64],
            ),
            ("vmovd", &["%eax", "%xmm0"], &[0xc5, 0xf9, 0x6e, 0xc0]),
            (
                "vmovd",
                &["%xmm8", "%r12d"],
                &[0xc4, 0x41, 0x79, 0x7e, 0xc4],
            ),
            (
                "vpaddd",
                &["%xmm1", "%xmm2", "%xmm3"],
                &[0xc5, 0xe9, 0xfe, 0xd9],
            ),
            (
                "vpaddd",
                &["%ymm9", "%ymm10", "%ymm11"],
                &[0xc4, 0x41, 0x2d, 0xfe, 0xd9],
            ),
            (
                "vpxor",
                &["%ymm7", "%ymm8", "%ymm9"],
                &[0xc5, 0x3d, 0xef, 0xcf],
            ),
            (
                "vpsrldq",
                &["$8", "%xmm0", "%xmm1"],
                &[0xc5, 0xf1, 0x73, 0xd8, 0x08],
            ),
            (
                "vpsrldq",
                &["$4", "%xmm12", "%xmm13"],
                &[0xc4, 0xc1, 0x11, 0x73, 0xdc, 0x04],
            ),
            (
                "vextracti128",
                &["$1", "%ymm0", "%xmm1"],
                &[0xc4, 0xe3, 0x7d, 0x39, 0xc1, 0x01],
            ),
            (
                "vextracti128",
                &["$1", "%ymm2", "16(%rsp)"],
                &[0xc4, 0xe3, 0x7d, 0x39, 0x54, 0x24, 0x10, 0x01],
            ),
            ("vzeroupper", &[], &[0xc5, 0xf8, 0x77]),
        ];
        for (opcode, operands, expected) in cases {
            assert_eq!(
                &encode_one(opcode, operands),
                expected,
                "{opcode} {operands:?}"
            );
        }
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

    #[test]
    fn exchange_popcnt_and_scan_match_sdm_bytes() {
        assert_eq!(
            encode_one("xchgq", &["%rax", "%rcx"]),
            vec![0x48, 0x87, 0xc1]
        );
        assert_eq!(
            encode_one("lock", &["cmpxchgq %r10", "(%r11)"]),
            vec![0xf0, 0x4d, 0x0f, 0xb1, 0x13]
        );
        assert_eq!(
            encode_one("lock", &["xaddb %al", "(%r10)"]),
            vec![0xf0, 0x41, 0x0f, 0xc0, 0x02]
        );
        assert_eq!(
            encode_one("popcntq", &["%rax", "%rax"]),
            vec![0xf3, 0x48, 0x0f, 0xb8, 0xc0]
        );
        assert_eq!(
            encode_one("bsfq", &["%rax", "%rax"]),
            vec![0x48, 0x0f, 0xbc, 0xc0]
        );
        assert_eq!(encode_one("mfence", &[]), vec![0x0f, 0xae, 0xf0]);
        assert_eq!(encode_one("rdtsc", &[]), vec![0x0f, 0x31]);
    }

    /// Bytes derived by hand from the field layouts. Intel SDM Vol. 2:
    /// LZCNT and TZCNT are F3 REX.W 0F BD / BC /r; SHLX, SARX and SHRX are
    /// VEX.LZ.{66,F3,F2}.0F38.W 'F7 /r'; VPMULLQ is EVEX.128.66.0F38.W1 40 /r.
    #[test]
    fn bit_count_and_bmi2_match_sdm_bytes() {
        assert_eq!(
            encode_one("lzcntq", &["%rax", "%rax"]),
            vec![0xf3, 0x48, 0x0f, 0xbd, 0xc0]
        );
        assert_eq!(
            encode_one("tzcntq", &["%rax", "%rax"]),
            vec![0xf3, 0x48, 0x0f, 0xbc, 0xc0]
        );
        assert_eq!(
            encode_one("lzcntl", &["%ecx", "%edx"]),
            vec![0xf3, 0x0f, 0xbd, 0xd1]
        );
        assert_eq!(
            encode_one("shlxq", &["%r12", "%r8", "%r8"]),
            vec![0xc4, 0x42, 0x99, 0xf7, 0xc0]
        );
        assert_eq!(
            encode_one("sarxq", &["%rax", "%r15", "%r15"]),
            vec![0xc4, 0x42, 0xfa, 0xf7, 0xff]
        );
        assert_eq!(
            encode_one("shrxl", &["%ecx", "%edx", "%eax"]),
            vec![0xc4, 0xe2, 0x73, 0xf7, 0xc2]
        );
        assert_eq!(
            encode_one("vpmullq", &["%xmm2", "%xmm0", "%xmm1"]),
            vec![0x62, 0xf2, 0xfd, 0x08, 0x40, 0xca]
        );
    }

    fn encode_with_fixup(opcode: &str, operands: &[&str]) -> (Vec<u8>, Option<RipFixup>) {
        let inst = ParsedInstruction {
            opcode: opcode.to_string(),
            operands: operands.iter().map(|op| op.to_string()).collect(),
        };
        let mut encoder = X86_64Encoder::new();
        let bytes = encoder.encode_instruction(&inst).expect("encode");
        (bytes, encoder.take_rip_fixup())
    }

    #[test]
    fn initial_exec_load_is_rip_relative_gottpoff() {
        assert_eq!(
            encode_with_fixup("movq", &["tvar@gottpoff(%rip)", "%rax"]),
            (
                vec![0x48, 0x8b, 0x05, 0, 0, 0, 0],
                Some(RipFixup {
                    disp_offset: 3,
                    trailing: 0,
                    tls: Some(TlsModel::InitialExec),
                })
            )
        );
    }

    #[test]
    fn fs_relative_load_through_register() {
        assert_eq!(
            encode_with_fixup("movq", &["%fs:(%rax)", "%rdx"]),
            (vec![0x64, 0x48, 0x8b, 0x10], None)
        );
    }

    #[test]
    fn fs_absolute_zero_is_disp32_sib() {
        assert_eq!(
            encode_with_fixup("movq", &["%fs:0", "%rax"]),
            (vec![0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0], None)
        );
        assert_eq!(
            encode_one("movq", &["%fs:0x0", "%rax"]),
            vec![0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0]
        );
    }

    #[test]
    fn local_exec_lea_is_base_relative_tpoff32() {
        assert_eq!(
            encode_with_fixup("leaq", &["tvar@tpoff(%rax)", "%rdx"]),
            (
                vec![0x48, 0x8d, 0x90, 0, 0, 0, 0],
                Some(RipFixup {
                    disp_offset: 3,
                    trailing: 0,
                    tls: Some(TlsModel::LocalExec),
                })
            )
        );
    }

    #[test]
    fn local_exec_fs_absolute_tpoff32() {
        assert_eq!(
            encode_with_fixup("movq", &["%fs:tvar@tpoff+8", "%rax"]),
            (
                vec![0x64, 0x48, 0x8b, 0x04, 0x25, 8, 0, 0, 0],
                Some(RipFixup {
                    disp_offset: 5,
                    trailing: 0,
                    tls: Some(TlsModel::LocalExec),
                })
            )
        );
    }

    #[test]
    fn tls_operands_reject_wrong_addressing() {
        let encode = |opcode: &str, operands: &[&str]| {
            X86_64Encoder::new()
                .encode_instruction(&ParsedInstruction {
                    opcode: opcode.to_string(),
                    operands: operands.iter().map(|op| op.to_string()).collect(),
                })
                .map_err(|e| e.to_string())
        };
        assert!(encode("movq", &["tvar@gottpoff(%rax)", "%rdx"]).is_err());
        assert!(encode("movq", &["tvar@tpoff(%rip)", "%rdx"]).is_err());
        assert!(encode("movq", &["tvar@tpoff", "%rdx"]).is_err());
    }

    #[test]
    fn unsupported_tls_models_are_named() {
        let err = X86_64Encoder::new()
            .encode_instruction(&ParsedInstruction {
                opcode: "leaq".into(),
                operands: vec!["tvar@tlsgd(%rip)".into(), "%rdi".into()],
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported TLS model @tlsgd"), "{err}");
        let err = X86_64Encoder::new()
            .encode_instruction(&ParsedInstruction {
                opcode: "movq".into(),
                operands: vec!["tvar@dtpoff(%rax)".into(), "%rdx".into()],
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported TLS model @dtpoff"), "{err}");
    }

    #[test]
    fn segment_override_selects_gs_prefix_for_absolute_operands() {
        assert_eq!(
            encode_one("movq", &["%gs:0x58", "%rax"]),
            vec![0x65, 0x48, 0x8b, 0x04, 0x25, 0x58, 0, 0, 0]
        );
        assert_eq!(
            encode_one("movq", &["%rdx", "%gs:8"]),
            vec![0x65, 0x48, 0x89, 0x14, 0x25, 8, 0, 0, 0]
        );
        assert_eq!(
            encode_one("movq", &["%gs:(%rax)", "%rdx"]),
            vec![0x65, 0x48, 0x8b, 0x10]
        );
        assert_eq!(
            encode_one("movq", &["%fs:0x28", "%rdx"]),
            vec![0x64, 0x48, 0x8b, 0x14, 0x25, 0x28, 0, 0, 0]
        );
    }

    #[test]
    fn gs_operands_refuse_tls_and_symbol_forms_they_cannot_encode() {
        let encode = |operands: &[&str]| {
            X86_64Encoder::new()
                .encode_instruction(&ParsedInstruction {
                    opcode: "movq".into(),
                    operands: operands.iter().map(|op| op.to_string()).collect(),
                })
                .map_err(|e| e.to_string())
        };
        assert!(encode(&["%gs:tvar@tpoff", "%rax"]).is_err());
        assert!(encode(&["%gs:tvar@gottpoff(%rip)", "%rax"]).is_err());
        assert!(encode(&["%gs:tvar", "%rax"]).is_err());
    }

    #[test]
    fn secrel32_displacement_is_base_relative_with_tls_fixup() {
        assert_eq!(
            encode_with_fixup("movl", &["_x@SECREL32(%rax)", "%eax"]),
            (
                vec![0x8b, 0x80, 0, 0, 0, 0],
                Some(RipFixup {
                    disp_offset: 2,
                    trailing: 0,
                    tls: Some(TlsModel::SecRel),
                })
            )
        );
    }

    #[test]
    fn secrel32_with_addend_and_sib_base_keeps_fixup_offset() {
        assert_eq!(
            encode_with_fixup("movl", &["_x@SECREL32+8(%rsp)", "%ecx"]),
            (
                vec![0x8b, 0x8c, 0x24, 8, 0, 0, 0],
                Some(RipFixup {
                    disp_offset: 3,
                    trailing: 0,
                    tls: Some(TlsModel::SecRel),
                })
            )
        );
    }

    #[test]
    fn secrel32_refuses_operands_without_a_plain_base() {
        let encode = |operands: &[&str]| {
            X86_64Encoder::new()
                .encode_instruction(&ParsedInstruction {
                    opcode: "movl".into(),
                    operands: operands.iter().map(|op| op.to_string()).collect(),
                })
                .map_err(|e| e.to_string())
        };
        assert!(encode(&["_x@SECREL32(%rip)", "%eax"]).is_err());
        assert!(encode(&["_x@SECREL32", "%eax"]).is_err());
        assert!(encode(&["%gs:_x@SECREL32(%rax)", "%eax"]).is_err());
        assert!(encode(&["%fs:_x@SECREL32(%rax)", "%eax"]).is_err());
    }

    /// The 32-bit x86 encoder and the ELF32 objects it writes.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[cfg(feature = "encoder")]
    mod i686 {
        use crate::Ras;
        use crate::test_support::TempDir;
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
        use std::process::Command;

        fn u16_at(bytes: &[u8], offset: usize) -> u16 {
            u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
        }

        fn u32_at(bytes: &[u8], offset: usize) -> u32 {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
        }

        struct Object {
            bytes: Vec<u8>,
        }

        impl Object {
            fn assemble(label: &str, asm: &str) -> Self {
                let tmp = TempDir::new(label);
                let path = tmp.path().join("out.o");
                Ras::new(TargetArchitecture::I686, TargetOperatingSystem::Linux)
                    .unwrap()
                    .assemble(asm, &path)
                    .unwrap();
                Self {
                    bytes: std::fs::read(&path).unwrap(),
                }
            }

            fn header_count(&self) -> usize {
                u16_at(&self.bytes, 48) as usize
            }

            fn header(&self, index: usize) -> usize {
                u32_at(&self.bytes, 32) as usize + index * 40
            }

            fn names(&self) -> &[u8] {
                let strings = u16_at(&self.bytes, 50) as usize;
                &self.bytes[u32_at(&self.bytes, self.header(strings) + 16) as usize..]
            }

            fn find(&self, name: &str) -> Option<usize> {
                (1..self.header_count()).find(|index| {
                    let at = u32_at(&self.bytes, self.header(*index)) as usize;
                    self.names()[at..].split(|b| *b == 0).next().unwrap() == name.as_bytes()
                })
            }

            fn section(&self, name: &str) -> &[u8] {
                let header = self.header(self.find(name).unwrap_or_else(|| panic!("no {name}")));
                let start = u32_at(&self.bytes, header + 16) as usize;
                &self.bytes[start..start + u32_at(&self.bytes, header + 20) as usize]
            }

            fn field(&self, name: &str, index: usize) -> u32 {
                u32_at(
                    &self.bytes,
                    self.header(self.find(name).unwrap()) + index * 4,
                )
            }

            /// Every relocation of `name` as (offset, symbol index, type, field value).
            fn relocations(&self, name: &str) -> Vec<(u32, u32, u32, i64)> {
                let target = &name[".rel".len()..];
                let applies_to = self.section(target).to_vec();
                self.section(name)
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|entry| {
                        let offset = u32_at(entry, 0);
                        let info = u32_at(entry, 4);
                        let kind = info & 0xff;
                        let width = match kind {
                            1 | 2 | 4 => 4,
                            20 => 2,
                            _ => 1,
                        };
                        let at = offset as usize;
                        let mut value = [0u8; 8];
                        value[..width].copy_from_slice(&applies_to[at..at + width]);
                        let mut stored = i64::from_le_bytes(value);
                        let bits = width * 8;
                        if stored >> (bits - 1) & 1 == 1 {
                            stored -= 1i64 << bits;
                        }
                        (offset, info >> 8, kind, stored)
                    })
                    .collect()
            }

            fn symbol_names(&self) -> Vec<(String, u32, u16)> {
                let strings = self.section(".strtab").to_vec();
                self.section(".symtab")
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .map(|entry| {
                        let at = u32_at(entry, 0) as usize;
                        let name = strings[at..].split(|b| *b == 0).next().unwrap();
                        (
                            String::from_utf8_lossy(name).into_owned(),
                            u32_at(entry, 4),
                            u16_at(entry, 14),
                        )
                    })
                    .collect()
            }
        }

        fn encode(line: &str) -> Vec<u8> {
            Object::assemble("i686-encode", &format!(".text\n.globl m\nm:\n    {line}\n"))
                .section(".text")
                .to_vec()
        }

        fn error(line: &str) -> String {
            let tmp = TempDir::new("i686-refuse");
            Ras::new(TargetArchitecture::I686, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(
                    &format!(".text\n.globl m\nm:\n    {line}\n"),
                    &tmp.path().join("out.o"),
                )
                .unwrap_err()
                .to_string()
        }

        /// Every mnemonic and operand shape lamina's i686 backend emits.
        const ENCODINGS: &[(&str, &[u8])] = &[
            ("pushl %ebp", &[0x55]),
            ("pushl %ebx", &[0x53]),
            ("pushl $7", &[0x6a, 0x07]),
            ("popl %ebx", &[0x5b]),
            ("movl %esp, %ebp", &[0x89, 0xe5]),
            ("subl $648, %esp", &[0x81, 0xec, 0x88, 0x02, 0x00, 0x00]),
            ("addl $8, %esp", &[0x83, 0xc4, 0x08]),
            ("movl $30, %eax", &[0xb8, 0x1e, 0x00, 0x00, 0x00]),
            ("movl %eax, -8(%ebp)", &[0x89, 0x45, 0xf8]),
            ("movl -8(%ebp), %ecx", &[0x8b, 0x4d, 0xf8]),
            ("movb %al, -9(%ebp)", &[0x88, 0x45, 0xf7]),
            ("movb -9(%ebp), %al", &[0x8a, 0x45, 0xf7]),
            ("movw %ax, -12(%ebp)", &[0x66, 0x89, 0x45, 0xf4]),
            ("xorl %eax, %eax", &[0x31, 0xc0]),
            ("addl %ecx, %eax", &[0x01, 0xc8]),
            ("andl %ecx, %eax", &[0x21, 0xc8]),
            ("orl %ecx, %eax", &[0x09, 0xc8]),
            ("subl %ecx, %eax", &[0x29, 0xc8]),
            ("cmpl %ecx, %eax", &[0x39, 0xc8]),
            ("testl %eax, %eax", &[0x85, 0xc0]),
            ("andb %cl, %al", &[0x20, 0xc8]),
            ("orb %cl, %al", &[0x08, 0xc8]),
            ("andl $65535, %eax", &[0x25, 0xff, 0xff, 0x00, 0x00]),
            ("cmpl $-1, %ecx", &[0x83, 0xf9, 0xff]),
            ("imull %ecx, %eax", &[0x0f, 0xaf, 0xc1]),
            ("imull $3, %eax, %eax", &[0x6b, 0xc0, 0x03]),
            ("idivl %ecx", &[0xf7, 0xf9]),
            ("divl %ecx", &[0xf7, 0xf1]),
            ("cltd", &[0x99]),
            ("shrl %cl, %eax", &[0xd3, 0xe8]),
            ("shll %cl, %eax", &[0xd3, 0xe0]),
            ("sarl %cl, %eax", &[0xd3, 0xf8]),
            ("rorl %cl, %eax", &[0xd3, 0xc8]),
            ("rolb %cl, %al", &[0xd2, 0xc0]),
            ("rolw $3, %ax", &[0x66, 0xc1, 0xc0, 0x03]),
            ("shrl $16, %eax", &[0xc1, 0xe8, 0x10]),
            ("cmovel %ecx, %eax", &[0x0f, 0x44, 0xc1]),
            ("cmovsl %ecx, %eax", &[0x0f, 0x48, 0xc1]),
            ("cmoval %edx, %ecx", &[0x0f, 0x47, 0xca]),
            ("sete %al", &[0x0f, 0x94, 0xc0]),
            ("setne %al", &[0x0f, 0x95, 0xc0]),
            ("seta %al", &[0x0f, 0x97, 0xc0]),
            ("setae %al", &[0x0f, 0x93, 0xc0]),
            ("setb %al", &[0x0f, 0x92, 0xc0]),
            ("setbe %al", &[0x0f, 0x96, 0xc0]),
            ("setg %al", &[0x0f, 0x9f, 0xc0]),
            ("setl %al", &[0x0f, 0x9c, 0xc0]),
            ("setp %cl", &[0x0f, 0x9a, 0xc1]),
            ("setnp %cl", &[0x0f, 0x9b, 0xc1]),
            ("movsbl %al, %eax", &[0x0f, 0xbe, 0xc0]),
            ("movsbl -9(%ebp), %eax", &[0x0f, 0xbe, 0x45, 0xf7]),
            ("movswl %ax, %eax", &[0x0f, 0xbf, 0xc0]),
            ("movswl -12(%ebp), %eax", &[0x0f, 0xbf, 0x45, 0xf4]),
            ("movzbl %al, %eax", &[0x0f, 0xb6, 0xc0]),
            ("movzbl -9(%ebp), %ecx", &[0x0f, 0xb6, 0x4d, 0xf7]),
            ("movzwl %ax, %eax", &[0x0f, 0xb7, 0xc0]),
            ("bsfl %eax, %edx", &[0x0f, 0xbc, 0xd0]),
            ("bsrl %eax, %edx", &[0x0f, 0xbd, 0xd0]),
            ("bswapl %eax", &[0x0f, 0xc8]),
            ("leal -8(%ebp), %eax", &[0x8d, 0x45, 0xf8]),
            ("leal 16(%esp,%ecx,4), %eax", &[0x8d, 0x44, 0x8c, 0x10]),
            ("call *32(%esp)", &[0xff, 0x54, 0x24, 0x20]),
            ("flds -8(%ebp)", &[0xd9, 0x45, 0xf8]),
            ("fldl -16(%ebp)", &[0xdd, 0x45, 0xf0]),
            ("fildl -8(%ebp)", &[0xdb, 0x45, 0xf8]),
            ("fildll -16(%ebp)", &[0xdf, 0x6d, 0xf0]),
            ("fstps -8(%ebp)", &[0xd9, 0x5d, 0xf8]),
            ("fstpl -16(%ebp)", &[0xdd, 0x5d, 0xf0]),
            ("fadds -8(%ebp)", &[0xd8, 0x45, 0xf8]),
            ("faddl -16(%ebp)", &[0xdc, 0x45, 0xf0]),
            ("fmuls -8(%ebp)", &[0xd8, 0x4d, 0xf8]),
            ("fmull -16(%ebp)", &[0xdc, 0x4d, 0xf0]),
            ("fsubl -16(%ebp)", &[0xdc, 0x65, 0xf0]),
            ("fdivl -16(%ebp)", &[0xdc, 0x75, 0xf0]),
            ("fchs", &[0xd9, 0xe0]),
            ("fsqrt", &[0xd9, 0xfa]),
            ("fstp %st(0)", &[0xdd, 0xd8]),
            ("fucomip %st(1), %st", &[0xdf, 0xe9]),
            ("int $0x80", &[0xcd, 0x80]),
            ("ud2", &[0x0f, 0x0b]),
            ("hlt", &[0xf4]),
            ("cli", &[0xfa]),
            ("sti", &[0xfb]),
            ("int3", &[0xcc]),
            ("pause", &[0xf3, 0x90]),
            ("cpuid", &[0x0f, 0xa2]),
            ("lfence", &[0x0f, 0xae, 0xe8]),
            ("sfence", &[0x0f, 0xae, 0xf8]),
            ("leave", &[0xc9]),
            ("ret", &[0xc3]),
            ("ret $8", &[0xc2, 0x08, 0x00]),
        ];

        #[test]
        fn encodes_the_instruction_set_lamina_emits() {
            for (line, expected) in ENCODINGS {
                assert_eq!(&encode(line), expected, "{line}");
            }
        }

        /// Nothing in 32-bit mode may carry a REX prefix, and no encoding may reach
        /// for a register or an addressing mode that only long mode has.
        #[test]
        fn refuses_long_mode_forms_by_name() {
            for (line, fragment) in [
                ("addq %rax, %rbx", "addq is not implemented"),
                ("pushq %rax", "pushq is not implemented"),
                ("cltq", "cltq is not implemented"),
                ("syscall", "syscall is not implemented"),
                ("movslq %eax, %ebx", "movslq is not implemented"),
                ("movabsq $1, %rax", "movabsq is not implemented"),
                ("movsd %xmm0, %xmm1", "movsd is not implemented"),
                ("vaddps %ymm0, %ymm1, %ymm2", "vaddps is not implemented"),
                ("leaq sym(%rip), %eax", "leaq is not implemented"),
                ("movl %r8d, %eax", "%r8d does not exist in 32-bit mode"),
                ("addl %rax, %ebx", "%rax does not exist in 32-bit mode"),
                ("movl 4(%rax), %eax", "Invalid memory operand"),
            ] {
                let message = error(line);
                assert!(message.contains(fragment), "{line}: {message}");
            }
        }

        #[test]
        fn branches_resolve_locally_and_relocate_across_symbols() {
            let object = Object::assemble(
                "i686-branch",
                ".text\n.globl m\nm:\n  je 1f\n  jne other\n  jmp 1f\n1:\n  call other\n  ret\n",
            );
            assert_eq!(
                object.section(".text"),
                [
                    // je .+2 and jmp .+0 reach a local label as short jumps.
                    0x74, 0x08, 0x0f, 0x85, 0xfc, 0xff, 0xff, 0xff, 0xeb, 0x00, 0xe8, 0xfc, 0xff,
                    0xff, 0xff, 0xc3
                ]
            );
            let relocations = object.relocations(".rel.text");
            let symbols = object.symbol_names();
            assert_eq!(
                relocations
                    .iter()
                    .map(|(offset, _, kind, field)| (*offset, *kind, *field))
                    .collect::<Vec<_>>(),
                [(4, 2, -4), (11, 2, -4)],
                "both references to `other` are R_386_PC32 with the addend in the field"
            );
            for (_, index, _, _) in &relocations {
                assert_eq!(symbols[*index as usize].0, "other");
            }
        }

        #[test]
        fn symbol_immediates_and_data_pointers_become_absolute_relocations() {
            let object = Object::assemble(
                "i686-abs",
                ".data\ntext: .ascii \"abcde\"\nptr: .long text+2\n.text\n.globl m\nm:\n  movl $text, %eax\n  ret\n",
            );
            assert_eq!(object.section(".text"), [0xb8, 0, 0, 0, 0, 0xc3]);
            let text = object.relocations(".rel.text");
            assert_eq!(text.len(), 1);
            assert_eq!((text[0].0, text[0].2, text[0].3), (1, 1, 0));
            let data = object.relocations(".rel.data");
            assert_eq!(data.len(), 1);
            assert_eq!(
                (data[0].0, data[0].2, data[0].3),
                (5, 1, 2),
                "`.long text+2` keeps its addend in the field an Elf32_Rel patches"
            );
        }

        #[test]
        fn writes_an_elf32_object() {
            let object = Object::assemble(
                "i686-elf",
                ".data\nv: .long 1\n.text\n.globl m\nm:\n  call other\n  ret\n",
            );
            assert_eq!(&object.bytes[..4], b"\x7fELF");
            assert_eq!(object.bytes[4], 1, "ELFCLASS32");
            assert_eq!(u16_at(&object.bytes, 18), 3, "EM_386");
            assert_eq!(u16_at(&object.bytes, 40), 52, "e_ehsize");
            assert_eq!(u16_at(&object.bytes, 46), 40, "e_shentsize");

            assert!(object.find(".rela.text").is_none(), "i386 uses Elf32_Rel");
            assert_eq!(object.field(".rel.text", 1), 9, "SHT_REL");
            assert_eq!(object.field(".rel.text", 9), 8, "sizeof(Elf32_Rel)");
            assert_eq!(object.field(".symtab", 9), 16, "sizeof(Elf32_Sym)");

            // Elf32_Sym orders its fields name, value, size, info, other, shndx; the
            // defined symbol must land in .text with its value at offset 4.
            let text = object.find(".text").unwrap() as u16;
            let symbols = object.symbol_names();
            assert!(symbols.contains(&("m".to_string(), 0, text)), "{symbols:?}");
            assert!(
                symbols
                    .iter()
                    .any(|(name, value, index)| name == "other" && *value == 0 && *index == 0),
                "{symbols:?}"
            );
        }

        /// The decisive check: an object ras assembled, linked and run.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn a_linked_program_runs_and_prints() {
            let tmp = TempDir::new("i686-run");
            let probe = Command::new("cc")
                .args(["-m32", "-xc", "-", "-o"])
                .arg(tmp.path().join("probe"))
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .and_then(|mut child| {
                    use std::io::Write;
                    child
                        .stdin
                        .take()
                        .unwrap()
                        .write_all(b"int main(void){return 0;}")?;
                    child.wait()
                });
            if !probe.is_ok_and(|status| status.success()) {
                return;
            }

            let object = tmp.path().join("program.o");
            Ras::new(TargetArchitecture::I686, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(
                    ".data\nmessage: .asciz \"i686 %d\\n\"\n\
                     .text\n.globl main\nmain:\n\
                     \x20 pushl %ebp\n  movl %esp, %ebp\n  subl $8, %esp\n\
                     \x20 movl $20, %eax\n  movl $2, %ecx\n  imull %ecx, %eax\n  addl $2, %eax\n\
                     \x20 movl %eax, 4(%esp)\n  movl $message, %eax\n  movl %eax, (%esp)\n\
                     \x20 call printf\n  xorl %eax, %eax\n  leave\n  ret\n",
                    &object,
                )
                .unwrap();

            let program = tmp.path().join("program");
            let link = Command::new("cc")
                .args(["-m32", "-no-pie"])
                .arg(&object)
                .arg("-o")
                .arg(&program)
                .output()
                .unwrap();
            assert!(
                link.status.success(),
                "{}",
                String::from_utf8_lossy(&link.stderr)
            );
            let run = Command::new(&program).output().unwrap();
            assert!(run.status.success(), "program exited {:?}", run.status);
            assert_eq!(String::from_utf8_lossy(&run.stdout), "i686 42\n");
        }
    }

    /// x86 thread-local relocations.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[cfg(feature = "encoder")]
    mod tls {
        use crate::Ras;
        use crate::test_support::TempDir;
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

        fn u16_at(bytes: &[u8], offset: usize) -> usize {
            u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize
        }

        fn u32_at(bytes: &[u8], offset: usize) -> u32 {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
        }

        fn u64_at(bytes: &[u8], offset: usize) -> usize {
            u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize
        }

        fn cstr(bytes: &[u8], offset: usize) -> &str {
            let end = bytes[offset..].iter().position(|b| *b == 0).unwrap();
            std::str::from_utf8(&bytes[offset..offset + end]).unwrap()
        }

        struct Header {
            name: String,
            kind: u32,
            flags: u64,
            offset: usize,
            size: usize,
            link: usize,
            entsize: usize,
        }

        fn headers(bytes: &[u8]) -> Vec<Header> {
            let table = u64_at(bytes, 40);
            let count = u16_at(bytes, 60);
            let names = u16_at(bytes, 62);
            let names_at = u64_at(bytes, table + names * 64 + 24);
            (0..count)
                .map(|i| {
                    let h = table + i * 64;
                    Header {
                        name: cstr(bytes, names_at + u32_at(bytes, h) as usize).to_string(),
                        kind: u32_at(bytes, h + 4),
                        flags: u64_at(bytes, h + 8) as u64,
                        offset: u64_at(bytes, h + 24),
                        size: u64_at(bytes, h + 32),
                        link: u32_at(bytes, h + 40) as usize,
                        entsize: u64_at(bytes, h + 56),
                    }
                })
                .collect()
        }

        /// `(offset, kind, symbol, addend)` for every relocation in `.rela.text`.
        fn text_relocations(bytes: &[u8]) -> Vec<(u64, u32, String, i64)> {
            let all = headers(bytes);
            let rela = all.iter().find(|h| h.name == ".rela.text").unwrap();
            let symtab = &all[rela.link];
            let strtab = &all[symtab.link];
            (0..rela.size / rela.entsize)
                .map(|i| {
                    let at = rela.offset + i * 24;
                    let offset = u64_at(bytes, at) as u64;
                    let info = u64_at(bytes, at + 8);
                    let addend = u64_at(bytes, at + 16) as i64;
                    let sym = symtab.offset + (info >> 32) * 24;
                    let name = cstr(bytes, strtab.offset + u32_at(bytes, sym) as usize).to_string();
                    (offset, (info & 0xffff_ffff) as u32, name, addend)
                })
                .collect()
        }

        fn assemble(name: &str, source: &str) -> Vec<u8> {
            let tmp = TempDir::new(&format!("x86_tls_{name}"));
            let path = tmp.path().join(format!("x86_tls_{name}.o"));
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(source, &path)
                .unwrap();
            std::fs::read(&path).unwrap()
        }

        #[test]
        fn initial_and_local_exec_sequences_emit_tls_relocations() {
            let bytes = assemble(
                "initial_local_exec",
                "movq tvar@gottpoff(%rip), %rax\n\
                 movq %fs:(%rax), %rdx\n\
                 movq %fs:0, %rax\n\
                 leaq tvar@tpoff(%rax), %rdx\n\
                 movq %fs:tzero@tpoff, %rcx\n",
            );
            let text = headers(&bytes)
                .into_iter()
                .find(|h| h.name == ".text")
                .unwrap();
            assert_eq!(
                &bytes[text.offset..text.offset + text.size],
                &[
                    0x48, 0x8b, 0x05, 0, 0, 0, 0, // movq tvar@gottpoff(%rip), %rax
                    0x64, 0x48, 0x8b, 0x10, // movq %fs:(%rax), %rdx
                    0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0, // movq %fs:0, %rax
                    0x48, 0x8d, 0x90, 0, 0, 0, 0, // leaq tvar@tpoff(%rax), %rdx
                    0x64, 0x48, 0x8b, 0x0c, 0x25, 0, 0, 0, 0, // movq %fs:tzero@tpoff, %rcx
                ][..]
            );
            assert_eq!(
                text_relocations(&bytes),
                vec![
                    (3, 22, "tvar".to_string(), -4),
                    (23, 23, "tvar".to_string(), 0),
                    (32, 23, "tzero".to_string(), 0),
                ]
            );
        }

        #[test]
        fn unsupported_tls_model_is_an_error() {
            let tmp = TempDir::new("x86_tls_bad");
            let path = tmp.path().join("x86_tls_bad.o");
            let result = Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble("leaq tvar@tlsgd(%rip), %rdi\n", &path);
            let message = format!("{:?}", result.unwrap_err());
            assert!(
                message.contains("unsupported TLS model @tlsgd"),
                "{message}"
            );
        }

        #[test]
        fn tls_section_and_symbol_flags() {
            let bytes = assemble(
                "section_flags",
                ".section .tdata,\"awT\",@progbits\n\
                 .globl tvar\n\
                 .type tvar,@tls_object\n\
                 tvar: .quad 0x1122334455667788\n\
                 .section .tbss,\"awT\",@nobits\n\
                 .globl tzero\n\
                 .type tzero,@tls_object\n\
                 tzero: .zero 16\n\
                 .text\n\
                 .globl _start\n\
                 _start: ret\n\
                 .section .data\n\
                 ptr: .quad tvar@tpoff\n",
            );
            let all = headers(&bytes);
            let tdata = all.iter().find(|h| h.name == ".tdata").unwrap();
            assert_eq!(tdata.flags & 0x400, 0x400);
            assert_eq!(tdata.kind, 1);
            assert_eq!(tdata.size, 8);
            let tbss = all.iter().find(|h| h.name == ".tbss").unwrap();
            assert_eq!(tbss.flags & 0x400, 0x400);
            assert_eq!(tbss.kind, 8);
            assert_eq!(tbss.size, 16);
            let data_rela = all.iter().find(|h| h.name == ".rela.data").unwrap();
            assert_eq!(u64_at(&bytes, data_rela.offset + 8) as u32, 18);
        }
    }

    /// These compare ras against GNU binutils and clang on ELF objects, so they
    /// only run where both the reference assembler and the format are available.
    #[cfg(target_os = "linux")]
    #[cfg(feature = "encoder")]
    mod oracle {
        use crate::Ras;
        use crate::test_support::TempDir;
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
        use std::process::Command;

        fn text(bytes: &[u8]) -> &[u8] {
            let u64_at = |at| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
            let headers = u64_at(40);
            let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
            let strings = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;
            let names = &bytes[u64_at(headers + strings * 64 + 24)..];
            for i in 1..count {
                let h = headers + i * 64;
                let name = u32::from_le_bytes(bytes[h..h + 4].try_into().unwrap()) as usize;
                if names[name..].starts_with(b".text\0") {
                    return &bytes[u64_at(h + 24)..u64_at(h + 24) + u64_at(h + 32)];
                }
            }
            panic!("Missing .text")
        }

        /// Length of the nop starting at `bytes`, covering the `66` and `cs` prefixed
        /// forms GNU as pads with, or `None` when it does not start with one.
        fn nop_length(bytes: &[u8]) -> Option<usize> {
            let mut at = 0;
            while bytes.get(at) == Some(&0x66) {
                at += 1;
            }
            if bytes.get(at) == Some(&0x2e) {
                at += 1;
            }
            match bytes.get(at)? {
                0x90 => Some(at + 1),
                0x0f if bytes.get(at + 1) == Some(&0x1f) => {
                    let modrm = *bytes.get(at + 2)?;
                    if (modrm >> 3) & 7 != 0 {
                        return None;
                    }
                    let (mode, rm) = (modrm >> 6, modrm & 7);
                    let mut length = at + 3 + usize::from(mode != 3 && rm == 4);
                    length += match mode {
                        1 => 1,
                        2 => 4,
                        0 if rm == 5 => 4,
                        _ => 0,
                    };
                    (length <= bytes.len()).then_some(length)
                }
                _ => None,
            }
        }

        /// Rewrite every nop as that many one-byte nops. Which forms an assembler pads
        /// with, and in what order, differs between binutils releases; where the
        /// padding starts and ends does not, and that is what these comparisons are
        /// about.
        fn flatten_nops(text: &[u8]) -> Vec<u8> {
            let mut out = Vec::with_capacity(text.len());
            let mut at = 0;
            while at < text.len() {
                match nop_length(&text[at..]) {
                    Some(length) => {
                        out.resize(out.len() + length, 0x90);
                        at += length;
                    }
                    None => {
                        out.push(text[at]);
                        at += 1;
                    }
                }
            }
            out
        }

        #[test]
        fn padding_comparison_ignores_nop_form() {
            let long_first = [
                0xc3, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0, 0x0f, 0x1f, 0x40, 0x00,
                0xc3,
            ];
            let short_first = [
                0xc3, 0x0f, 0x1f, 0x40, 0x00, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0,
                0xc3,
            ];
            assert_eq!(flatten_nops(&long_first), flatten_nops(&short_first));
            let shorter = [0xc3, 0x0f, 0x1f, 0x40, 0x00, 0xc3];
            assert_ne!(flatten_nops(&long_first), flatten_nops(&shorter));
        }

        #[test]
        fn move_widths_extensions_and_scalar_operations_match_gas() {
            let tmp = TempDir::new("x86_oracle");
            let dir = tmp.path();
            let source = dir.join("input.s");
            let reference = dir.join("reference.o");
            let output = dir.join("ras.o");
            let mut asm = String::from(".text\nendbr64\n");
            for (width, reg) in [("b", "%al"), ("w", "%ax"), ("l", "%eax"), ("q", "%rax")] {
                for base in ["%rsp", "%rbp", "%r12", "%r13", "%r11"] {
                    for offset in [0, 8, -8, 256, -256] {
                        asm.push_str(&format!("mov{width} {reg}, {offset}({base})\nmov{width} {offset}({base}), {reg}\nmov{width} $42, {offset}({base})\n"));
                    }
                }
            }
            asm.push_str("movl %r9d, %r10d\nmovw %r9w, %r10w\nmovb %r9b, %r10b\nmovb %ah, %cl\nmovb %spl, %dil\nmovq $42, %rax\nmovq $4294967295, %rax\nmovq $-42, %r9\nmovzbl (%rsp), %ecx\nmovzbq (%rax), %r9\nmovzwq (%r13), %r10\nmovzwl 8(%r12), %ecx\ncmovel %eax, %ecx\ncmovel %r9d, %r10d\nmulss %xmm1, %xmm0\nmulsd %xmm10, %xmm9\nsqrtsd %xmm1, %xmm0\nsqrtss %xmm11, %xmm12\nminsd %xmm2, %xmm3\nminss %xmm4, %xmm5\nmaxsd %xmm13, %xmm14\nmaxss %xmm6, %xmm7\nmovq $0x8000000000000000, %rax\nret\n");
            asm.push_str(".globl callee\ncallee: call callee\ncall .Llocal\n.Llocal: ret\n");
            for padding in 1..32 {
                asm.push_str(".p2align 5\n");
                for _ in 0..32 - padding {
                    asm.push_str("ret\n");
                }
                asm.push_str(".p2align 5\n");
            }
            std::fs::write(&source, &asm).unwrap();
            let result = Command::new("as")
                .arg("--64")
                .arg(&source)
                .arg("-o")
                .arg(&reference)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(&asm, &output)
                .unwrap();
            assert_eq!(
                flatten_nops(text(&std::fs::read(&output).unwrap())),
                flatten_nops(text(&std::fs::read(&reference).unwrap()))
            );
        }

        #[test]
        fn memory_move_rejects_an_unrepresentable_immediate() {
            let tmp = TempDir::new("invalid_immediate");
            let path = tmp.path().join("invalid_immediate.o");
            assert!(
                Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                    .unwrap()
                    .assemble("movq $4294967295, (%rax)\n", &path)
                    .is_err()
            );
        }

        fn oracle(asm: &str, name: &str) -> (Vec<u8>, Vec<u8>) {
            let tmp = TempDir::new(&format!("x86_{name}"));
            let dir = tmp.path();
            let source = dir.join("input.s");
            let reference = dir.join("reference.o");
            let output = dir.join("ras.o");
            std::fs::write(&source, asm).unwrap();
            let result = Command::new("as")
                .args(["--64", "-o"])
                .arg(&reference)
                .arg(&source)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(asm, &output)
                .unwrap();
            (
                std::fs::read(output).unwrap(),
                std::fs::read(reference).unwrap(),
            )
        }

        fn section<'a>(bytes: &'a [u8], name: &str) -> &'a [u8] {
            let at =
                |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
            let headers = at(40);
            let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
            let strings = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;
            let names = &bytes[at(headers + strings * 64 + 24)..];
            for i in 1..count {
                let h = headers + i * 64;
                let n = u32::from_le_bytes(bytes[h..h + 4].try_into().unwrap()) as usize;
                if names[n..].split(|b| *b == 0).next().unwrap() == name.as_bytes() {
                    return &bytes[at(h + 24)..at(h + 24) + at(h + 32)];
                }
            }
            panic!("Missing section {name}")
        }

        fn relocations(bytes: &[u8], name: &str) -> Vec<(u64, u32, String, i64)> {
            let at = |bytes: &[u8], offset| {
                u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
            };
            let symbols = section(bytes, ".symtab");
            let strings = section(bytes, ".strtab");
            section(bytes, name)
                .as_chunks::<24>()
                .0
                .iter()
                .map(|entry| {
                    let info = at(entry, 8);
                    let symbol = &symbols[(info >> 32) as usize * 24..];
                    let offset = u32::from_le_bytes(symbol[..4].try_into().unwrap()) as usize;
                    let name = strings[offset..].split(|b| *b == 0).next().unwrap();
                    (
                        at(entry, 0),
                        info as u32,
                        String::from_utf8(name.to_vec()).unwrap(),
                        at(entry, 16) as i64,
                    )
                })
                .collect()
        }

        #[test]
        fn gcc_instruction_forms_match_gas() {
            let (output, reference) =
                oracle(include_str!("../../../testcases/x86_gcc.s"), "gcc_forms");
            assert_eq!(text(&output), text(&reference));
            assert_eq!(
                relocations(&output, ".rela.text"),
                relocations(&reference, ".rela.text")
            );
        }

        #[test]
        fn truncating_float_to_integer_conversions_match_gas() {
            let asm = ".text\ncvttss2sil %xmm0, %eax\ncvttss2siq %xmm9, %r10\ncvttss2siq 8(%rsp), %rax\ncvttss2sil -4(%r13), %r9d\ncvttsd2siq %xmm1, %rcx\ncvttsd2sil 16(%r12), %edx\nret\n";
            let (output, reference) = oracle(asm, "cvtt_to_int");
            assert_eq!(text(&output), text(&reference));
        }

        #[test]
        fn scalar_float_comparisons_match_gas() {
            let asm = ".text\nucomiss %xmm0, %xmm1\nucomiss %xmm15, %xmm3\ncomiss %xmm2, %xmm7\nucomiss (%rax), %xmm0\ncomiss 8(%rbp,%rcx,4), %xmm9\nucomisd %xmm0, %xmm1\ncomisd %xmm4, %xmm5\nret\n";
            let (output, reference) = oracle(asm, "scalar_cmp");
            assert_eq!(text(&output), text(&reference));
        }

        #[test]
        fn float_width_conversions_match_gas() {
            let asm = ".text\ncvtsd2ss %xmm1, %xmm0\ncvtsd2ss %xmm9, %xmm10\ncvtsd2ss (%rsp), %xmm1\ncvtsd2ss 8(%r13), %xmm2\ncvtss2sd %xmm9, %xmm1\ncvtss2sd -4(%rbx), %xmm15\nret\n";
            let (output, reference) = oracle(asm, "float_width");
            assert_eq!(text(&output), text(&reference));
        }

        #[test]
        fn integer_to_single_conversions_match_gas() {
            let asm = ".text\ncvtsi2ssl %eax, %xmm0\ncvtsi2ssl %r9d, %xmm10\ncvtsi2ssl (%rsp), %xmm1\ncvtsi2ssl -12(%r12), %xmm7\ncvtsi2ssq %r9, %xmm10\ncvtsi2ssq 8(%rbp), %xmm2\nret\n";
            let (output, reference) = oracle(asm, "int_to_single");
            assert_eq!(text(&output), text(&reference));
        }

        #[test]
        fn ret_with_stack_adjustment_matches_gas() {
            let asm = ".text\nret $5\nret $0x10\nret $65535\nret\nret $0\n";
            let (output, reference) = oracle(asm, "ret_imm");
            assert_eq!(text(&output), text(&reference));
        }

        #[test]
        fn movabs_immediates_match_gas() {
            let asm = ".text\nmovabs $0x123456789abcdef0, %rcx\nmovabs $1, %rax\nmovabs $-2147483649, %r12\nmovabsq $9223372036854775807, %r9\nret\n";
            let (output, reference) = oracle(asm, "movabs");
            assert_eq!(text(&output), text(&reference));
        }

        #[test]
        fn absolute_symbols_in_data_directives_match_gas() {
            let asm = ".text\n.byte foo\n.long foo\n.quad later\n.byte later+1\n.short bar-2\n.set foo, 5\n.equ bar, 0x20\n.set later, 9\n.set alias, foo\n.byte alias\n.set alias2, alias+3\n.quad alias2\nret\n";
            let (output, reference) = oracle(asm, "abs_data");
            assert_eq!(text(&output), text(&reference));
        }

        #[test]
        fn direct_branches_to_symbols_match_gas() {
            let asm = ".text\ncall foo@PLT\njmp bar+8\nje baz-4\njne .Lend\ncall foo\n.Lend: ret\n";
            let (output, reference) = oracle(asm, "direct_branch");
            assert_eq!(text(&output), text(&reference));
        }

        #[test]
        fn rip_addends_and_prefixed_fixups_match_gas() {
            let asm = ".text\n.globl entry\nentry: movq 120+external(%rip), %rax\nmovq external+120(%rip), %r9\ncmpw $3, -8+external(%rip)\nmovq 8+.Llocal(%rip), %rax\n.Llocal: ret\nleaq 4+entry(%rip), %r9\nlock addw $1, external(%rip)\nnotrack jmp *external(%rip)\npshufd $3, 16+external(%rip), %xmm9\nlock addw $1, %fs:8(%rax)\n";
            let (output, reference) = oracle(asm, "rip_addends");
            assert_eq!(text(&output), text(&reference));
            assert_eq!(
                relocations(&output, ".rela.text"),
                relocations(&reference, ".rela.text")
            );
            let entries = relocations(&output, ".rela.text");
            assert_eq!(entries[0].3, 116);
            assert_eq!(entries[2].3, -13);
        }

        #[test]
        fn alignment_sensitive_branches_match_gas() {
            let mut asm = String::from(".text\n");
            for i in 100..160 {
                asm.push_str(&format!(".p2align 4\n.Lstart{i}: je .Lend{i}\n.space {i}, 0x90\njne .Lstart{i}\n.p2align 4,,10\n.p2align 3\njmp .Lend{i}\n.space 8, 0x90\n.Lend{i}: ret\n"));
            }
            let (output, reference) = oracle(&asm, "branch_alignment");
            assert_eq!(flatten_nops(text(&output)), flatten_nops(text(&reference)));
        }

        #[test]
        fn zero_max_skip_aligns_like_gas() {
            let asm = ".data\n.byte 1\n.p2align 4,,0\n.byte 2\n.p2align 4,,4\n.byte 3\n";
            let (output, reference) = oracle(asm, "zero_max_skip");
            assert_eq!(section(&output, ".data"), section(&reference, ".data"));
        }

        #[test]
        fn compiler_string_escapes_match_gas() {
            let asm = r#".text
        .ascii "\001\012\123\377\0x\b\f\v\a\n\t\r\\\""
        .asciz "\0007\01a\1234"
        "#;
            let (output, reference) = oracle(asm, "string_escapes");
            assert_eq!(text(&output), text(&reference));
        }

        #[test]
        fn invalid_compiler_operands_fail() {
            let tmp = TempDir::new("invalid_compiler");
            let path = tmp.path().join("invalid_compiler.o");
            for asm in [
                "cmpw %eax, %ax",
                "movd %ax, %xmm0",
                "mov %rax, $1",
                "mov %rax, %xmm0",
                "mov (%rax), %xmm0",
                "movswl %al, %eax",
                "movsbl %ecx, %eax",
                "movzwq %r9d, %rax",
                "movhps %xmm0, %xmm1",
                "pshufd $256, %xmm0, %xmm1",
                "cvtsi2sdq %eax, %xmm0",
                "ret $65536",
                "ret %rax",
                "endbr64 %rax",
                "leave %rax",
                "cltq %eax",
                "cqto %rax",
                "syscall $1",
                "ud2 %eax",
                "call *%eax",
                "jmp *%r9d",
                ".p2align 4,,,",
                "call %rax",
                "jmp 8(%rax)",
                "je %rax",
                "call foo bar",
                "call $5",
                ".set x y, 5",
                ".type foo bar, @function",
                ".size foo bar, 4",
                ".comm foo bar, 8",
                ".local foo bar",
                ".weak foo bar",
                "notrack jmp %rax",
                "lock cmpq $1, (%rax)",
                "rep addq %rax, %rbx",
                "movq external+other(%rip), %rax",
                "movq 2147483648+external(%rip), %rax",
            ] {
                assert!(
                    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                        .unwrap()
                        .assemble(asm, &path)
                        .is_err(),
                    "{asm}"
                );
            }
        }

        #[test]
        fn cross_section_branch_relocation_kinds_match_gas() {
            let asm = ".text\ncall .Lcold\njmp .Lcold\ncall cold_global\n.section .text.unlikely,\"ax\",@progbits\n.Lcold: ret\n.globl cold_global\ncold_global: ret\n";
            let (output, reference) = oracle(asm, "cold_branches");
            assert_eq!(text(&output), text(&reference));
            let fields = |bytes: &[u8]| {
                let mut entries = relocations(bytes, ".rela.text")
                    .into_iter()
                    .map(|(offset, kind, _, addend)| (offset, kind, addend))
                    .collect::<Vec<_>>();
                entries.sort();
                entries
            };
            assert_eq!(fields(&output), fields(&reference));
            assert_eq!(
                fields(&output)
                    .iter()
                    .map(|entry| entry.1)
                    .collect::<Vec<_>>(),
                [2, 2, 4]
            );
        }

        /// Every packed-SSE form the x86_64 backend can emit, against GNU as.
        #[test]
        fn packed_sse_forms_match_gas() {
            const BINARY: &[&str] = &[
                "paddb",
                "paddw",
                "paddd",
                "paddq",
                "psubb",
                "psubw",
                "psubd",
                "psubq",
                "pmullw",
                "pmuludq",
                "pmulld",
                "pand",
                "pandn",
                "por",
                "pxor",
                "psllw",
                "pslld",
                "psllq",
                "psrlw",
                "psrld",
                "psrlq",
                "psraw",
                "psrad",
                "punpcklbw",
                "punpcklwd",
                "punpckldq",
                "punpcklqdq",
                "punpckhbw",
                "punpckhwd",
                "punpckhdq",
                "punpckhqdq",
                "pcmpeqb",
                "pcmpeqw",
                "pcmpeqd",
                "pcmpgtb",
                "pcmpgtw",
                "pcmpgtd",
                "addps",
                "subps",
                "mulps",
                "divps",
                "minps",
                "maxps",
                "sqrtps",
                "andps",
                "andnps",
                "orps",
                "xorps",
                "unpcklps",
                "unpckhps",
                "addpd",
                "subpd",
                "mulpd",
                "divpd",
                "minpd",
                "maxpd",
                "sqrtpd",
                "andpd",
                "andnpd",
                "orpd",
                "xorpd",
                "unpcklpd",
                "unpckhpd",
            ];
            const SOURCES: &[&str] = &[
                "%xmm0",
                "%xmm7",
                "%xmm8",
                "%xmm15",
                "(%rax)",
                "(%r12)",
                "(%r13)",
                "(%rsp)",
                "(%rbp)",
                "1024(%rbx)",
                "-128(%r14)",
                "(%rax,%r9,8)",
                "(%r12,%r13,2)",
            ];
            const DESTINATIONS: &[&str] = &["%xmm0", "%xmm6", "%xmm8", "%xmm14"];
            let mut asm = String::from(".text\n");
            for mnemonic in BINARY {
                for source in SOURCES {
                    for destination in DESTINATIONS {
                        asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
                    }
                }
            }
            for mnemonic in [
                "psllw", "pslld", "psllq", "psrlw", "psrld", "psrlq", "psraw", "psrad", "pslldq",
                "psrldq",
            ] {
                for count in [0, 1, 31, 255] {
                    for destination in DESTINATIONS {
                        asm.push_str(&format!("{mnemonic} ${count}, {destination}\n"));
                    }
                }
            }
            for mnemonic in ["pshufd", "pshuflw", "pshufhw", "shufps", "shufpd"] {
                for source in SOURCES {
                    for destination in DESTINATIONS {
                        asm.push_str(&format!("{mnemonic} $27, {source}, {destination}\n"));
                    }
                }
            }
            for mnemonic in ["movdqu", "movdqa", "movaps", "movapd", "movups"] {
                for source in SOURCES {
                    for destination in DESTINATIONS {
                        asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
                        asm.push_str(&format!("{mnemonic} {destination}, {source}\n"));
                    }
                }
            }
            let (output, reference) = oracle(&asm, "packed_sse");
            assert_eq!(text(&output), text(&reference));
        }

        /// SSE4.1 lane transfers across both escape maps, register widths and REX.
        #[test]
        fn lane_transfer_forms_match_gas() {
            const WORD: &[&str] = &[
                "%eax", "%ecx", "%edi", "%esp", "%ebp", "%r8d", "%r13d", "%r15d",
            ];
            const QUAD: &[&str] = &[
                "%rax", "%rcx", "%rdi", "%rsp", "%rbp", "%r8", "%r13", "%r15",
            ];
            const MEMORY: &[&str] = &["(%rax)", "(%r12,%r13,2)", "-9(%rbp)", "(%rsp)"];
            const VECTORS: &[&str] = &["%xmm0", "%xmm7", "%xmm8", "%xmm15"];
            let mut asm = String::from(".text\n");
            for (suffix, registers, last) in [
                ("b", WORD, 15),
                ("w", WORD, 7),
                ("d", WORD, 3),
                ("q", QUAD, 1),
            ] {
                for vector in VECTORS {
                    for register in registers {
                        asm.push_str(&format!("pextr{suffix} $0, {vector}, {register}\n"));
                        asm.push_str(&format!("pinsr{suffix} ${last}, {register}, {vector}\n"));
                    }
                    if suffix == "w" {
                        continue;
                    }
                    for memory in MEMORY {
                        asm.push_str(&format!("pextr{suffix} ${last}, {vector}, {memory}\n"));
                        asm.push_str(&format!("pinsr{suffix} $0, {memory}, {vector}\n"));
                    }
                }
            }
            let (output, reference) = oracle(&asm, "lane_transfer");
            assert_eq!(text(&output), text(&reference));
        }

        /// The encoded bytes of each packed form, pinned so a regression cannot hide
        /// behind a reference assembler that moved with it.
        #[test]
        fn packed_sse_encodings_are_pinned() {
            const CASES: &[(&str, &str)] = &[
                ("paddb %xmm1, %xmm2", "660ffcd1"),
                ("paddb %xmm9, %xmm12", "66450ffce1"),
                ("paddb 8(%rbp), %xmm3", "660ffc5d08"),
                ("paddw %xmm1, %xmm2", "660ffdd1"),
                ("paddw %xmm9, %xmm12", "66450ffde1"),
                ("paddw 8(%rbp), %xmm3", "660ffd5d08"),
                ("paddd %xmm1, %xmm2", "660ffed1"),
                ("paddd %xmm9, %xmm12", "66450ffee1"),
                ("paddd 8(%rbp), %xmm3", "660ffe5d08"),
                ("paddq %xmm1, %xmm2", "660fd4d1"),
                ("paddq %xmm9, %xmm12", "66450fd4e1"),
                ("paddq 8(%rbp), %xmm3", "660fd45d08"),
                ("psubb %xmm1, %xmm2", "660ff8d1"),
                ("psubb %xmm9, %xmm12", "66450ff8e1"),
                ("psubb 8(%rbp), %xmm3", "660ff85d08"),
                ("psubw %xmm1, %xmm2", "660ff9d1"),
                ("psubw %xmm9, %xmm12", "66450ff9e1"),
                ("psubw 8(%rbp), %xmm3", "660ff95d08"),
                ("psubd %xmm1, %xmm2", "660ffad1"),
                ("psubd %xmm9, %xmm12", "66450ffae1"),
                ("psubd 8(%rbp), %xmm3", "660ffa5d08"),
                ("psubq %xmm1, %xmm2", "660ffbd1"),
                ("psubq %xmm9, %xmm12", "66450ffbe1"),
                ("psubq 8(%rbp), %xmm3", "660ffb5d08"),
                ("pmullw %xmm1, %xmm2", "660fd5d1"),
                ("pmullw %xmm9, %xmm12", "66450fd5e1"),
                ("pmullw 8(%rbp), %xmm3", "660fd55d08"),
                ("pmuludq %xmm1, %xmm2", "660ff4d1"),
                ("pmuludq %xmm9, %xmm12", "66450ff4e1"),
                ("pmuludq 8(%rbp), %xmm3", "660ff45d08"),
                ("pmulld %xmm1, %xmm2", "660f3840d1"),
                ("pmulld %xmm9, %xmm12", "66450f3840e1"),
                ("pmulld 8(%rbp), %xmm3", "660f38405d08"),
                ("pand %xmm1, %xmm2", "660fdbd1"),
                ("pand %xmm9, %xmm12", "66450fdbe1"),
                ("pand 8(%rbp), %xmm3", "660fdb5d08"),
                ("pandn %xmm1, %xmm2", "660fdfd1"),
                ("pandn %xmm9, %xmm12", "66450fdfe1"),
                ("pandn 8(%rbp), %xmm3", "660fdf5d08"),
                ("por %xmm1, %xmm2", "660febd1"),
                ("por %xmm9, %xmm12", "66450febe1"),
                ("por 8(%rbp), %xmm3", "660feb5d08"),
                ("pxor %xmm1, %xmm2", "660fefd1"),
                ("pxor %xmm9, %xmm12", "66450fefe1"),
                ("pxor 8(%rbp), %xmm3", "660fef5d08"),
                ("punpcklbw %xmm1, %xmm2", "660f60d1"),
                ("punpcklbw %xmm9, %xmm12", "66450f60e1"),
                ("punpcklbw 8(%rbp), %xmm3", "660f605d08"),
                ("punpcklwd %xmm1, %xmm2", "660f61d1"),
                ("punpcklwd %xmm9, %xmm12", "66450f61e1"),
                ("punpcklwd 8(%rbp), %xmm3", "660f615d08"),
                ("punpckldq %xmm1, %xmm2", "660f62d1"),
                ("punpckldq %xmm9, %xmm12", "66450f62e1"),
                ("punpckldq 8(%rbp), %xmm3", "660f625d08"),
                ("punpcklqdq %xmm1, %xmm2", "660f6cd1"),
                ("punpcklqdq %xmm9, %xmm12", "66450f6ce1"),
                ("punpcklqdq 8(%rbp), %xmm3", "660f6c5d08"),
                ("punpckhbw %xmm1, %xmm2", "660f68d1"),
                ("punpckhbw %xmm9, %xmm12", "66450f68e1"),
                ("punpckhbw 8(%rbp), %xmm3", "660f685d08"),
                ("punpckhwd %xmm1, %xmm2", "660f69d1"),
                ("punpckhwd %xmm9, %xmm12", "66450f69e1"),
                ("punpckhwd 8(%rbp), %xmm3", "660f695d08"),
                ("punpckhdq %xmm1, %xmm2", "660f6ad1"),
                ("punpckhdq %xmm9, %xmm12", "66450f6ae1"),
                ("punpckhdq 8(%rbp), %xmm3", "660f6a5d08"),
                ("punpckhqdq %xmm1, %xmm2", "660f6dd1"),
                ("punpckhqdq %xmm9, %xmm12", "66450f6de1"),
                ("punpckhqdq 8(%rbp), %xmm3", "660f6d5d08"),
                ("pcmpeqb %xmm1, %xmm2", "660f74d1"),
                ("pcmpeqb %xmm9, %xmm12", "66450f74e1"),
                ("pcmpeqb 8(%rbp), %xmm3", "660f745d08"),
                ("pcmpeqw %xmm1, %xmm2", "660f75d1"),
                ("pcmpeqw %xmm9, %xmm12", "66450f75e1"),
                ("pcmpeqw 8(%rbp), %xmm3", "660f755d08"),
                ("pcmpeqd %xmm1, %xmm2", "660f76d1"),
                ("pcmpeqd %xmm9, %xmm12", "66450f76e1"),
                ("pcmpeqd 8(%rbp), %xmm3", "660f765d08"),
                ("pcmpgtb %xmm1, %xmm2", "660f64d1"),
                ("pcmpgtb %xmm9, %xmm12", "66450f64e1"),
                ("pcmpgtb 8(%rbp), %xmm3", "660f645d08"),
                ("pcmpgtw %xmm1, %xmm2", "660f65d1"),
                ("pcmpgtw %xmm9, %xmm12", "66450f65e1"),
                ("pcmpgtw 8(%rbp), %xmm3", "660f655d08"),
                ("pcmpgtd %xmm1, %xmm2", "660f66d1"),
                ("pcmpgtd %xmm9, %xmm12", "66450f66e1"),
                ("pcmpgtd 8(%rbp), %xmm3", "660f665d08"),
                ("addps %xmm1, %xmm2", "0f58d1"),
                ("addps %xmm9, %xmm12", "450f58e1"),
                ("addps 8(%rbp), %xmm3", "0f585d08"),
                ("subps %xmm1, %xmm2", "0f5cd1"),
                ("subps %xmm9, %xmm12", "450f5ce1"),
                ("subps 8(%rbp), %xmm3", "0f5c5d08"),
                ("mulps %xmm1, %xmm2", "0f59d1"),
                ("mulps %xmm9, %xmm12", "450f59e1"),
                ("mulps 8(%rbp), %xmm3", "0f595d08"),
                ("divps %xmm1, %xmm2", "0f5ed1"),
                ("divps %xmm9, %xmm12", "450f5ee1"),
                ("divps 8(%rbp), %xmm3", "0f5e5d08"),
                ("minps %xmm1, %xmm2", "0f5dd1"),
                ("minps %xmm9, %xmm12", "450f5de1"),
                ("minps 8(%rbp), %xmm3", "0f5d5d08"),
                ("maxps %xmm1, %xmm2", "0f5fd1"),
                ("maxps %xmm9, %xmm12", "450f5fe1"),
                ("maxps 8(%rbp), %xmm3", "0f5f5d08"),
                ("sqrtps %xmm1, %xmm2", "0f51d1"),
                ("sqrtps %xmm9, %xmm12", "450f51e1"),
                ("sqrtps 8(%rbp), %xmm3", "0f515d08"),
                ("andps %xmm1, %xmm2", "0f54d1"),
                ("andps %xmm9, %xmm12", "450f54e1"),
                ("andps 8(%rbp), %xmm3", "0f545d08"),
                ("andnps %xmm1, %xmm2", "0f55d1"),
                ("andnps %xmm9, %xmm12", "450f55e1"),
                ("andnps 8(%rbp), %xmm3", "0f555d08"),
                ("orps %xmm1, %xmm2", "0f56d1"),
                ("orps %xmm9, %xmm12", "450f56e1"),
                ("orps 8(%rbp), %xmm3", "0f565d08"),
                ("xorps %xmm1, %xmm2", "0f57d1"),
                ("xorps %xmm9, %xmm12", "450f57e1"),
                ("xorps 8(%rbp), %xmm3", "0f575d08"),
                ("unpcklps %xmm1, %xmm2", "0f14d1"),
                ("unpcklps %xmm9, %xmm12", "450f14e1"),
                ("unpcklps 8(%rbp), %xmm3", "0f145d08"),
                ("unpckhps %xmm1, %xmm2", "0f15d1"),
                ("unpckhps %xmm9, %xmm12", "450f15e1"),
                ("unpckhps 8(%rbp), %xmm3", "0f155d08"),
                ("addpd %xmm1, %xmm2", "660f58d1"),
                ("addpd %xmm9, %xmm12", "66450f58e1"),
                ("addpd 8(%rbp), %xmm3", "660f585d08"),
                ("subpd %xmm1, %xmm2", "660f5cd1"),
                ("subpd %xmm9, %xmm12", "66450f5ce1"),
                ("subpd 8(%rbp), %xmm3", "660f5c5d08"),
                ("mulpd %xmm1, %xmm2", "660f59d1"),
                ("mulpd %xmm9, %xmm12", "66450f59e1"),
                ("mulpd 8(%rbp), %xmm3", "660f595d08"),
                ("divpd %xmm1, %xmm2", "660f5ed1"),
                ("divpd %xmm9, %xmm12", "66450f5ee1"),
                ("divpd 8(%rbp), %xmm3", "660f5e5d08"),
                ("minpd %xmm1, %xmm2", "660f5dd1"),
                ("minpd %xmm9, %xmm12", "66450f5de1"),
                ("minpd 8(%rbp), %xmm3", "660f5d5d08"),
                ("maxpd %xmm1, %xmm2", "660f5fd1"),
                ("maxpd %xmm9, %xmm12", "66450f5fe1"),
                ("maxpd 8(%rbp), %xmm3", "660f5f5d08"),
                ("sqrtpd %xmm1, %xmm2", "660f51d1"),
                ("sqrtpd %xmm9, %xmm12", "66450f51e1"),
                ("sqrtpd 8(%rbp), %xmm3", "660f515d08"),
                ("andpd %xmm1, %xmm2", "660f54d1"),
                ("andpd %xmm9, %xmm12", "66450f54e1"),
                ("andpd 8(%rbp), %xmm3", "660f545d08"),
                ("andnpd %xmm1, %xmm2", "660f55d1"),
                ("andnpd %xmm9, %xmm12", "66450f55e1"),
                ("andnpd 8(%rbp), %xmm3", "660f555d08"),
                ("orpd %xmm1, %xmm2", "660f56d1"),
                ("orpd %xmm9, %xmm12", "66450f56e1"),
                ("orpd 8(%rbp), %xmm3", "660f565d08"),
                ("xorpd %xmm1, %xmm2", "660f57d1"),
                ("xorpd %xmm9, %xmm12", "66450f57e1"),
                ("xorpd 8(%rbp), %xmm3", "660f575d08"),
                ("unpcklpd %xmm1, %xmm2", "660f14d1"),
                ("unpcklpd %xmm9, %xmm12", "66450f14e1"),
                ("unpcklpd 8(%rbp), %xmm3", "660f145d08"),
                ("unpckhpd %xmm1, %xmm2", "660f15d1"),
                ("unpckhpd %xmm9, %xmm12", "66450f15e1"),
                ("unpckhpd 8(%rbp), %xmm3", "660f155d08"),
                ("psllw %xmm1, %xmm2", "660ff1d1"),
                ("psllw %xmm9, %xmm12", "66450ff1e1"),
                ("psllw $5, %xmm2", "660f71f205"),
                ("psllw $5, %xmm12", "66410f71f405"),
                ("pslld %xmm1, %xmm2", "660ff2d1"),
                ("pslld %xmm9, %xmm12", "66450ff2e1"),
                ("pslld $5, %xmm2", "660f72f205"),
                ("pslld $5, %xmm12", "66410f72f405"),
                ("psllq %xmm1, %xmm2", "660ff3d1"),
                ("psllq %xmm9, %xmm12", "66450ff3e1"),
                ("psllq $5, %xmm2", "660f73f205"),
                ("psllq $5, %xmm12", "66410f73f405"),
                ("psrlw %xmm1, %xmm2", "660fd1d1"),
                ("psrlw %xmm9, %xmm12", "66450fd1e1"),
                ("psrlw $5, %xmm2", "660f71d205"),
                ("psrlw $5, %xmm12", "66410f71d405"),
                ("psrld %xmm1, %xmm2", "660fd2d1"),
                ("psrld %xmm9, %xmm12", "66450fd2e1"),
                ("psrld $5, %xmm2", "660f72d205"),
                ("psrld $5, %xmm12", "66410f72d405"),
                ("psrlq %xmm1, %xmm2", "660fd3d1"),
                ("psrlq %xmm9, %xmm12", "66450fd3e1"),
                ("psrlq $5, %xmm2", "660f73d205"),
                ("psrlq $5, %xmm12", "66410f73d405"),
                ("psraw %xmm1, %xmm2", "660fe1d1"),
                ("psraw %xmm9, %xmm12", "66450fe1e1"),
                ("psraw $5, %xmm2", "660f71e205"),
                ("psraw $5, %xmm12", "66410f71e405"),
                ("psrad %xmm1, %xmm2", "660fe2d1"),
                ("psrad %xmm9, %xmm12", "66450fe2e1"),
                ("psrad $5, %xmm2", "660f72e205"),
                ("psrad $5, %xmm12", "66410f72e405"),
                ("pslldq $5, %xmm2", "660f73fa05"),
                ("pslldq $5, %xmm12", "66410f73fc05"),
                ("psrldq $5, %xmm2", "660f73da05"),
                ("psrldq $5, %xmm12", "66410f73dc05"),
                ("shufps $27, %xmm1, %xmm2", "0fc6d11b"),
                ("shufps $27, %xmm9, %xmm12", "450fc6e11b"),
                ("pshufhw $27, %xmm1, %xmm2", "f30f70d11b"),
                ("pextrb $1, %xmm2, %eax", "660f3a14d001"),
                ("pextrb $1, %xmm12, %r9d", "66450f3a14e101"),
                ("pextrb $1, %xmm12, 8(%rbp)", "66440f3a14650801"),
                ("pinsrb $1, %eax, %xmm2", "660f3a20d001"),
                ("pinsrb $1, %r9d, %xmm12", "66450f3a20e101"),
                ("pinsrb $1, 8(%rbp), %xmm12", "66440f3a20650801"),
                ("pextrd $1, %xmm2, %eax", "660f3a16d001"),
                ("pextrd $1, %xmm12, %r9d", "66450f3a16e101"),
                ("pextrd $1, %xmm12, 8(%rbp)", "66440f3a16650801"),
                ("pinsrd $1, %eax, %xmm2", "660f3a22d001"),
                ("pinsrd $1, %r9d, %xmm12", "66450f3a22e101"),
                ("pinsrd $1, 8(%rbp), %xmm12", "66440f3a22650801"),
                ("pextrq $1, %xmm2, %rax", "66480f3a16d001"),
                ("pextrq $1, %xmm12, %r9", "664d0f3a16e101"),
                ("pextrq $1, %xmm12, 8(%rbp)", "664c0f3a16650801"),
                ("pinsrq $1, %rax, %xmm2", "66480f3a22d001"),
                ("pinsrq $1, %r9, %xmm12", "664d0f3a22e101"),
                ("pinsrq $1, 8(%rbp), %xmm12", "664c0f3a22650801"),
            ];
            for (asm, expected) in CASES {
                let tmp = TempDir::new("pinned_sse");
                let path = tmp.path().join("pinned.o");
                Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                    .unwrap()
                    .assemble(&format!(".text\n{asm}\n"), &path)
                    .unwrap();
                let bytes = std::fs::read(&path).unwrap();
                let encoded: String = text(&bytes).iter().map(|b| format!("{b:02x}")).collect();
                assert_eq!(&encoded, expected, "{asm}");
            }
        }

        /// A lane transfer whose register width disagrees with the mnemonic is not
        /// silently encoded as the other width.
        #[test]
        fn lane_transfer_rejects_a_mismatched_register_width() {
            for asm in [
                "pextrd $0, %xmm0, %rax",
                "pextrq $0, %xmm0, %eax",
                "pinsrd $0, %rax, %xmm0",
                "pinsrq $0, %eax, %xmm0",
            ] {
                let tmp = TempDir::new("lane_width");
                let path = tmp.path().join("lane.o");
                assert!(
                    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                        .unwrap()
                        .assemble(&format!(".text\n{asm}\n"), &path)
                        .is_err(),
                    "{asm}"
                );
            }
        }

        /// The SSE2, SSSE3, SSE4.1 and SSE4.2 forms GNU as accepts, across both
        /// escape maps, every register slot and the awkward addressing bases.
        #[test]
        fn extended_sse_forms_match_gas() {
            const BINARY: &[&str] = &[
                "pmaddwd",
                "pshufb",
                "packuswb",
                "packsswb",
                "packssdw",
                "packusdw",
                "pavgb",
                "pavgw",
                "psignb",
                "psignw",
                "psignd",
                "pabsb",
                "pabsw",
                "pabsd",
                "pminsb",
                "pminsw",
                "pminsd",
                "pmaxsb",
                "pmaxsw",
                "pmaxsd",
                "pminub",
                "pminuw",
                "pminud",
                "pmaxub",
                "pmaxuw",
                "pmaxud",
                "pcmpeqq",
                "pcmpgtq",
                "pmuldq",
                "pmulhw",
                "pmulhuw",
                "pmulhrsw",
                "pmaddubsw",
                "psadbw",
                "phaddw",
                "phaddd",
                "phaddsw",
                "phsubw",
                "phsubd",
                "phsubsw",
                "pmovsxbw",
                "pmovsxbd",
                "pmovsxbq",
                "pmovsxwd",
                "pmovsxwq",
                "pmovsxdq",
                "pmovzxbw",
                "pmovzxbd",
                "pmovzxbq",
                "pmovzxwd",
                "pmovzxwq",
                "pmovzxdq",
                "ptest",
                "phminposuw",
                "paddsb",
                "paddsw",
                "paddusb",
                "paddusw",
                "psubsb",
                "psubsw",
                "psubusb",
                "psubusw",
                "addsubps",
                "addsubpd",
                "haddps",
                "haddpd",
                "hsubps",
                "hsubpd",
                "movshdup",
                "movsldup",
                "movddup",
                "rcpps",
                "rsqrtps",
                "cvtdq2ps",
                "cvtps2dq",
                "cvttps2dq",
                "cvtdq2pd",
                "cvtpd2ps",
                "cvtps2pd",
                "cvttpd2dq",
                "cvtpd2dq",
                "aesenc",
                "aesenclast",
                "aesdec",
                "aesdeclast",
                "aesimc",
            ];
            const SOURCES: &[&str] = &[
                "%xmm0",
                "%xmm7",
                "%xmm8",
                "%xmm15",
                "(%rax)",
                "(%r12)",
                "(%r13)",
                "(%rsp)",
                "(%rbp)",
                "(%r12,%r13,2)",
                "(%rax,%r9,8)",
                "1024(%rbx)",
                "-128(%r14)",
                "pinned(%rip)",
                "8(%r13)",
            ];
            const DESTINATIONS: &[&str] = &["%xmm0", "%xmm7", "%xmm8", "%xmm15"];
            let mut asm = String::from(".text\npinned: .quad 0\n");
            for mnemonic in BINARY {
                for source in SOURCES {
                    for destination in DESTINATIONS {
                        asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
                    }
                }
            }
            for mnemonic in ["blendvps", "blendvpd", "pblendvb"] {
                for source in SOURCES {
                    for destination in DESTINATIONS {
                        asm.push_str(&format!("{mnemonic} %xmm0, {source}, {destination}\n"));
                    }
                }
            }
            for (mnemonic, counts) in [
                ("pblendw", &[0, 1, 127, 128, 255][..]),
                ("palignr", &[0, 1, 15, 16, 255]),
                ("pclmulqdq", &[0, 1, 16, 17, 255]),
                ("blendps", &[0, 15, 255]),
                ("blendpd", &[0, 3, 255]),
                ("insertps", &[0, 78, 255]),
                ("dpps", &[0, 255]),
                ("dppd", &[0, 51, 255]),
                ("mpsadbw", &[0, 7, 255]),
                ("roundps", &[0, 1, 11, 255]),
                ("roundpd", &[0, 2, 255]),
                ("roundss", &[0, 3, 255]),
                ("roundsd", &[0, 4, 255]),
                ("cmpps", &[0, 7, 31, 255]),
                ("cmppd", &[0, 7, 255]),
                ("aeskeygenassist", &[0, 255]),
                ("pcmpistri", &[0, 255]),
                ("pcmpistrm", &[0, 255]),
                ("pcmpestri", &[0, 255]),
                ("pcmpestrm", &[0, 255]),
            ] {
                for count in counts {
                    for source in SOURCES {
                        for destination in DESTINATIONS {
                            asm.push_str(&format!(
                                "{mnemonic} ${count}, {source}, {destination}\n"
                            ));
                        }
                    }
                }
            }
            for source in SOURCES.iter().filter(|source| source.contains('(')) {
                for destination in DESTINATIONS {
                    for mnemonic in ["movntdq", "movntps", "movntpd"] {
                        asm.push_str(&format!("{mnemonic} {destination}, {source}\n"));
                    }
                    for mnemonic in ["lddqu", "movntdqa"] {
                        asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
                    }
                    for mnemonic in ["movlps", "movhps", "movlpd", "movhpd", "movupd"] {
                        asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
                        asm.push_str(&format!("{mnemonic} {destination}, {source}\n"));
                    }
                }
                asm.push_str(&format!("prefetchnta {source}\nprefetcht0 {source}\n"));
                asm.push_str(&format!("prefetcht1 {source}\nprefetcht2 {source}\n"));
                asm.push_str(&format!("ldmxcsr {source}\nstmxcsr {source}\n"));
            }
            for vector in DESTINATIONS {
                for register in ["%eax", "%esp", "%ebp", "%r9d", "%r12d", "%r15d"] {
                    for mnemonic in ["pmovmskb", "movmskps", "movmskpd"] {
                        asm.push_str(&format!("{mnemonic} {vector}, {register}\n"));
                    }
                    asm.push_str(&format!("cvtss2si {vector}, {register}\n"));
                    asm.push_str(&format!("cvtsd2si {vector}, {register}\n"));
                    asm.push_str(&format!("cvttss2si {vector}, {register}\n"));
                }
                for other in DESTINATIONS {
                    asm.push_str(&format!("movlhps {vector}, {other}\n"));
                    asm.push_str(&format!("movhlps {vector}, {other}\n"));
                    asm.push_str(&format!("rcpss {vector}, {other}\n"));
                    asm.push_str(&format!("rsqrtss {vector}, {other}\n"));
                }
            }
            asm.push_str("sfence\nlfence\nmfence\npause\n");
            let (output, reference) = oracle(&asm, "extended_sse");
            assert_eq!(text(&output), text(&reference));
        }

        /// `pextrw` picks its opcode map from the kind of its destination, and
        /// `extractps` shares the lane-transfer path.
        #[test]
        fn word_and_lane_extraction_forms_match_gas() {
            const MEMORY: &[&str] = &[
                "(%rax)",
                "(%r12)",
                "(%r13)",
                "(%rsp)",
                "(%rbp)",
                "(%r12,%r13,2)",
                "pinned(%rip)",
                "-128(%r14)",
                "1024(%rbx)",
            ];
            const VECTORS: &[&str] = &["%xmm0", "%xmm7", "%xmm8", "%xmm15"];
            let mut asm = String::from(".text\npinned: .quad 0\n");
            for lane in [0, 1, 7, 255] {
                for vector in VECTORS {
                    for register in ["%eax", "%esp", "%ebp", "%r9d", "%r12d", "%r15d"] {
                        asm.push_str(&format!("pextrw ${lane}, {vector}, {register}\n"));
                        asm.push_str(&format!("pinsrw ${lane}, {register}, {vector}\n"));
                        asm.push_str(&format!("extractps ${lane}, {vector}, {register}\n"));
                    }
                    for memory in MEMORY {
                        asm.push_str(&format!("pextrw ${lane}, {vector}, {memory}\n"));
                        asm.push_str(&format!("pinsrw ${lane}, {memory}, {vector}\n"));
                        asm.push_str(&format!("extractps ${lane}, {vector}, {memory}\n"));
                    }
                }
            }
            let (output, reference) = oracle(&asm, "word_extraction");
            assert_eq!(text(&output), text(&reference));
        }

        /// SSE4.2 CRC32 over every source width, including the byte registers that
        /// need a REX of their own.
        #[test]
        fn crc32_and_non_temporal_stores_match_gas() {
            const MEMORY: &[&str] = &["(%rax)", "(%r12)", "(%r13)", "(%rsp)", "(%rbp)", "-9(%r14)"];
            let mut asm = String::from(".text\n");
            for (suffix, sources, destinations) in [
                (
                    "b",
                    &[
                        "%al", "%cl", "%spl", "%bpl", "%sil", "%dil", "%r9b", "%r15b",
                    ][..],
                    &["%eax", "%ecx", "%esp", "%ebp", "%r9d", "%r15d"][..],
                ),
                (
                    "w",
                    &["%ax", "%cx", "%sp", "%bp", "%r9w", "%r15w"],
                    &["%eax", "%ecx", "%esp", "%ebp", "%r9d", "%r15d"],
                ),
                (
                    "l",
                    &["%eax", "%ecx", "%esp", "%ebp", "%r9d", "%r15d"],
                    &["%eax", "%ecx", "%esp", "%ebp", "%r9d", "%r15d"],
                ),
                (
                    "q",
                    &["%rax", "%rcx", "%rsp", "%rbp", "%r9", "%r15"],
                    &["%rax", "%rcx", "%rsp", "%rbp", "%r9", "%r15"],
                ),
            ] {
                for destination in destinations {
                    for source in sources {
                        asm.push_str(&format!("crc32{suffix} {source}, {destination}\n"));
                    }
                    for memory in MEMORY {
                        asm.push_str(&format!("crc32{suffix} {memory}, {destination}\n"));
                    }
                }
            }
            for memory in MEMORY {
                for register in ["%eax", "%ebp", "%r12d"] {
                    asm.push_str(&format!("movntil {register}, {memory}\n"));
                }
                for register in ["%rax", "%rbp", "%r12"] {
                    asm.push_str(&format!("movntiq {register}, {memory}\n"));
                }
            }
            let (output, reference) = oracle(&asm, "crc32");
            assert_eq!(text(&output), text(&reference));
        }

        /// The AVX2 per-lane shifts, which select the three-byte VEX form because
        /// they live in the `0f 38` map.
        #[test]
        fn variable_vector_shifts_match_gas() {
            const MEMORY: &[&str] = &[
                "(%rax)",
                "(%r12)",
                "(%r13)",
                "(%rsp)",
                "(%rbp)",
                "(%r12,%r13,2)",
                "pinned(%rip)",
                "1024(%rbx)",
            ];
            let mut asm = String::from(".text\npinned: .quad 0\n");
            for mnemonic in ["vpsllvd", "vpsrlvd", "vpsravd", "vpsllvq", "vpsrlvq"] {
                for width in ["xmm", "ymm"] {
                    let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
                    for count in &registers {
                        for source in &registers {
                            for destination in &registers {
                                asm.push_str(&format!(
                                    "{mnemonic} {count}, {source}, {destination}\n"
                                ));
                            }
                        }
                    }
                    for memory in MEMORY {
                        for source in &registers {
                            asm.push_str(&format!("{mnemonic} {memory}, {source}, %{width}3\n"));
                        }
                    }
                }
            }
            let (output, reference) = oracle(&asm, "variable_shifts");
            assert_eq!(text(&output), text(&reference));
        }

        /// The VEX counterparts of the packed set, over both vector widths and the
        /// register numbers that force the three-byte prefix.
        #[test]
        fn avx_packed_forms_match_gas() {
            const NDS: &[&str] = &[
                "vpaddb",
                "vpaddw",
                "vpaddd",
                "vpaddq",
                "vpsubb",
                "vpsubq",
                "vpmullw",
                "vpmuludq",
                "vpmaddwd",
                "vpsadbw",
                "vpavgb",
                "vpacksswb",
                "vpackuswb",
                "vpand",
                "vpandn",
                "vpor",
                "vpxor",
                "vpunpcklbw",
                "vpunpckhqdq",
                "vpcmpeqb",
                "vpcmpgtd",
                "vaddps",
                "vmulpd",
                "vdivps",
                "vminpd",
                "vmaxps",
                "vandnpd",
                "vunpcklps",
                "vaddsubps",
                "vhaddpd",
                "vpshufb",
                "vphaddw",
                "vpmaddubsw",
                "vpsignd",
                "vpmulhrsw",
                "vpmuldq",
                "vpcmpeqq",
                "vpackusdw",
                "vpcmpgtq",
                "vpminsb",
                "vpmaxud",
                "vpmulld",
                "vaddss",
                "vdivsd",
                "vsqrtss",
                "vcvtss2sd",
                "vmovhlps",
                "vmovlhps",
            ];
            const RM: &[&str] = &[
                "vsqrtps",
                "vsqrtpd",
                "vrcpps",
                "vmovshdup",
                "vmovddup",
                "vcvtdq2ps",
                "vcvttps2dq",
                "vptest",
                "vpabsb",
                "vpabsd",
                "vpmovsxbw",
                "vpmovzxdq",
                "vucomiss",
                "vcomisd",
                "vbroadcastss",
                "vpbroadcastb",
                "vpbroadcastq",
            ];
            const MEMORY: &[&str] = &[
                "(%rax)",
                "(%r12)",
                "(%r13)",
                "(%rsp)",
                "(%rbp)",
                "(%r12,%r13,2)",
                "pinned(%rip)",
                "-128(%r14)",
            ];
            let scalar = |name: &str| {
                name.ends_with("ss")
                    || name.ends_with("sd")
                    || matches!(name, "vmovhlps" | "vmovlhps")
            };
            let mut asm = String::from(".text\npinned: .quad 0\n");
            for mnemonic in NDS {
                for width in ["xmm", "ymm"] {
                    if width == "ymm" && scalar(mnemonic) {
                        continue;
                    }
                    let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
                    for second in &registers {
                        for first in &registers {
                            for destination in &registers {
                                asm.push_str(&format!(
                                    "{mnemonic} {second}, {first}, {destination}\n"
                                ));
                            }
                        }
                    }
                    if matches!(*mnemonic, "vmovhlps" | "vmovlhps") {
                        continue;
                    }
                    for memory in MEMORY {
                        asm.push_str(&format!("{mnemonic} {memory}, %{width}2, %{width}3\n"));
                    }
                }
            }
            for mnemonic in RM {
                for source in [0, 7, 8, 15].map(|n| format!("%xmm{n}")) {
                    for destination in [0, 7, 8, 15].map(|n| format!("%xmm{n}")) {
                        asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
                    }
                }
                for memory in MEMORY {
                    asm.push_str(&format!("{mnemonic} {memory}, %xmm3\n"));
                }
            }
            for (mnemonic, count) in [
                ("vpblendw", 255),
                ("vpalignr", 15),
                ("vblendps", 15),
                ("vdpps", 255),
                ("vmpsadbw", 7),
                ("vcmpps", 7),
                ("vshufps", 27),
                ("vshufpd", 3),
            ] {
                for width in ["xmm", "ymm"] {
                    let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
                    for second in &registers {
                        for destination in &registers {
                            asm.push_str(&format!(
                                "{mnemonic} ${count}, {second}, %{width}2, {destination}\n"
                            ));
                        }
                    }
                    for memory in MEMORY {
                        asm.push_str(&format!(
                            "{mnemonic} ${count}, {memory}, %{width}2, %{width}3\n"
                        ));
                    }
                }
            }
            for (mnemonic, count) in [
                ("vpshufd", 27),
                ("vpshufhw", 27),
                ("vpshuflw", 27),
                ("vroundps", 11),
                ("vpermq", 27),
                ("vpermilps", 27),
            ] {
                for width in ["xmm", "ymm"] {
                    if width == "xmm" && mnemonic == "vpermq" {
                        continue;
                    }
                    let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
                    for source in &registers {
                        for destination in &registers {
                            asm.push_str(&format!(
                                "{mnemonic} ${count}, {source}, {destination}\n"
                            ));
                        }
                    }
                    for memory in MEMORY {
                        asm.push_str(&format!("{mnemonic} ${count}, {memory}, %{width}3\n"));
                    }
                }
            }
            for mnemonic in ["vblendvps", "vblendvpd", "vpblendvb"] {
                for width in ["xmm", "ymm"] {
                    let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
                    for mask in &registers {
                        for second in &registers {
                            asm.push_str(&format!(
                                "{mnemonic} {mask}, {second}, %{width}2, %{width}3\n"
                            ));
                        }
                    }
                    for memory in MEMORY {
                        asm.push_str(&format!(
                            "{mnemonic} %{width}4, {memory}, %{width}2, %{width}3\n"
                        ));
                    }
                }
            }
            for mnemonic in [
                "vpsllw", "vpslld", "vpsllq", "vpsrlw", "vpsrld", "vpsrlq", "vpsraw", "vpsrad",
                "vpslldq", "vpsrldq",
            ] {
                for count in [0, 1, 31, 255] {
                    for width in ["xmm", "ymm"] {
                        for source in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                            asm.push_str(&format!("{mnemonic} ${count}, {source}, %{width}3\n"));
                        }
                    }
                }
            }
            for mnemonic in [
                "vmovaps", "vmovapd", "vmovups", "vmovupd", "vmovdqa", "vmovdqu",
            ] {
                for width in ["xmm", "ymm"] {
                    for register in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                        for memory in MEMORY {
                            asm.push_str(&format!("{mnemonic} {memory}, {register}\n"));
                            asm.push_str(&format!("{mnemonic} {register}, {memory}\n"));
                        }
                        for other in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                            asm.push_str(&format!("{mnemonic} {register}, {other}\n"));
                        }
                    }
                }
            }
            for mnemonic in ["vmovntdq", "vmovntps", "vmovntpd"] {
                for width in ["xmm", "ymm"] {
                    for register in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                        for memory in MEMORY {
                            asm.push_str(&format!("{mnemonic} {register}, {memory}\n"));
                        }
                    }
                }
            }
            asm.push_str("vzeroall\nvzeroupper\n");
            let (output, reference) = oracle(&asm, "avx_packed");
            assert_eq!(text(&output), text(&reference));
        }

        /// The VEX lane transfers, scalar moves and FMA3 set, where the two-byte
        /// prefix is chosen only when no field it cannot carry is in use.
        #[test]
        fn avx_transfers_and_fma_match_gas() {
            const MEMORY: &[&str] = &[
                "(%rax)",
                "(%r12)",
                "(%r13)",
                "(%rsp)",
                "(%rbp)",
                "(%r12,%r13,2)",
                "pinned(%rip)",
                "-128(%r14)",
            ];
            let vectors = || [0, 7, 8, 15].map(|n| format!("%xmm{n}"));
            let mut asm = String::from(".text\npinned: .quad 0\n");
            for lane in [0, 1, 15, 255] {
                for vector in vectors() {
                    for register in ["%eax", "%esp", "%ebp", "%r9d", "%r12d", "%r15d"] {
                        for mnemonic in ["vpextrb", "vpextrw", "vpextrd", "vextractps"] {
                            asm.push_str(&format!("{mnemonic} ${lane}, {vector}, {register}\n"));
                        }
                        for mnemonic in ["vpinsrb", "vpinsrw", "vpinsrd"] {
                            for first in vectors() {
                                asm.push_str(&format!(
                                    "{mnemonic} ${lane}, {register}, {first}, {vector}\n"
                                ));
                            }
                        }
                    }
                    for register in ["%rax", "%rsp", "%rbp", "%r9", "%r12", "%r15"] {
                        asm.push_str(&format!("vpextrq ${lane}, {vector}, {register}\n"));
                        for first in vectors() {
                            asm.push_str(&format!(
                                "vpinsrq ${lane}, {register}, {first}, {vector}\n"
                            ));
                        }
                    }
                    for memory in MEMORY {
                        for mnemonic in ["vpextrb", "vpextrw", "vpextrd", "vpextrq", "vextractps"] {
                            asm.push_str(&format!("{mnemonic} ${lane}, {vector}, {memory}\n"));
                        }
                        for mnemonic in ["vpinsrb", "vpinsrw", "vpinsrd", "vpinsrq"] {
                            asm.push_str(&format!(
                                "{mnemonic} ${lane}, {memory}, %xmm2, {vector}\n"
                            ));
                        }
                    }
                }
            }
            for width in ["xmm", "ymm"] {
                for vector in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                    for register in ["%eax", "%esp", "%ebp", "%r9d", "%r12d", "%r15d"] {
                        for mnemonic in ["vpmovmskb", "vmovmskps", "vmovmskpd"] {
                            asm.push_str(&format!("{mnemonic} {vector}, {register}\n"));
                        }
                    }
                }
            }
            for vector in vectors() {
                for register in ["%rax", "%rsp", "%rbp", "%r9", "%r12", "%r15"] {
                    asm.push_str(&format!("vmovq {register}, {vector}\n"));
                    asm.push_str(&format!("vmovq {vector}, {register}\n"));
                }
                for register in ["%eax", "%esp", "%r12d"] {
                    asm.push_str(&format!("vmovd {register}, {vector}\n"));
                    asm.push_str(&format!("vmovd {vector}, {register}\n"));
                }
                for other in vectors() {
                    asm.push_str(&format!("vmovq {other}, {vector}\n"));
                    for mnemonic in ["vmovss", "vmovsd"] {
                        for first in vectors() {
                            asm.push_str(&format!("{mnemonic} {other}, {first}, {vector}\n"));
                        }
                    }
                }
                for memory in MEMORY {
                    asm.push_str(&format!("vmovq {memory}, {vector}\n"));
                    asm.push_str(&format!("vmovq {vector}, {memory}\n"));
                    for mnemonic in ["vmovss", "vmovsd"] {
                        asm.push_str(&format!("{mnemonic} {memory}, {vector}\n"));
                        asm.push_str(&format!("{mnemonic} {vector}, {memory}\n"));
                    }
                    for mnemonic in ["vmovhps", "vmovlps", "vmovhpd", "vmovlpd"] {
                        asm.push_str(&format!("{mnemonic} {memory}, %xmm2, {vector}\n"));
                        asm.push_str(&format!("{mnemonic} {vector}, {memory}\n"));
                    }
                }
                for register in ["%eax", "%esp", "%r12d"] {
                    for first in vectors() {
                        asm.push_str(&format!("vcvtsi2ssl {register}, {first}, {vector}\n"));
                        asm.push_str(&format!("vcvtsi2sdl {register}, {first}, {vector}\n"));
                    }
                }
                for register in ["%rax", "%rsp", "%r12"] {
                    asm.push_str(&format!("vcvtsi2ssq {register}, %xmm2, {vector}\n"));
                    asm.push_str(&format!("vcvtsi2sdq {register}, %xmm2, {vector}\n"));
                }
                for register in ["%eax", "%r12d", "%rax", "%r12"] {
                    for mnemonic in ["vcvtss2si", "vcvtsd2si", "vcvttss2si", "vcvttsd2si"] {
                        asm.push_str(&format!("{mnemonic} {vector}, {register}\n"));
                    }
                }
            }
            for operation in ["madd", "msub", "nmadd", "nmsub"] {
                for order in ["132", "213", "231"] {
                    for kind in ["ps", "pd", "ss", "sd"] {
                        let mnemonic = format!("vf{operation}{order}{kind}");
                        let width = if kind.starts_with('p') { "ymm" } else { "xmm" };
                        let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
                        for second in &registers {
                            for destination in &registers {
                                asm.push_str(&format!(
                                    "{mnemonic} {second}, %{width}2, {destination}\n"
                                ));
                            }
                        }
                        for memory in MEMORY {
                            asm.push_str(&format!("{mnemonic} {memory}, %{width}2, %{width}3\n"));
                        }
                    }
                }
            }
            for operation in ["maddsub", "msubadd"] {
                for order in ["132", "213", "231"] {
                    for kind in ["ps", "pd"] {
                        let mnemonic = format!("vf{operation}{order}{kind}");
                        for second in [0, 7, 8, 15].map(|n| format!("%ymm{n}")) {
                            asm.push_str(&format!("{mnemonic} {second}, %ymm2, %ymm15\n"));
                        }
                    }
                }
            }
            for mnemonic in ["vmaskmovps", "vmaskmovpd", "vpmaskmovd", "vpmaskmovq"] {
                for width in ["xmm", "ymm"] {
                    for register in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                        for memory in MEMORY {
                            asm.push_str(&format!("{mnemonic} {memory}, {register}, %{width}3\n"));
                            asm.push_str(&format!("{mnemonic} {register}, %{width}2, {memory}\n"));
                        }
                    }
                }
            }
            for lane in [0, 1] {
                for width in [("xmm", "ymm")] {
                    for register in [0, 7, 8, 15].map(|n| format!("%{}{n}", width.0)) {
                        asm.push_str(&format!("vinserti128 ${lane}, {register}, %ymm2, %ymm3\n"));
                        asm.push_str(&format!("vinsertf128 ${lane}, {register}, %ymm2, %ymm3\n"));
                        asm.push_str(&format!("vextracti128 ${lane}, %ymm9, {register}\n"));
                        asm.push_str(&format!("vextractf128 ${lane}, %ymm9, {register}\n"));
                    }
                    for memory in MEMORY {
                        asm.push_str(&format!("vextracti128 ${lane}, %ymm9, {memory}\n"));
                        asm.push_str(&format!("vinserti128 ${lane}, {memory}, %ymm2, %ymm3\n"));
                        asm.push_str(&format!("vperm2i128 ${lane}, {memory}, %ymm2, %ymm3\n"));
                    }
                }
            }
            for second in [0, 7, 8, 15].map(|n| format!("%ymm{n}")) {
                asm.push_str(&format!("vpermd {second}, %ymm2, %ymm3\n"));
                asm.push_str(&format!("vpermps {second}, %ymm2, %ymm3\n"));
                asm.push_str(&format!("vpermilps {second}, %ymm2, %ymm3\n"));
                asm.push_str(&format!("vpermilpd {second}, %ymm2, %ymm3\n"));
                asm.push_str(&format!("vperm2f128 $1, {second}, %ymm2, %ymm3\n"));
                asm.push_str(&format!("vpermpd $27, {second}, %ymm3\n"));
                asm.push_str(&format!(
                    "vbroadcastsd {}, %ymm3\n",
                    second.replace('y', "x")
                ));
            }
            for memory in MEMORY {
                asm.push_str(&format!("vbroadcasti128 {memory}, %ymm3\n"));
                asm.push_str(&format!("vbroadcastf128 {memory}, %ymm3\n"));
                asm.push_str(&format!("vbroadcastsd {memory}, %ymm3\n"));
                asm.push_str(&format!("vlddqu {memory}, %ymm3\n"));
                asm.push_str(&format!("vmovntdqa {memory}, %xmm3\n"));
            }
            let (output, reference) = oracle(&asm, "avx_transfers");
            assert_eq!(text(&output), text(&reference));
        }

        /// The encoded bytes of the forms this pass added, pinned so a regression
        /// cannot hide behind a reference assembler that moved with it.
        #[test]
        fn extended_encodings_are_pinned() {
            const CASES: &[(&str, &str)] = &[
                ("pmaddwd %xmm1, %xmm2", "660ff5d1"),
                ("pmaddwd %xmm9, %xmm12", "66450ff5e1"),
                ("pmaddwd %xmm15, %xmm0", "66410ff5c7"),
                ("pmaddwd 8(%rbp), %xmm3", "660ff55d08"),
                ("pmaddwd (%r12), %xmm8", "66450ff50424"),
                ("pmaddwd (%r13,%rax,4), %xmm15", "66450ff57c8500"),
                ("pshufb %xmm1, %xmm2", "660f3800d1"),
                ("pshufb %xmm9, %xmm12", "66450f3800e1"),
                ("pshufb %xmm15, %xmm0", "66410f3800c7"),
                ("pshufb 8(%rbp), %xmm3", "660f38005d08"),
                ("pshufb (%r12), %xmm8", "66450f38000424"),
                ("pshufb (%r13,%rax,4), %xmm15", "66450f38007c8500"),
                ("packuswb %xmm1, %xmm2", "660f67d1"),
                ("packuswb %xmm9, %xmm12", "66450f67e1"),
                ("packuswb %xmm15, %xmm0", "66410f67c7"),
                ("packuswb 8(%rbp), %xmm3", "660f675d08"),
                ("packuswb (%r12), %xmm8", "66450f670424"),
                ("packuswb (%r13,%rax,4), %xmm15", "66450f677c8500"),
                ("packsswb %xmm1, %xmm2", "660f63d1"),
                ("packsswb %xmm9, %xmm12", "66450f63e1"),
                ("packsswb %xmm15, %xmm0", "66410f63c7"),
                ("packsswb 8(%rbp), %xmm3", "660f635d08"),
                ("packsswb (%r12), %xmm8", "66450f630424"),
                ("packsswb (%r13,%rax,4), %xmm15", "66450f637c8500"),
                ("packssdw %xmm1, %xmm2", "660f6bd1"),
                ("packssdw %xmm9, %xmm12", "66450f6be1"),
                ("packssdw %xmm15, %xmm0", "66410f6bc7"),
                ("packssdw 8(%rbp), %xmm3", "660f6b5d08"),
                ("packssdw (%r12), %xmm8", "66450f6b0424"),
                ("packssdw (%r13,%rax,4), %xmm15", "66450f6b7c8500"),
                ("packusdw %xmm1, %xmm2", "660f382bd1"),
                ("packusdw %xmm9, %xmm12", "66450f382be1"),
                ("packusdw %xmm15, %xmm0", "66410f382bc7"),
                ("packusdw 8(%rbp), %xmm3", "660f382b5d08"),
                ("packusdw (%r12), %xmm8", "66450f382b0424"),
                ("packusdw (%r13,%rax,4), %xmm15", "66450f382b7c8500"),
                ("pavgb %xmm1, %xmm2", "660fe0d1"),
                ("pavgb %xmm9, %xmm12", "66450fe0e1"),
                ("pavgb %xmm15, %xmm0", "66410fe0c7"),
                ("pavgb 8(%rbp), %xmm3", "660fe05d08"),
                ("pavgb (%r12), %xmm8", "66450fe00424"),
                ("pavgb (%r13,%rax,4), %xmm15", "66450fe07c8500"),
                ("pavgw %xmm1, %xmm2", "660fe3d1"),
                ("pavgw %xmm9, %xmm12", "66450fe3e1"),
                ("pavgw %xmm15, %xmm0", "66410fe3c7"),
                ("pavgw 8(%rbp), %xmm3", "660fe35d08"),
                ("pavgw (%r12), %xmm8", "66450fe30424"),
                ("pavgw (%r13,%rax,4), %xmm15", "66450fe37c8500"),
                ("psignb %xmm1, %xmm2", "660f3808d1"),
                ("psignb %xmm9, %xmm12", "66450f3808e1"),
                ("psignb %xmm15, %xmm0", "66410f3808c7"),
                ("psignb 8(%rbp), %xmm3", "660f38085d08"),
                ("psignb (%r12), %xmm8", "66450f38080424"),
                ("psignb (%r13,%rax,4), %xmm15", "66450f38087c8500"),
                ("psignw %xmm1, %xmm2", "660f3809d1"),
                ("psignw %xmm9, %xmm12", "66450f3809e1"),
                ("psignw %xmm15, %xmm0", "66410f3809c7"),
                ("psignw 8(%rbp), %xmm3", "660f38095d08"),
                ("psignw (%r12), %xmm8", "66450f38090424"),
                ("psignw (%r13,%rax,4), %xmm15", "66450f38097c8500"),
                ("psignd %xmm1, %xmm2", "660f380ad1"),
                ("psignd %xmm9, %xmm12", "66450f380ae1"),
                ("psignd %xmm15, %xmm0", "66410f380ac7"),
                ("psignd 8(%rbp), %xmm3", "660f380a5d08"),
                ("psignd (%r12), %xmm8", "66450f380a0424"),
                ("psignd (%r13,%rax,4), %xmm15", "66450f380a7c8500"),
                ("pabsb %xmm1, %xmm2", "660f381cd1"),
                ("pabsb %xmm9, %xmm12", "66450f381ce1"),
                ("pabsb %xmm15, %xmm0", "66410f381cc7"),
                ("pabsb 8(%rbp), %xmm3", "660f381c5d08"),
                ("pabsb (%r12), %xmm8", "66450f381c0424"),
                ("pabsb (%r13,%rax,4), %xmm15", "66450f381c7c8500"),
                ("pabsw %xmm1, %xmm2", "660f381dd1"),
                ("pabsw %xmm9, %xmm12", "66450f381de1"),
                ("pabsw %xmm15, %xmm0", "66410f381dc7"),
                ("pabsw 8(%rbp), %xmm3", "660f381d5d08"),
                ("pabsw (%r12), %xmm8", "66450f381d0424"),
                ("pabsw (%r13,%rax,4), %xmm15", "66450f381d7c8500"),
                ("pabsd %xmm1, %xmm2", "660f381ed1"),
                ("pabsd %xmm9, %xmm12", "66450f381ee1"),
                ("pabsd %xmm15, %xmm0", "66410f381ec7"),
                ("pabsd 8(%rbp), %xmm3", "660f381e5d08"),
                ("pabsd (%r12), %xmm8", "66450f381e0424"),
                ("pabsd (%r13,%rax,4), %xmm15", "66450f381e7c8500"),
                ("pminsb %xmm1, %xmm2", "660f3838d1"),
                ("pminsb %xmm9, %xmm12", "66450f3838e1"),
                ("pminsb %xmm15, %xmm0", "66410f3838c7"),
                ("pminsb 8(%rbp), %xmm3", "660f38385d08"),
                ("pminsb (%r12), %xmm8", "66450f38380424"),
                ("pminsb (%r13,%rax,4), %xmm15", "66450f38387c8500"),
                ("pminsw %xmm1, %xmm2", "660fead1"),
                ("pminsw %xmm9, %xmm12", "66450feae1"),
                ("pminsw %xmm15, %xmm0", "66410feac7"),
                ("pminsw 8(%rbp), %xmm3", "660fea5d08"),
                ("pminsw (%r12), %xmm8", "66450fea0424"),
                ("pminsw (%r13,%rax,4), %xmm15", "66450fea7c8500"),
                ("pminsd %xmm1, %xmm2", "660f3839d1"),
                ("pminsd %xmm9, %xmm12", "66450f3839e1"),
                ("pminsd %xmm15, %xmm0", "66410f3839c7"),
                ("pminsd 8(%rbp), %xmm3", "660f38395d08"),
                ("pminsd (%r12), %xmm8", "66450f38390424"),
                ("pminsd (%r13,%rax,4), %xmm15", "66450f38397c8500"),
                ("pmaxsb %xmm1, %xmm2", "660f383cd1"),
                ("pmaxsb %xmm9, %xmm12", "66450f383ce1"),
                ("pmaxsb %xmm15, %xmm0", "66410f383cc7"),
                ("pmaxsb 8(%rbp), %xmm3", "660f383c5d08"),
                ("pmaxsb (%r12), %xmm8", "66450f383c0424"),
                ("pmaxsb (%r13,%rax,4), %xmm15", "66450f383c7c8500"),
                ("pmaxsw %xmm1, %xmm2", "660feed1"),
                ("pmaxsw %xmm9, %xmm12", "66450feee1"),
                ("pmaxsw %xmm15, %xmm0", "66410feec7"),
                ("pmaxsw 8(%rbp), %xmm3", "660fee5d08"),
                ("pmaxsw (%r12), %xmm8", "66450fee0424"),
                ("pmaxsw (%r13,%rax,4), %xmm15", "66450fee7c8500"),
                ("pmaxsd %xmm1, %xmm2", "660f383dd1"),
                ("pmaxsd %xmm9, %xmm12", "66450f383de1"),
                ("pmaxsd %xmm15, %xmm0", "66410f383dc7"),
                ("pmaxsd 8(%rbp), %xmm3", "660f383d5d08"),
                ("pmaxsd (%r12), %xmm8", "66450f383d0424"),
                ("pmaxsd (%r13,%rax,4), %xmm15", "66450f383d7c8500"),
                ("pminub %xmm1, %xmm2", "660fdad1"),
                ("pminub %xmm9, %xmm12", "66450fdae1"),
                ("pminub %xmm15, %xmm0", "66410fdac7"),
                ("pminub 8(%rbp), %xmm3", "660fda5d08"),
                ("pminub (%r12), %xmm8", "66450fda0424"),
                ("pminub (%r13,%rax,4), %xmm15", "66450fda7c8500"),
                ("pminuw %xmm1, %xmm2", "660f383ad1"),
                ("pminuw %xmm9, %xmm12", "66450f383ae1"),
                ("pminuw %xmm15, %xmm0", "66410f383ac7"),
                ("pminuw 8(%rbp), %xmm3", "660f383a5d08"),
                ("pminuw (%r12), %xmm8", "66450f383a0424"),
                ("pminuw (%r13,%rax,4), %xmm15", "66450f383a7c8500"),
                ("pminud %xmm1, %xmm2", "660f383bd1"),
                ("pminud %xmm9, %xmm12", "66450f383be1"),
                ("pminud %xmm15, %xmm0", "66410f383bc7"),
                ("pminud 8(%rbp), %xmm3", "660f383b5d08"),
                ("pminud (%r12), %xmm8", "66450f383b0424"),
                ("pminud (%r13,%rax,4), %xmm15", "66450f383b7c8500"),
                ("pmaxub %xmm1, %xmm2", "660fded1"),
                ("pmaxub %xmm9, %xmm12", "66450fdee1"),
                ("pmaxub %xmm15, %xmm0", "66410fdec7"),
                ("pmaxub 8(%rbp), %xmm3", "660fde5d08"),
                ("pmaxub (%r12), %xmm8", "66450fde0424"),
                ("pmaxub (%r13,%rax,4), %xmm15", "66450fde7c8500"),
                ("pmaxuw %xmm1, %xmm2", "660f383ed1"),
                ("pmaxuw %xmm9, %xmm12", "66450f383ee1"),
                ("pmaxuw %xmm15, %xmm0", "66410f383ec7"),
                ("pmaxuw 8(%rbp), %xmm3", "660f383e5d08"),
                ("pmaxuw (%r12), %xmm8", "66450f383e0424"),
                ("pmaxuw (%r13,%rax,4), %xmm15", "66450f383e7c8500"),
                ("pmaxud %xmm1, %xmm2", "660f383fd1"),
                ("pmaxud %xmm9, %xmm12", "66450f383fe1"),
                ("pmaxud %xmm15, %xmm0", "66410f383fc7"),
                ("pmaxud 8(%rbp), %xmm3", "660f383f5d08"),
                ("pmaxud (%r12), %xmm8", "66450f383f0424"),
                ("pmaxud (%r13,%rax,4), %xmm15", "66450f383f7c8500"),
                ("pcmpeqq %xmm1, %xmm2", "660f3829d1"),
                ("pcmpeqq %xmm9, %xmm12", "66450f3829e1"),
                ("pcmpeqq %xmm15, %xmm0", "66410f3829c7"),
                ("pcmpeqq 8(%rbp), %xmm3", "660f38295d08"),
                ("pcmpeqq (%r12), %xmm8", "66450f38290424"),
                ("pcmpeqq (%r13,%rax,4), %xmm15", "66450f38297c8500"),
                ("pcmpgtq %xmm1, %xmm2", "660f3837d1"),
                ("pcmpgtq %xmm9, %xmm12", "66450f3837e1"),
                ("pcmpgtq %xmm15, %xmm0", "66410f3837c7"),
                ("pcmpgtq 8(%rbp), %xmm3", "660f38375d08"),
                ("pcmpgtq (%r12), %xmm8", "66450f38370424"),
                ("pcmpgtq (%r13,%rax,4), %xmm15", "66450f38377c8500"),
                ("pmuldq %xmm1, %xmm2", "660f3828d1"),
                ("pmuldq %xmm9, %xmm12", "66450f3828e1"),
                ("pmuldq %xmm15, %xmm0", "66410f3828c7"),
                ("pmuldq 8(%rbp), %xmm3", "660f38285d08"),
                ("pmuldq (%r12), %xmm8", "66450f38280424"),
                ("pmuldq (%r13,%rax,4), %xmm15", "66450f38287c8500"),
                ("pmulhw %xmm1, %xmm2", "660fe5d1"),
                ("pmulhw %xmm9, %xmm12", "66450fe5e1"),
                ("pmulhw %xmm15, %xmm0", "66410fe5c7"),
                ("pmulhw 8(%rbp), %xmm3", "660fe55d08"),
                ("pmulhw (%r12), %xmm8", "66450fe50424"),
                ("pmulhw (%r13,%rax,4), %xmm15", "66450fe57c8500"),
                ("pmulhuw %xmm1, %xmm2", "660fe4d1"),
                ("pmulhuw %xmm9, %xmm12", "66450fe4e1"),
                ("pmulhuw %xmm15, %xmm0", "66410fe4c7"),
                ("pmulhuw 8(%rbp), %xmm3", "660fe45d08"),
                ("pmulhuw (%r12), %xmm8", "66450fe40424"),
                ("pmulhuw (%r13,%rax,4), %xmm15", "66450fe47c8500"),
                ("pmulhrsw %xmm1, %xmm2", "660f380bd1"),
                ("pmulhrsw %xmm9, %xmm12", "66450f380be1"),
                ("pmulhrsw %xmm15, %xmm0", "66410f380bc7"),
                ("pmulhrsw 8(%rbp), %xmm3", "660f380b5d08"),
                ("pmulhrsw (%r12), %xmm8", "66450f380b0424"),
                ("pmulhrsw (%r13,%rax,4), %xmm15", "66450f380b7c8500"),
                ("pmaddubsw %xmm1, %xmm2", "660f3804d1"),
                ("pmaddubsw %xmm9, %xmm12", "66450f3804e1"),
                ("pmaddubsw %xmm15, %xmm0", "66410f3804c7"),
                ("pmaddubsw 8(%rbp), %xmm3", "660f38045d08"),
                ("pmaddubsw (%r12), %xmm8", "66450f38040424"),
                ("pmaddubsw (%r13,%rax,4), %xmm15", "66450f38047c8500"),
                ("psadbw %xmm1, %xmm2", "660ff6d1"),
                ("psadbw %xmm9, %xmm12", "66450ff6e1"),
                ("psadbw %xmm15, %xmm0", "66410ff6c7"),
                ("psadbw 8(%rbp), %xmm3", "660ff65d08"),
                ("psadbw (%r12), %xmm8", "66450ff60424"),
                ("psadbw (%r13,%rax,4), %xmm15", "66450ff67c8500"),
                ("phaddw %xmm1, %xmm2", "660f3801d1"),
                ("phaddw %xmm9, %xmm12", "66450f3801e1"),
                ("phaddw %xmm15, %xmm0", "66410f3801c7"),
                ("phaddw 8(%rbp), %xmm3", "660f38015d08"),
                ("phaddw (%r12), %xmm8", "66450f38010424"),
                ("phaddw (%r13,%rax,4), %xmm15", "66450f38017c8500"),
                ("phaddd %xmm1, %xmm2", "660f3802d1"),
                ("phaddd %xmm9, %xmm12", "66450f3802e1"),
                ("phaddd %xmm15, %xmm0", "66410f3802c7"),
                ("phaddd 8(%rbp), %xmm3", "660f38025d08"),
                ("phaddd (%r12), %xmm8", "66450f38020424"),
                ("phaddd (%r13,%rax,4), %xmm15", "66450f38027c8500"),
                ("phaddsw %xmm1, %xmm2", "660f3803d1"),
                ("phaddsw %xmm9, %xmm12", "66450f3803e1"),
                ("phaddsw %xmm15, %xmm0", "66410f3803c7"),
                ("phaddsw 8(%rbp), %xmm3", "660f38035d08"),
                ("phaddsw (%r12), %xmm8", "66450f38030424"),
                ("phaddsw (%r13,%rax,4), %xmm15", "66450f38037c8500"),
                ("phsubw %xmm1, %xmm2", "660f3805d1"),
                ("phsubw %xmm9, %xmm12", "66450f3805e1"),
                ("phsubw %xmm15, %xmm0", "66410f3805c7"),
                ("phsubw 8(%rbp), %xmm3", "660f38055d08"),
                ("phsubw (%r12), %xmm8", "66450f38050424"),
                ("phsubw (%r13,%rax,4), %xmm15", "66450f38057c8500"),
                ("phsubd %xmm1, %xmm2", "660f3806d1"),
                ("phsubd %xmm9, %xmm12", "66450f3806e1"),
                ("phsubd %xmm15, %xmm0", "66410f3806c7"),
                ("phsubd 8(%rbp), %xmm3", "660f38065d08"),
                ("phsubd (%r12), %xmm8", "66450f38060424"),
                ("phsubd (%r13,%rax,4), %xmm15", "66450f38067c8500"),
                ("phsubsw %xmm1, %xmm2", "660f3807d1"),
                ("phsubsw %xmm9, %xmm12", "66450f3807e1"),
                ("phsubsw %xmm15, %xmm0", "66410f3807c7"),
                ("phsubsw 8(%rbp), %xmm3", "660f38075d08"),
                ("phsubsw (%r12), %xmm8", "66450f38070424"),
                ("phsubsw (%r13,%rax,4), %xmm15", "66450f38077c8500"),
                ("pmovsxbw %xmm1, %xmm2", "660f3820d1"),
                ("pmovsxbw %xmm9, %xmm12", "66450f3820e1"),
                ("pmovsxbw %xmm15, %xmm0", "66410f3820c7"),
                ("pmovsxbw 8(%rbp), %xmm3", "660f38205d08"),
                ("pmovsxbw (%r12), %xmm8", "66450f38200424"),
                ("pmovsxbw (%r13,%rax,4), %xmm15", "66450f38207c8500"),
                ("pmovsxdq %xmm1, %xmm2", "660f3825d1"),
                ("pmovsxdq %xmm9, %xmm12", "66450f3825e1"),
                ("pmovsxdq %xmm15, %xmm0", "66410f3825c7"),
                ("pmovsxdq 8(%rbp), %xmm3", "660f38255d08"),
                ("pmovsxdq (%r12), %xmm8", "66450f38250424"),
                ("pmovsxdq (%r13,%rax,4), %xmm15", "66450f38257c8500"),
                ("pmovzxbw %xmm1, %xmm2", "660f3830d1"),
                ("pmovzxbw %xmm9, %xmm12", "66450f3830e1"),
                ("pmovzxbw %xmm15, %xmm0", "66410f3830c7"),
                ("pmovzxbw 8(%rbp), %xmm3", "660f38305d08"),
                ("pmovzxbw (%r12), %xmm8", "66450f38300424"),
                ("pmovzxbw (%r13,%rax,4), %xmm15", "66450f38307c8500"),
                ("pmovzxdq %xmm1, %xmm2", "660f3835d1"),
                ("pmovzxdq %xmm9, %xmm12", "66450f3835e1"),
                ("pmovzxdq %xmm15, %xmm0", "66410f3835c7"),
                ("pmovzxdq 8(%rbp), %xmm3", "660f38355d08"),
                ("pmovzxdq (%r12), %xmm8", "66450f38350424"),
                ("pmovzxdq (%r13,%rax,4), %xmm15", "66450f38357c8500"),
                ("ptest %xmm1, %xmm2", "660f3817d1"),
                ("ptest %xmm9, %xmm12", "66450f3817e1"),
                ("ptest %xmm15, %xmm0", "66410f3817c7"),
                ("ptest 8(%rbp), %xmm3", "660f38175d08"),
                ("ptest (%r12), %xmm8", "66450f38170424"),
                ("ptest (%r13,%rax,4), %xmm15", "66450f38177c8500"),
                ("phminposuw %xmm1, %xmm2", "660f3841d1"),
                ("phminposuw %xmm9, %xmm12", "66450f3841e1"),
                ("phminposuw %xmm15, %xmm0", "66410f3841c7"),
                ("phminposuw 8(%rbp), %xmm3", "660f38415d08"),
                ("phminposuw (%r12), %xmm8", "66450f38410424"),
                ("phminposuw (%r13,%rax,4), %xmm15", "66450f38417c8500"),
                ("paddsb %xmm1, %xmm2", "660fecd1"),
                ("paddsb %xmm9, %xmm12", "66450fece1"),
                ("paddsb %xmm15, %xmm0", "66410fecc7"),
                ("paddsb 8(%rbp), %xmm3", "660fec5d08"),
                ("paddsb (%r12), %xmm8", "66450fec0424"),
                ("paddsb (%r13,%rax,4), %xmm15", "66450fec7c8500"),
                ("paddusw %xmm1, %xmm2", "660fddd1"),
                ("paddusw %xmm9, %xmm12", "66450fdde1"),
                ("paddusw %xmm15, %xmm0", "66410fddc7"),
                ("paddusw 8(%rbp), %xmm3", "660fdd5d08"),
                ("paddusw (%r12), %xmm8", "66450fdd0424"),
                ("paddusw (%r13,%rax,4), %xmm15", "66450fdd7c8500"),
                ("psubsb %xmm1, %xmm2", "660fe8d1"),
                ("psubsb %xmm9, %xmm12", "66450fe8e1"),
                ("psubsb %xmm15, %xmm0", "66410fe8c7"),
                ("psubsb 8(%rbp), %xmm3", "660fe85d08"),
                ("psubsb (%r12), %xmm8", "66450fe80424"),
                ("psubsb (%r13,%rax,4), %xmm15", "66450fe87c8500"),
                ("psubusw %xmm1, %xmm2", "660fd9d1"),
                ("psubusw %xmm9, %xmm12", "66450fd9e1"),
                ("psubusw %xmm15, %xmm0", "66410fd9c7"),
                ("psubusw 8(%rbp), %xmm3", "660fd95d08"),
                ("psubusw (%r12), %xmm8", "66450fd90424"),
                ("psubusw (%r13,%rax,4), %xmm15", "66450fd97c8500"),
                ("aesenc %xmm1, %xmm2", "660f38dcd1"),
                ("aesenc %xmm9, %xmm12", "66450f38dce1"),
                ("aesenc %xmm15, %xmm0", "66410f38dcc7"),
                ("aesenc 8(%rbp), %xmm3", "660f38dc5d08"),
                ("aesenc (%r12), %xmm8", "66450f38dc0424"),
                ("aesenc (%r13,%rax,4), %xmm15", "66450f38dc7c8500"),
                ("aesdeclast %xmm1, %xmm2", "660f38dfd1"),
                ("aesdeclast %xmm9, %xmm12", "66450f38dfe1"),
                ("aesdeclast %xmm15, %xmm0", "66410f38dfc7"),
                ("aesdeclast 8(%rbp), %xmm3", "660f38df5d08"),
                ("aesdeclast (%r12), %xmm8", "66450f38df0424"),
                ("aesdeclast (%r13,%rax,4), %xmm15", "66450f38df7c8500"),
                ("aesimc %xmm1, %xmm2", "660f38dbd1"),
                ("aesimc %xmm9, %xmm12", "66450f38dbe1"),
                ("aesimc %xmm15, %xmm0", "66410f38dbc7"),
                ("aesimc 8(%rbp), %xmm3", "660f38db5d08"),
                ("aesimc (%r12), %xmm8", "66450f38db0424"),
                ("aesimc (%r13,%rax,4), %xmm15", "66450f38db7c8500"),
                ("maskmovdqu %xmm1, %xmm2", "660ff7d1"),
                ("maskmovdqu %xmm9, %xmm12", "66450ff7e1"),
                ("maskmovdqu %xmm15, %xmm0", "66410ff7c7"),
                ("blendvps %xmm0, %xmm1, %xmm2", "660f3814d1"),
                ("blendvps %xmm0, %xmm9, %xmm12", "66450f3814e1"),
                ("blendvps %xmm0, (%r13), %xmm8", "66450f38144500"),
                ("blendvpd %xmm0, %xmm1, %xmm2", "660f3815d1"),
                ("blendvpd %xmm0, %xmm9, %xmm12", "66450f3815e1"),
                ("blendvpd %xmm0, (%r13), %xmm8", "66450f38154500"),
                ("pblendvb %xmm0, %xmm1, %xmm2", "660f3810d1"),
                ("pblendvb %xmm0, %xmm9, %xmm12", "66450f3810e1"),
                ("pblendvb %xmm0, (%r13), %xmm8", "66450f38104500"),
                ("pblendw $255, %xmm1, %xmm2", "660f3a0ed1ff"),
                ("pblendw $255, %xmm9, %xmm12", "66450f3a0ee1ff"),
                ("pblendw $255, (%r12), %xmm8", "66450f3a0e0424ff"),
                ("palignr $15, %xmm1, %xmm2", "660f3a0fd10f"),
                ("palignr $15, %xmm9, %xmm12", "66450f3a0fe10f"),
                ("palignr $15, (%r12), %xmm8", "66450f3a0f04240f"),
                ("pclmulqdq $17, %xmm1, %xmm2", "660f3a44d111"),
                ("pclmulqdq $17, %xmm9, %xmm12", "66450f3a44e111"),
                ("pclmulqdq $17, (%r12), %xmm8", "66450f3a44042411"),
                ("blendps $15, %xmm1, %xmm2", "660f3a0cd10f"),
                ("blendps $15, %xmm9, %xmm12", "66450f3a0ce10f"),
                ("blendps $15, (%r12), %xmm8", "66450f3a0c04240f"),
                ("blendpd $3, %xmm1, %xmm2", "660f3a0dd103"),
                ("blendpd $3, %xmm9, %xmm12", "66450f3a0de103"),
                ("blendpd $3, (%r12), %xmm8", "66450f3a0d042403"),
                ("insertps $78, %xmm1, %xmm2", "660f3a21d14e"),
                ("insertps $78, %xmm9, %xmm12", "66450f3a21e14e"),
                ("insertps $78, (%r12), %xmm8", "66450f3a2104244e"),
                ("dpps $255, %xmm1, %xmm2", "660f3a40d1ff"),
                ("dpps $255, %xmm9, %xmm12", "66450f3a40e1ff"),
                ("dpps $255, (%r12), %xmm8", "66450f3a400424ff"),
                ("dppd $51, %xmm1, %xmm2", "660f3a41d133"),
                ("dppd $51, %xmm9, %xmm12", "66450f3a41e133"),
                ("dppd $51, (%r12), %xmm8", "66450f3a41042433"),
                ("mpsadbw $7, %xmm1, %xmm2", "660f3a42d107"),
                ("mpsadbw $7, %xmm9, %xmm12", "66450f3a42e107"),
                ("mpsadbw $7, (%r12), %xmm8", "66450f3a42042407"),
                ("roundps $11, %xmm1, %xmm2", "660f3a08d10b"),
                ("roundps $11, %xmm9, %xmm12", "66450f3a08e10b"),
                ("roundps $11, (%r12), %xmm8", "66450f3a0804240b"),
                ("roundpd $2, %xmm1, %xmm2", "660f3a09d102"),
                ("roundpd $2, %xmm9, %xmm12", "66450f3a09e102"),
                ("roundpd $2, (%r12), %xmm8", "66450f3a09042402"),
                ("roundss $3, %xmm1, %xmm2", "660f3a0ad103"),
                ("roundss $3, %xmm9, %xmm12", "66450f3a0ae103"),
                ("roundss $3, (%r12), %xmm8", "66450f3a0a042403"),
                ("roundsd $4, %xmm1, %xmm2", "660f3a0bd104"),
                ("roundsd $4, %xmm9, %xmm12", "66450f3a0be104"),
                ("roundsd $4, (%r12), %xmm8", "66450f3a0b042404"),
                ("cmpps $7, %xmm1, %xmm2", "0fc2d107"),
                ("cmpps $7, %xmm9, %xmm12", "450fc2e107"),
                ("cmpps $7, (%r12), %xmm8", "450fc2042407"),
                ("cmppd $7, %xmm1, %xmm2", "660fc2d107"),
                ("cmppd $7, %xmm9, %xmm12", "66450fc2e107"),
                ("cmppd $7, (%r12), %xmm8", "66450fc2042407"),
                ("aeskeygenassist $255, %xmm1, %xmm2", "660f3adfd1ff"),
                ("aeskeygenassist $255, %xmm9, %xmm12", "66450f3adfe1ff"),
                ("aeskeygenassist $255, (%r12), %xmm8", "66450f3adf0424ff"),
                ("pcmpistri $1, %xmm1, %xmm2", "660f3a63d101"),
                ("pcmpistri $1, %xmm9, %xmm12", "66450f3a63e101"),
                ("pcmpistri $1, (%r12), %xmm8", "66450f3a63042401"),
                ("pcmpestrm $1, %xmm1, %xmm2", "660f3a60d101"),
                ("pcmpestrm $1, %xmm9, %xmm12", "66450f3a60e101"),
                ("pcmpestrm $1, (%r12), %xmm8", "66450f3a60042401"),
                ("movntdq %xmm1, (%rax)", "660fe708"),
                ("movntdq %xmm9, 8(%r12)", "66450fe74c2408"),
                ("movntps %xmm1, (%rax)", "0f2b08"),
                ("movntps %xmm9, 8(%r12)", "450f2b4c2408"),
                ("movntpd %xmm1, (%rax)", "660f2b08"),
                ("movntpd %xmm9, 8(%r12)", "66450f2b4c2408"),
                ("lddqu (%rax), %xmm1", "f20ff008"),
                ("lddqu (%r13), %xmm9", "f2450ff04d00"),
                ("movntdqa (%rax), %xmm1", "660f382a08"),
                ("movntdqa (%r12,%rax,8), %xmm15", "66450f382a3cc4"),
                ("pmovmskb %xmm1, %eax", "660fd7c1"),
                ("pmovmskb %xmm15, %r9d", "66450fd7cf"),
                ("movmskps %xmm1, %eax", "0f50c1"),
                ("movmskps %xmm15, %r9d", "450f50cf"),
                ("movmskpd %xmm1, %eax", "660f50c1"),
                ("movmskpd %xmm15, %r9d", "66450f50cf"),
                ("pextrw $0, %xmm1, %eax", "660fc5c100"),
                ("pextrw $0, %xmm9, %r12d", "66450fc5e100"),
                ("pextrw $0, %xmm1, (%rax)", "660f3a150800"),
                ("pextrw $0, %xmm9, (%r13)", "66450f3a154d0000"),
                ("pextrw $0, %xmm15, 1024(%rbx)", "66440f3a15bb0004000000"),
                ("pextrw $0, %xmm8, (%r12,%r13,2)", "66470f3a15046c00"),
                ("pextrw $7, %xmm1, %eax", "660fc5c107"),
                ("pextrw $7, %xmm9, %r12d", "66450fc5e107"),
                ("pextrw $7, %xmm1, (%rax)", "660f3a150807"),
                ("pextrw $7, %xmm9, (%r13)", "66450f3a154d0007"),
                ("pextrw $7, %xmm15, 1024(%rbx)", "66440f3a15bb0004000007"),
                ("pextrw $7, %xmm8, (%r12,%r13,2)", "66470f3a15046c07"),
                ("extractps $1, %xmm1, %eax", "660f3a17c801"),
                ("extractps $3, %xmm9, (%r13)", "66450f3a174d0003"),
                ("crc32b %al, %eax", "f20f38f0c0"),
                ("crc32b %r9b, %r12d", "f2450f38f0e1"),
                ("crc32w %ax, %ecx", "66f20f38f1c8"),
                ("crc32l (%r13), %eax", "f2410f38f14500"),
                ("crc32q %rax, %rcx", "f2480f38f1c8"),
                ("crc32q (%r12), %r15", "f24d0f38f13c24"),
                ("movntil %eax, (%rax)", "0fc300"),
                ("movntiq %r9, 8(%r13)", "4d0fc34d08"),
                ("prefetcht0 (%rax)", "0f1808"),
                ("prefetchnta (%r12)", "410f180424"),
                ("prefetcht2 8(%r13)", "410f185d08"),
                ("ldmxcsr (%rax)", "0fae10"),
                ("stmxcsr (%r13)", "410fae5d00"),
                ("sfence", "0faef8"),
                ("lfence", "0faee8"),
                ("pause", "f390"),
                ("cvtss2si %xmm1, %eax", "f30f2dc1"),
                ("cvtsd2si %xmm9, %r12", "f24d0f2de1"),
                ("cvttss2si %xmm15, %rax", "f3490f2cc7"),
                ("rcpss %xmm1, %xmm2", "f30f53d1"),
                ("rsqrtss (%r13), %xmm9", "f3450f524d00"),
                ("movlps (%rax), %xmm1", "0f1208"),
                ("movlps %xmm9, (%r13)", "450f134d00"),
                ("movhpd (%rax), %xmm1", "660f1608"),
                ("movhpd %xmm9, (%r13)", "66450f174d00"),
                ("movupd (%rax), %xmm1", "660f1008"),
                ("movupd %xmm9, (%r13)", "66450f114d00"),
                ("movlhps %xmm1, %xmm2", "0f16d1"),
                ("movlhps %xmm15, %xmm8", "450f16c7"),
                ("vpsllvd %xmm1, %xmm2, %xmm3", "c4e26947d9"),
                ("vpsllvd %xmm9, %xmm12, %xmm15", "c4421947f9"),
                ("vpsllvd %ymm1, %ymm2, %ymm3", "c4e26d47d9"),
                ("vpsllvd %ymm9, %ymm12, %ymm15", "c4421d47f9"),
                ("vpsllvd (%r13), %ymm2, %ymm3", "c4c26d475d00"),
                ("vpsllvd (%r12,%rax,8), %xmm2, %xmm3", "c4c269471cc4"),
                ("vpsrlvd %xmm1, %xmm2, %xmm3", "c4e26945d9"),
                ("vpsrlvd %xmm9, %xmm12, %xmm15", "c4421945f9"),
                ("vpsrlvd %ymm1, %ymm2, %ymm3", "c4e26d45d9"),
                ("vpsrlvd %ymm9, %ymm12, %ymm15", "c4421d45f9"),
                ("vpsrlvd (%r13), %ymm2, %ymm3", "c4c26d455d00"),
                ("vpsrlvd (%r12,%rax,8), %xmm2, %xmm3", "c4c269451cc4"),
                ("vpsravd %xmm1, %xmm2, %xmm3", "c4e26946d9"),
                ("vpsravd %xmm9, %xmm12, %xmm15", "c4421946f9"),
                ("vpsravd %ymm1, %ymm2, %ymm3", "c4e26d46d9"),
                ("vpsravd %ymm9, %ymm12, %ymm15", "c4421d46f9"),
                ("vpsravd (%r13), %ymm2, %ymm3", "c4c26d465d00"),
                ("vpsravd (%r12,%rax,8), %xmm2, %xmm3", "c4c269461cc4"),
                ("vpsllvq %xmm1, %xmm2, %xmm3", "c4e2e947d9"),
                ("vpsllvq %xmm9, %xmm12, %xmm15", "c4429947f9"),
                ("vpsllvq %ymm1, %ymm2, %ymm3", "c4e2ed47d9"),
                ("vpsllvq %ymm9, %ymm12, %ymm15", "c4429d47f9"),
                ("vpsllvq (%r13), %ymm2, %ymm3", "c4c2ed475d00"),
                ("vpsllvq (%r12,%rax,8), %xmm2, %xmm3", "c4c2e9471cc4"),
                ("vpsrlvq %xmm1, %xmm2, %xmm3", "c4e2e945d9"),
                ("vpsrlvq %xmm9, %xmm12, %xmm15", "c4429945f9"),
                ("vpsrlvq %ymm1, %ymm2, %ymm3", "c4e2ed45d9"),
                ("vpsrlvq %ymm9, %ymm12, %ymm15", "c4429d45f9"),
                ("vpsrlvq (%r13), %ymm2, %ymm3", "c4c2ed455d00"),
                ("vpsrlvq (%r12,%rax,8), %xmm2, %xmm3", "c4c2e9451cc4"),
                ("vpaddb %xmm1, %xmm2, %xmm3", "c5e9fcd9"),
                ("vpaddb %ymm9, %ymm12, %ymm15", "c4411dfcf9"),
                ("vpaddb (%r13), %ymm2, %ymm3", "c4c16dfc5d00"),
                ("vpand %xmm1, %xmm2, %xmm3", "c5e9dbd9"),
                ("vpand %ymm9, %ymm12, %ymm15", "c4411ddbf9"),
                ("vpand (%r13), %ymm2, %ymm3", "c4c16ddb5d00"),
                ("vpshufb %xmm1, %xmm2, %xmm3", "c4e26900d9"),
                ("vpshufb %ymm9, %ymm12, %ymm15", "c4421d00f9"),
                ("vpshufb (%r13), %ymm2, %ymm3", "c4c26d005d00"),
                ("vpminsd %xmm1, %xmm2, %xmm3", "c4e26939d9"),
                ("vpminsd %ymm9, %ymm12, %ymm15", "c4421d39f9"),
                ("vpminsd (%r13), %ymm2, %ymm3", "c4c26d395d00"),
                ("vpmulld %xmm1, %xmm2, %xmm3", "c4e26940d9"),
                ("vpmulld %ymm9, %ymm12, %ymm15", "c4421d40f9"),
                ("vpmulld (%r13), %ymm2, %ymm3", "c4c26d405d00"),
                ("vaddps %xmm1, %xmm2, %xmm3", "c5e858d9"),
                ("vaddps %ymm9, %ymm12, %ymm15", "c4411c58f9"),
                ("vaddps (%r13), %ymm2, %ymm3", "c4c16c585d00"),
                ("vpcmpeqq %xmm1, %xmm2, %xmm3", "c4e26929d9"),
                ("vpcmpeqq %ymm9, %ymm12, %ymm15", "c4421d29f9"),
                ("vpcmpeqq (%r13), %ymm2, %ymm3", "c4c26d295d00"),
                ("vpsadbw %xmm1, %xmm2, %xmm3", "c5e9f6d9"),
                ("vpsadbw %ymm9, %ymm12, %ymm15", "c4411df6f9"),
                ("vpsadbw (%r13), %ymm2, %ymm3", "c4c16df65d00"),
                ("vpabsd %xmm1, %xmm2", "c4e2791ed1"),
                ("vpabsd (%r13), %ymm15", "c4427d1e7d00"),
                ("vptest %xmm1, %xmm2", "c4e27917d1"),
                ("vptest (%r13), %ymm15", "c4427d177d00"),
                ("vsqrtpd %xmm1, %xmm2", "c5f951d1"),
                ("vsqrtpd (%r13), %ymm15", "c4417d517d00"),
                ("vpmovzxbw %xmm1, %xmm2", "c4e27930d1"),
                ("vpmovzxbw %xmm9, %ymm15", "c4427d30f9"),
                ("vbroadcastss %xmm1, %xmm2", "c4e27918d1"),
                ("vbroadcastss %xmm9, %ymm15", "c4427d18f9"),
                ("vpbroadcastd %xmm1, %xmm2", "c4e27958d1"),
                ("vpbroadcastd %xmm9, %ymm15", "c4427d58f9"),
                ("vmovntdqa (%rax), %xmm2", "c4e2792a10"),
                ("vmovntdqa (%r13), %ymm15", "c4427d2a7d00"),
                ("vlddqu (%rax), %xmm2", "c5fbf010"),
                ("vlddqu (%r13), %ymm15", "c4417ff07d00"),
                ("vblendvps %xmm4, %xmm1, %xmm2, %xmm3", "c4e3694ad940"),
                ("vpblendvb %ymm12, %ymm9, %ymm15, %ymm1", "c4c3054cc9c0"),
                ("vpalignr $15, %ymm1, %ymm2, %ymm3", "c4e36d0fd90f"),
                ("vroundps $11, %ymm1, %ymm2", "c4e37d08d10b"),
                ("vinserti128 $1, %xmm1, %ymm2, %ymm3", "c4e36d38d901"),
                ("vextracti128 $1, %ymm9, %xmm12", "c4437d39cc01"),
                ("vextractf128 $1, %ymm1, (%r13)", "c4c37d194d0001"),
                ("vperm2i128 $1, %ymm1, %ymm2, %ymm3", "c4e36d46d901"),
                ("vpermq $27, %ymm9, %ymm15", "c443fd00f91b"),
                ("vpermd %ymm1, %ymm2, %ymm3", "c4e26d36d9"),
                ("vpermilps $27, %ymm1, %ymm2", "c4e37d04d11b"),
                ("vpermilps %ymm1, %ymm2, %ymm3", "c4e26d0cd9"),
                ("vpsllw $7, %ymm1, %ymm2", "c5ed71f107"),
                ("vpsrldq $4, %xmm9, %xmm12", "c4c11973d904"),
                ("vpextrw $3, %xmm1, %eax", "c5f9c5c103"),
                ("vpextrw $3, %xmm9, (%r13)", "c44379154d0003"),
                ("vpextrb $15, %xmm15, (%r12)", "c44379143c240f"),
                ("vpextrq $1, %xmm9, %r12", "c443f916cc01"),
                ("vpinsrb $1, %eax, %xmm1, %xmm2", "c4e37120d001"),
                ("vpinsrq $1, %r12, %xmm9, %xmm15", "c443b122fc01"),
                ("vpinsrw $7, (%r13), %xmm1, %xmm2", "c4c171c4550007"),
                ("vextractps $3, %xmm9, %r12d", "c4437917cc03"),
                ("vpmovmskb %ymm9, %r12d", "c4417dd7e1"),
                ("vmovmskpd %xmm15, %eax", "c4c17950c7"),
                ("vmovq %xmm8, %xmm0", "c579d6c0"),
                ("vmovq %rax, %xmm9", "c461f96ec8"),
                ("vmovq %xmm9, (%r13)", "c44179d64d00"),
                ("vmovq (%r12), %xmm15", "c4417a7e3c24"),
                ("vmovss %xmm15, %xmm7, %xmm0", "c54211f8"),
                ("vmovsd (%r13), %xmm9", "c4417b104d00"),
                ("vmovss %xmm9, (%r12)", "c4417a110c24"),
                ("vmovhps (%r13), %xmm1, %xmm2", "c4c170165500"),
                ("vmovlpd %xmm9, (%r12)", "c44179130c24"),
                ("vcvtsi2sdl %eax, %xmm1, %xmm2", "c5f32ad0"),
                ("vcvtsi2ssq %r12, %xmm9, %xmm15", "c441b22afc"),
                ("vcvttsd2si %xmm9, %r12", "c441fb2ce1"),
                ("vcvtss2si (%r13), %eax", "c4c17a2d4500"),
                ("vfmadd132ps %ymm1, %ymm2, %ymm3", "c4e26d98d9"),
                ("vfmadd213sd %xmm9, %xmm12, %xmm15", "c44299a9f9"),
                ("vfnmsub231pd (%r13), %ymm2, %ymm3", "c4c2edbe5d00"),
                ("vfmaddsub132ps %ymm9, %ymm12, %ymm15", "c4421d96f9"),
                ("vmaskmovps (%r13), %ymm1, %ymm2", "c4c2752c5500"),
                ("vpmaskmovq %xmm9, %xmm12, (%r12)", "c442998e0c24"),
                ("vzeroall", "c5fc77"),
                ("vzeroupper", "c5f877"),
                ("vmovaps %ymm9, (%r13)", "c4417c294d00"),
                ("vmovupd (%r12), %ymm15", "c4417d103c24"),
                ("vmovntdq %ymm9, (%r12)", "c4417de70c24"),
            ];
            for (asm, expected) in CASES {
                let tmp = TempDir::new("pinned_extended");
                let path = tmp.path().join("pinned.o");
                Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                    .unwrap()
                    .assemble(&format!(".text\n{asm}\n"), &path)
                    .unwrap();
                let bytes = std::fs::read(&path).unwrap();
                let encoded: String = text(&bytes).iter().map(|b| format!("{b:02x}")).collect();
                assert_eq!(&encoded, expected, "{asm}");
            }
        }

        /// Forms GNU as refuses are not quietly encoded as something else.
        #[test]
        fn invalid_packed_forms_are_rejected() {
            for asm in [
                "movntdq %xmm1, %xmm2",
                "movntps %xmm1, %xmm2",
                "lddqu %xmm1, %xmm2",
                "movntdqa %xmm1, %xmm2",
                "vlddqu %xmm1, %xmm2",
                "pblendvb %xmm1, %xmm2, %xmm3",
                "blendvps %xmm4, %xmm2, %xmm3",
                "extractps $1, %eax, %xmm1",
                "vpsllw %ymm1, %ymm2, %ymm3",
                "vpsrad %ymm1, %ymm2, %ymm3",
                "pmovmskb %xmm1, %rax",
                "prefetcht0 %xmm1",
                "crc32b %eax, %eax",
                "crc32q %eax, %rax",
                "cvtss2si %xmm1, %ax",
            ] {
                let tmp = TempDir::new("invalid_packed");
                let path = tmp.path().join("invalid.o");
                assert!(
                    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                        .unwrap()
                        .assemble(&format!(".text\n{asm}\n"), &path)
                        .is_err(),
                    "{asm}"
                );
            }
        }

        /// Encodings taken from GNU as 2.42, pinned so the comparison does not need
        /// the reference assembler present.
        const EVEX_FORMS: &[(&str, &[u8])] = &[
            (
                "vpaddd %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0xd9],
            ),
            (
                "vpaddq %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0xed, 0x48, 0xd4, 0xd9],
            ),
            (
                "vpaddb %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfc, 0xd9],
            ),
            (
                "vpaddd %zmm17, %zmm22, %zmm31",
                &[0x62, 0x21, 0x4d, 0x40, 0xfe, 0xf9],
            ),
            (
                "vpaddd %zmm31, %zmm16, %zmm0",
                &[0x62, 0x91, 0x7d, 0x40, 0xfe, 0xc7],
            ),
            (
                "vaddps %xmm17, %xmm2, %xmm3",
                &[0x62, 0xb1, 0x6c, 0x08, 0x58, 0xd9],
            ),
            (
                "vaddps %ymm17, %ymm2, %ymm3",
                &[0x62, 0xb1, 0x6c, 0x28, 0x58, 0xd9],
            ),
            (
                "vaddpd %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0xed, 0x48, 0x58, 0xd9],
            ),
            (
                "vpaddd %zmm1, %zmm2, %zmm3{%k1}",
                &[0x62, 0xf1, 0x6d, 0x49, 0xfe, 0xd9],
            ),
            (
                "vpaddd %zmm1, %zmm2, %zmm3{%k7}{z}",
                &[0x62, 0xf1, 0x6d, 0xcf, 0xfe, 0xd9],
            ),
            (
                "vpaddd (%rax), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x18],
            ),
            (
                "vpaddd 64(%rax), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x58, 0x01],
            ),
            (
                "vpaddd -64(%rax), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x58, 0xff],
            ),
            (
                "vpaddd 63(%rax), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x98, 0x3f, 0x00, 0x00, 0x00],
            ),
            (
                "vpaddd 8128(%rax), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x58, 0x7f],
            ),
            (
                "vpaddd 8192(%rax), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x98, 0x00, 0x20, 0x00, 0x00],
            ),
            (
                "vpaddd -8192(%rax), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x58, 0x80],
            ),
            (
                "vpaddd -8256(%rax), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x98, 0xc0, 0xdf, 0xff, 0xff],
            ),
            (
                "vpaddd (%r12), %zmm2, %zmm3",
                &[0x62, 0xd1, 0x6d, 0x48, 0xfe, 0x1c, 0x24],
            ),
            (
                "vpaddd (%r13), %zmm2, %zmm3",
                &[0x62, 0xd1, 0x6d, 0x48, 0xfe, 0x5d, 0x00],
            ),
            (
                "vpaddd (%rsp), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x1c, 0x24],
            ),
            (
                "vpaddd (%rbp), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x5d, 0x00],
            ),
            (
                "vpaddd 64(%r13,%r14,8), %zmm2, %zmm3",
                &[0x62, 0x91, 0x6d, 0x48, 0xfe, 0x5c, 0xf5, 0x01],
            ),
            (
                "vpaddd -64(%r12,%r14,8), %zmm30, %zmm31{%k5}",
                &[0x62, 0x01, 0x0d, 0x45, 0xfe, 0x7c, 0xf4, 0xff],
            ),
            (
                "vpaddd sym(%rip), %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x1d, 0xf6, 0xff, 0xff, 0xff],
            ),
            (
                "vpaddd (,%rbx,4), %zmm2, %zmm3",
                &[
                    0x62, 0xf1, 0x6d, 0x48, 0xfe, 0x1c, 0x9d, 0x00, 0x00, 0x00, 0x00,
                ],
            ),
            (
                "vpaddd 4(%rax){1to16}, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x58, 0xfe, 0x58, 0x01],
            ),
            (
                "vpaddd -4(%rax){1to16}, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x58, 0xfe, 0x58, 0xff],
            ),
            (
                "vpaddd 508(%rax){1to16}, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x58, 0xfe, 0x58, 0x7f],
            ),
            (
                "vpaddd 512(%rax){1to16}, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x58, 0xfe, 0x98, 0x00, 0x02, 0x00, 0x00],
            ),
            (
                "vaddpd 8(%rax){1to8}, %zmm2, %zmm3",
                &[0x62, 0xf1, 0xed, 0x58, 0x58, 0x58, 0x01],
            ),
            (
                "vaddpd 4(%rax){1to4}, %ymm2, %ymm3",
                &[0x62, 0xf1, 0xed, 0x38, 0x58, 0x98, 0x04, 0x00, 0x00, 0x00],
            ),
            (
                "vaddps {rn-sae}, %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6c, 0x18, 0x58, 0xd9],
            ),
            (
                "vaddps {rd-sae}, %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6c, 0x38, 0x58, 0xd9],
            ),
            (
                "vaddps {ru-sae}, %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6c, 0x58, 0x58, 0xd9],
            ),
            (
                "vaddps {rz-sae}, %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6c, 0x78, 0x58, 0xd9],
            ),
            (
                "vcmpps $1, {sae}, %zmm1, %zmm2, %k3",
                &[0x62, 0xf1, 0x6c, 0x18, 0xc2, 0xd9, 0x01],
            ),
            (
                "vaddss {rn-sae}, %xmm17, %xmm2, %xmm3",
                &[0x62, 0xb1, 0x6e, 0x18, 0x58, 0xd9],
            ),
            (
                "vcmpps $1, %zmm1, %zmm2, %k3",
                &[0x62, 0xf1, 0x6c, 0x48, 0xc2, 0xd9, 0x01],
            ),
            (
                "vcmppd $2, 64(%rax), %zmm2, %k3{%k1}",
                &[0x62, 0xf1, 0xed, 0x49, 0xc2, 0x58, 0x01, 0x02],
            ),
            (
                "vpcmpeqd %zmm1, %zmm2, %k3",
                &[0x62, 0xf1, 0x6d, 0x48, 0x76, 0xd9],
            ),
            (
                "vpcmpgtq %xmm17, %xmm2, %k3",
                &[0x62, 0xb2, 0xed, 0x08, 0x37, 0xd9],
            ),
            (
                "vmovaps %zmm1, %zmm2",
                &[0x62, 0xf1, 0x7c, 0x48, 0x28, 0xd1],
            ),
            (
                "vmovaps %zmm1, 64(%rax)",
                &[0x62, 0xf1, 0x7c, 0x48, 0x29, 0x48, 0x01],
            ),
            (
                "vmovaps 64(%rax), %zmm1",
                &[0x62, 0xf1, 0x7c, 0x48, 0x28, 0x48, 0x01],
            ),
            (
                "vmovdqa64 %zmm1, 64(%rax)",
                &[0x62, 0xf1, 0xfd, 0x48, 0x7f, 0x48, 0x01],
            ),
            (
                "vmovdqu32 64(%rax), %zmm1",
                &[0x62, 0xf1, 0x7e, 0x48, 0x6f, 0x48, 0x01],
            ),
            (
                "vmovdqa32 %zmm17, %zmm2",
                &[0x62, 0xb1, 0x7d, 0x48, 0x6f, 0xd1],
            ),
            (
                "vmovaps %zmm1, 64(%rax){%k1}",
                &[0x62, 0xf1, 0x7c, 0x49, 0x29, 0x48, 0x01],
            ),
            (
                "vextractf32x4 $1, %zmm1, %xmm2",
                &[0x62, 0xf3, 0x7d, 0x48, 0x19, 0xca, 0x01],
            ),
            (
                "vextractf32x4 $1, %zmm1, 16(%rax){%k1}",
                &[0x62, 0xf3, 0x7d, 0x49, 0x19, 0x48, 0x01, 0x01],
            ),
            (
                "vextracti64x4 $1, %zmm1, %ymm2",
                &[0x62, 0xf3, 0xfd, 0x48, 0x3b, 0xca, 0x01],
            ),
            (
                "vinsertf32x4 $1, %xmm1, %zmm2, %zmm3",
                &[0x62, 0xf3, 0x6d, 0x48, 0x18, 0xd9, 0x01],
            ),
            (
                "vinsertf64x4 $1, %ymm1, %zmm2, %zmm3",
                &[0x62, 0xf3, 0xed, 0x48, 0x1a, 0xd9, 0x01],
            ),
            (
                "vpshufd $0x1b, %zmm1, %zmm2",
                &[0x62, 0xf1, 0x7d, 0x48, 0x70, 0xd1, 0x1b],
            ),
            (
                "vpshufd $0x1b, 64(%rax){1to16}, %zmm2{%k1}{z}",
                &[0x62, 0xf1, 0x7d, 0xd9, 0x70, 0x50, 0x10, 0x1b],
            ),
            (
                "vpternlogd $0xca, %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf3, 0x6d, 0x48, 0x25, 0xd9, 0xca],
            ),
            (
                "vshufps $-1, %zmm2, %zmm5, %zmm4",
                &[0x62, 0xf1, 0x54, 0x48, 0xc6, 0xe2, 0xff],
            ),
            (
                "vpslld $3, %zmm1, %zmm2",
                &[0x62, 0xf1, 0x6d, 0x48, 0x72, 0xf1, 0x03],
            ),
            (
                "vpsllq $3, %zmm1, %zmm2",
                &[0x62, 0xf1, 0xed, 0x48, 0x73, 0xf1, 0x03],
            ),
            (
                "vpsraq $3, %zmm1, %zmm2",
                &[0x62, 0xf1, 0xed, 0x48, 0x72, 0xe1, 0x03],
            ),
            (
                "vprold $7, %xmm0, %xmm4",
                &[0x62, 0xf1, 0x5d, 0x08, 0x72, 0xc8, 0x07],
            ),
            (
                "vprorq $7, %ymm17, %ymm2{%k3}",
                &[0x62, 0xb1, 0xed, 0x2b, 0x72, 0xc1, 0x07],
            ),
            (
                "vpslld $3, 64(%rax), %zmm2",
                &[0x62, 0xf1, 0x6d, 0x48, 0x72, 0x70, 0x01, 0x03],
            ),
            (
                "vpslld $3, 4(%rax){1to16}, %zmm2",
                &[0x62, 0xf1, 0x6d, 0x58, 0x72, 0x70, 0x01, 0x03],
            ),
            (
                "vpslld %xmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xf2, 0xd9],
            ),
            (
                "vpsllvd %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf2, 0x6d, 0x48, 0x47, 0xd9],
            ),
            (
                "vcvtdq2pd %ymm1, %zmm2",
                &[0x62, 0xf1, 0x7e, 0x48, 0xe6, 0xd1],
            ),
            (
                "vcvtps2pd %ymm1, %zmm2",
                &[0x62, 0xf1, 0x7c, 0x48, 0x5a, 0xd1],
            ),
            (
                "vcvtpd2ps %zmm1, %ymm2",
                &[0x62, 0xf1, 0xfd, 0x48, 0x5a, 0xd1],
            ),
            (
                "vcvtpd2ps (%rax), %ymm4",
                &[0x62, 0xf1, 0xfd, 0x48, 0x5a, 0x20],
            ),
            (
                "vcvtpd2dq 64(%rax), %ymm4",
                &[0x62, 0xf1, 0xff, 0x48, 0xe6, 0x60, 0x01],
            ),
            (
                "vpmovzxbw %ymm1, %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x30, 0xd1],
            ),
            (
                "vpmovzxbq 8(%rax), %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x32, 0x50, 0x01],
            ),
            (
                "vpmovzxbd 16(%rax), %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x31, 0x50, 0x01],
            ),
            (
                "vpbroadcastd %eax, %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x7c, 0xd0],
            ),
            (
                "vpbroadcastd %xmm1, %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x58, 0xd1],
            ),
            (
                "vbroadcastf32x4 16(%rax), %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x1a, 0x50, 0x01],
            ),
            (
                "vfmadd213ps %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf2, 0x6d, 0x48, 0xa8, 0xd9],
            ),
            (
                "vfmadd132pd 8(%rax){1to8}, %zmm2, %zmm3",
                &[0x62, 0xf2, 0xed, 0x58, 0x98, 0x58, 0x01],
            ),
            (
                "vpermq $1, %zmm1, %zmm2",
                &[0x62, 0xf3, 0xfd, 0x48, 0x00, 0xd1, 0x01],
            ),
            (
                "vpermq %ymm0, %ymm5, %ymm4",
                &[0x62, 0xf2, 0xd5, 0x28, 0x36, 0xe0],
            ),
            (
                "vpermd %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf2, 0x6d, 0x48, 0x36, 0xd9],
            ),
            (
                "vpandd %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0x6d, 0x48, 0xdb, 0xd9],
            ),
            (
                "vpxorq %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf1, 0xed, 0x48, 0xef, 0xd9],
            ),
            (
                "vpmullq %xmm2, %xmm0, %xmm1",
                &[0x62, 0xf2, 0xfd, 0x08, 0x40, 0xca],
            ),
            (
                "vsqrtps %zmm1, %zmm2",
                &[0x62, 0xf1, 0x7c, 0x48, 0x51, 0xd1],
            ),
            (
                "vmovddup %zmm1, %zmm2",
                &[0x62, 0xf1, 0xff, 0x48, 0x12, 0xd1],
            ),
            (
                "vmovddup 8(%rax), %xmm17",
                &[0x62, 0xe1, 0xff, 0x08, 0x12, 0x48, 0x01],
            ),
            (
                "vrcp14ps %zmm1, %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x4c, 0xd1],
            ),
            (
                "vpblendmd %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf2, 0x6d, 0x48, 0x64, 0xd9],
            ),
            (
                "vscalefps %zmm1, %zmm2, %zmm3",
                &[0x62, 0xf2, 0x6d, 0x48, 0x2c, 0xd9],
            ),
        ];
        const GATHER_FORMS: &[(&str, &[u8])] = &[
            (
                "vpgatherdd %ymm3, (%rax,%ymm2,4), %ymm1",
                &[0xc4, 0xe2, 0x65, 0x90, 0x0c, 0x90],
            ),
            (
                "vpgatherdd %ymm3, 4(%rax,%ymm2,4), %ymm1",
                &[0xc4, 0xe2, 0x65, 0x90, 0x4c, 0x90, 0x04],
            ),
            (
                "vpgatherdd %xmm3, (%rax,%xmm2,4), %xmm1",
                &[0xc4, 0xe2, 0x61, 0x90, 0x0c, 0x90],
            ),
            (
                "vpgatherdq %ymm3, (%rax,%xmm2,4), %ymm1",
                &[0xc4, 0xe2, 0xe5, 0x90, 0x0c, 0x90],
            ),
            (
                "vpgatherqd %xmm3, (%rax,%ymm2,4), %xmm1",
                &[0xc4, 0xe2, 0x65, 0x91, 0x0c, 0x90],
            ),
            (
                "vpgatherqq %ymm3, 8(%r12,%ymm9,8), %ymm1",
                &[0xc4, 0x82, 0xe5, 0x91, 0x4c, 0xcc, 0x08],
            ),
            (
                "vgatherdps %ymm3, (%rax,%ymm2,4), %ymm1",
                &[0xc4, 0xe2, 0x65, 0x92, 0x0c, 0x90],
            ),
            (
                "vgatherdpd %ymm3, (%rbp,%xmm2,2), %ymm1",
                &[0xc4, 0xe2, 0xe5, 0x92, 0x4c, 0x55, 0x00],
            ),
            (
                "vgatherqps %xmm3, (%rax,%ymm2,4), %xmm1",
                &[0xc4, 0xe2, 0x65, 0x93, 0x0c, 0x90],
            ),
            (
                "vgatherqpd %ymm3, (%rax,%ymm2,4), %ymm1",
                &[0xc4, 0xe2, 0xe5, 0x93, 0x0c, 0x90],
            ),
            (
                "vpgatherdd %ymm3, (%r13,%ymm2,4), %ymm1",
                &[0xc4, 0xc2, 0x65, 0x90, 0x4c, 0x95, 0x00],
            ),
            (
                "vpgatherdd %ymm3, (%rsp,%ymm2,4), %ymm1",
                &[0xc4, 0xe2, 0x65, 0x90, 0x0c, 0x94],
            ),
            (
                "vpgatherdd %ymm3, 128(%rax,%ymm2,4), %ymm1",
                &[0xc4, 0xe2, 0x65, 0x90, 0x8c, 0x90, 0x80, 0x00, 0x00, 0x00],
            ),
            (
                "vpgatherdd %ymm3, -129(%rax,%ymm2,4), %ymm1",
                &[0xc4, 0xe2, 0x65, 0x90, 0x8c, 0x90, 0x7f, 0xff, 0xff, 0xff],
            ),
            (
                "vpgatherdd %ymm3, 127(%rax,%ymm15,4), %ymm1",
                &[0xc4, 0xa2, 0x65, 0x90, 0x4c, 0xb8, 0x7f],
            ),
            (
                "vpgatherdd %ymm11, (%r12,%ymm13,8), %ymm14",
                &[0xc4, 0x02, 0x25, 0x90, 0x34, 0xec],
            ),
            (
                "vpgatherdd %ymm3, (,%ymm2,4), %ymm1",
                &[0xc4, 0xe2, 0x65, 0x90, 0x0c, 0x95, 0x00, 0x00, 0x00, 0x00],
            ),
            (
                "vgatherdps (%rax,%ymm2,4), %ymm1{%k1}",
                &[0x62, 0xf2, 0x7d, 0x29, 0x92, 0x0c, 0x90],
            ),
            (
                "vgatherdps 64(%rax,%zmm2,4), %zmm1{%k1}",
                &[0x62, 0xf2, 0x7d, 0x49, 0x92, 0x4c, 0x90, 0x10],
            ),
            (
                "vpgatherdd 64(%rax,%zmm2,4), %zmm1{%k1}",
                &[0x62, 0xf2, 0x7d, 0x49, 0x90, 0x4c, 0x90, 0x10],
            ),
            (
                "vpgatherdd (%rax,%ymm18,4), %ymm1{%k1}",
                &[0x62, 0xf2, 0x7d, 0x21, 0x90, 0x0c, 0x90],
            ),
            (
                "vpgatherqq 64(%r13,%zmm30,8), %zmm31{%k7}",
                &[0x62, 0x02, 0xfd, 0x47, 0x91, 0x7c, 0xf5, 0x08],
            ),
        ];
        const MMX_FORMS: &[(&str, &[u8])] = &[
            ("emms", &[0x0f, 0x77]),
            ("femms", &[0x0f, 0x0e]),
            ("movq %mm1, %mm2", &[0x0f, 0x6f, 0xd1]),
            ("movq (%rax), %mm2", &[0x0f, 0x6f, 0x10]),
            ("movq %mm2, (%rax)", &[0x0f, 0x7f, 0x10]),
            ("movq %rax, %mm2", &[0x48, 0x0f, 0x6e, 0xd0]),
            ("movq %mm2, %rax", &[0x48, 0x0f, 0x7e, 0xd0]),
            ("movq %r12, %mm2", &[0x49, 0x0f, 0x6e, 0xd4]),
            ("movd %eax, %mm2", &[0x0f, 0x6e, 0xd0]),
            ("movd %mm2, %eax", &[0x0f, 0x7e, 0xd0]),
            ("movd %r8d, %mm2", &[0x41, 0x0f, 0x6e, 0xd0]),
            ("movd %rax, %mm2", &[0x48, 0x0f, 0x6e, 0xd0]),
            ("paddd %mm1, %mm2", &[0x0f, 0xfe, 0xd1]),
            ("paddd (%rax), %mm2", &[0x0f, 0xfe, 0x10]),
            ("paddd 8(%r13), %mm2", &[0x41, 0x0f, 0xfe, 0x55, 0x08]),
            (
                "paddd sym(%rip), %mm2",
                &[0x0f, 0xfe, 0x15, 0xf9, 0xff, 0xff, 0xff],
            ),
            ("paddq %mm0, %mm7", &[0x0f, 0xd4, 0xf8]),
            ("psubusw %mm3, %mm4", &[0x0f, 0xd9, 0xe3]),
            ("pmulhrsw %mm1, %mm2", &[0x0f, 0x38, 0x0b, 0xd1]),
            ("pshufb %mm1, %mm2", &[0x0f, 0x38, 0x00, 0xd1]),
            ("pabsd %mm1, %mm2", &[0x0f, 0x38, 0x1e, 0xd1]),
            ("phaddsw %mm1, %mm2", &[0x0f, 0x38, 0x03, 0xd1]),
            ("psllw $3, %mm1", &[0x0f, 0x71, 0xf1, 0x03]),
            ("pslld $3, %mm1", &[0x0f, 0x72, 0xf1, 0x03]),
            ("psllq $3, %mm1", &[0x0f, 0x73, 0xf1, 0x03]),
            ("psraw $7, %mm7", &[0x0f, 0x71, 0xe7, 0x07]),
            ("pshufw $0x1b, %mm1, %mm2", &[0x0f, 0x70, 0xd1, 0x1b]),
            ("pshufw $3, (%rax), %mm2", &[0x0f, 0x70, 0x10, 0x03]),
            ("palignr $3, %mm1, %mm2", &[0x0f, 0x3a, 0x0f, 0xd1, 0x03]),
            ("pmovmskb %mm1, %eax", &[0x0f, 0xd7, 0xc1]),
            ("pmovmskb %mm1, %rax", &[0x0f, 0xd7, 0xc1]),
            ("maskmovq %mm1, %mm2", &[0x0f, 0xf7, 0xd1]),
            ("movntq %mm1, 8(%r13)", &[0x41, 0x0f, 0xe7, 0x4d, 0x08]),
            ("pextrw $3, %mm1, %eax", &[0x0f, 0xc5, 0xc1, 0x03]),
            ("pextrw $3, %mm1, %r12d", &[0x44, 0x0f, 0xc5, 0xe1, 0x03]),
            ("pinsrw $3, %eax, %mm1", &[0x0f, 0xc4, 0xc8, 0x03]),
            ("pinsrw $3, (%rax), %mm1", &[0x0f, 0xc4, 0x08, 0x03]),
            ("cvtpi2ps %mm1, %xmm2", &[0x0f, 0x2a, 0xd1]),
            ("cvtps2pi %xmm1, %mm2", &[0x0f, 0x2d, 0xd1]),
            ("cvttps2pi %xmm1, %mm2", &[0x0f, 0x2c, 0xd1]),
            ("cvtpi2pd %mm1, %xmm2", &[0x66, 0x0f, 0x2a, 0xd1]),
            ("cvtpd2pi %xmm1, %mm2", &[0x66, 0x0f, 0x2d, 0xd1]),
            ("cvttpd2pi %xmm1, %mm2", &[0x66, 0x0f, 0x2c, 0xd1]),
            ("cvtpi2ps (%rax), %xmm2", &[0x0f, 0x2a, 0x10]),
        ];

        const SCATTER_FORMS: &[(&str, &[u8])] = &[
            (
                "vpscatterdd %zmm1, (%rax,%zmm2,4){%k1}",
                &[0x62, 0xf2, 0x7d, 0x49, 0xa0, 0x0c, 0x90],
            ),
            (
                "vpscatterdd %ymm1, 64(%rax,%ymm2,4){%k7}",
                &[0x62, 0xf2, 0x7d, 0x2f, 0xa0, 0x4c, 0x90, 0x10],
            ),
            (
                "vpscatterdd %xmm31, -4(%rbp,%xmm16,8){%k3}",
                &[0x62, 0x62, 0x7d, 0x03, 0xa0, 0x7c, 0xc5, 0xff],
            ),
            (
                "vpscatterdq %zmm1, 128(%r13,%ymm2,2){%k2}",
                &[0x62, 0xd2, 0xfd, 0x4a, 0xa0, 0x4c, 0x55, 0x10],
            ),
            (
                "vpscatterdq %ymm8, (%r12,%xmm23,1){%k4}",
                &[0x62, 0x52, 0xfd, 0x24, 0xa0, 0x04, 0x3c],
            ),
            (
                "vpscatterqd %ymm1, 512(%rsp,%zmm2,8){%k5}",
                &[
                    0x62, 0xf2, 0x7d, 0x4d, 0xa1, 0x8c, 0xd4, 0x00, 0x02, 0x00, 0x00,
                ],
            ),
            (
                "vpscatterqd %xmm1, -512(%rax,%ymm31,4){%k6}",
                &[0x62, 0xb2, 0x7d, 0x26, 0xa1, 0x4c, 0xb8, 0x80],
            ),
            (
                "vpscatterqq %zmm31, 1016(%r15,%zmm16,8){%k7}",
                &[0x62, 0x42, 0xfd, 0x47, 0xa1, 0x7c, 0xc7, 0x7f],
            ),
            (
                "vpscatterqq %xmm0, (,%xmm7,4){%k1}",
                &[
                    0x62, 0xf2, 0xfd, 0x09, 0xa1, 0x04, 0xbd, 0x00, 0x00, 0x00, 0x00,
                ],
            ),
            (
                "vscatterdps %zmm16, 4(%rax,%zmm0,1){%k1}",
                &[0x62, 0xe2, 0x7d, 0x49, 0xa2, 0x44, 0x00, 0x01],
            ),
            (
                "vscatterdps %xmm1, 508(%rbp,%xmm2,4){%k7}",
                &[0x62, 0xf2, 0x7d, 0x0f, 0xa2, 0x4c, 0x95, 0x7f],
            ),
            (
                "vscatterdpd %zmm1, -1024(%r13,%ymm30,8){%k2}",
                &[0x62, 0x92, 0xfd, 0x42, 0xa2, 0x4c, 0xf5, 0x80],
            ),
            (
                "vscatterdpd %xmm23, 8(%rsp,%xmm8,2){%k3}",
                &[0x62, 0xa2, 0xfd, 0x0b, 0xa2, 0x7c, 0x44, 0x01],
            ),
            (
                "vscatterqps %ymm1, 1020(%rax,%zmm2,4){%k1}",
                &[
                    0x62, 0xf2, 0x7d, 0x49, 0xa3, 0x8c, 0x90, 0xfc, 0x03, 0x00, 0x00,
                ],
            ),
            (
                "vscatterqpd %zmm1, 127(%r12,%zmm2,1){%k4}",
                &[
                    0x62, 0xd2, 0xfd, 0x4c, 0xa3, 0x8c, 0x14, 0x7f, 0x00, 0x00, 0x00,
                ],
            ),
            (
                "vscatterqpd %ymm7, -8(%rbp,%ymm9,8){%k5}",
                &[0x62, 0xb2, 0xfd, 0x2d, 0xa3, 0x7c, 0xcd, 0xff],
            ),
        ];
        const VSIB_PREFETCH_FORMS: &[(&str, &[u8])] = &[
            (
                "vgatherpf0dps 64(%rax,%zmm2,4){%k1}",
                &[0x62, 0xf2, 0x7d, 0x49, 0xc6, 0x4c, 0x90, 0x10],
            ),
            (
                "vgatherpf0qps (%r12,%zmm31,8){%k7}",
                &[0x62, 0x92, 0x7d, 0x47, 0xc7, 0x0c, 0xfc],
            ),
            (
                "vgatherpf1dps 508(%rbp,%zmm16,4){%k2}",
                &[0x62, 0xf2, 0x7d, 0x42, 0xc6, 0x54, 0x85, 0x7f],
            ),
            (
                "vgatherpf1qps -4(%rsp,%zmm0,1){%k3}",
                &[0x62, 0xf2, 0x7d, 0x4b, 0xc7, 0x54, 0x04, 0xff],
            ),
            (
                "vgatherpf0dpd 8(%r13,%ymm2,2){%k4}",
                &[0x62, 0xd2, 0xfd, 0x4c, 0xc6, 0x4c, 0x55, 0x01],
            ),
            (
                "vgatherpf0qpd 1016(%rax,%zmm7,8){%k5}",
                &[0x62, 0xf2, 0xfd, 0x4d, 0xc7, 0x4c, 0xf8, 0x7f],
            ),
            (
                "vgatherpf1dpd -512(%rbp,%ymm23,8){%k6}",
                &[0x62, 0xf2, 0xfd, 0x46, 0xc6, 0x54, 0xfd, 0xc0],
            ),
            (
                "vgatherpf1qpd (%rax,%zmm8,4){%k1}",
                &[0x62, 0xb2, 0xfd, 0x49, 0xc7, 0x14, 0x80],
            ),
            (
                "vscatterpf0dps 128(%rax,%zmm2,4){%k1}",
                &[0x62, 0xf2, 0x7d, 0x49, 0xc6, 0x6c, 0x90, 0x20],
            ),
            (
                "vscatterpf0qps (%rbp,%zmm2,4){%k7}",
                &[0x62, 0xf2, 0x7d, 0x4f, 0xc7, 0x6c, 0x95, 0x00],
            ),
            (
                "vscatterpf1dps -1024(%r15,%zmm31,8){%k2}",
                &[
                    0x62, 0x92, 0x7d, 0x42, 0xc6, 0xb4, 0xff, 0x00, 0xfc, 0xff, 0xff,
                ],
            ),
            (
                "vscatterpf1qps 4(%rsp,%zmm16,1){%k3}",
                &[0x62, 0xf2, 0x7d, 0x43, 0xc7, 0x74, 0x04, 0x01],
            ),
            (
                "vscatterpf0dpd (%r12,%ymm2,8){%k4}",
                &[0x62, 0xd2, 0xfd, 0x4c, 0xc6, 0x2c, 0xd4],
            ),
            (
                "vscatterpf0qpd 512(%rax,%zmm2,2){%k5}",
                &[0x62, 0xf2, 0xfd, 0x4d, 0xc7, 0x6c, 0x50, 0x40],
            ),
            (
                "vscatterpf1dpd -8(%r13,%ymm0,4){%k6}",
                &[0x62, 0xd2, 0xfd, 0x4e, 0xc6, 0x74, 0x85, 0xff],
            ),
            (
                "vscatterpf1qpd 1020(%rbp,%zmm7,1){%k7}",
                &[
                    0x62, 0xf2, 0xfd, 0x4f, 0xc7, 0xb4, 0x3d, 0xfc, 0x03, 0x00, 0x00,
                ],
            ),
        ];
        const OPMASK_FORMS: &[(&str, &[u8])] = &[
            ("kandb %k1, %k2, %k3", &[0xc5, 0xed, 0x41, 0xd9]),
            ("kandb %k7, %k0, %k5", &[0xc5, 0xfd, 0x41, 0xef]),
            ("kandw %k1, %k2, %k3", &[0xc5, 0xec, 0x41, 0xd9]),
            ("kandw %k7, %k0, %k5", &[0xc5, 0xfc, 0x41, 0xef]),
            ("kandd %k1, %k2, %k3", &[0xc4, 0xe1, 0xed, 0x41, 0xd9]),
            ("kandd %k7, %k0, %k5", &[0xc4, 0xe1, 0xfd, 0x41, 0xef]),
            ("kandq %k1, %k2, %k3", &[0xc4, 0xe1, 0xec, 0x41, 0xd9]),
            ("kandq %k7, %k0, %k5", &[0xc4, 0xe1, 0xfc, 0x41, 0xef]),
            ("kandnb %k1, %k2, %k3", &[0xc5, 0xed, 0x42, 0xd9]),
            ("kandnb %k7, %k0, %k5", &[0xc5, 0xfd, 0x42, 0xef]),
            ("kandnw %k1, %k2, %k3", &[0xc5, 0xec, 0x42, 0xd9]),
            ("kandnw %k7, %k0, %k5", &[0xc5, 0xfc, 0x42, 0xef]),
            ("kandnd %k1, %k2, %k3", &[0xc4, 0xe1, 0xed, 0x42, 0xd9]),
            ("kandnd %k7, %k0, %k5", &[0xc4, 0xe1, 0xfd, 0x42, 0xef]),
            ("kandnq %k1, %k2, %k3", &[0xc4, 0xe1, 0xec, 0x42, 0xd9]),
            ("kandnq %k7, %k0, %k5", &[0xc4, 0xe1, 0xfc, 0x42, 0xef]),
            ("korb %k1, %k2, %k3", &[0xc5, 0xed, 0x45, 0xd9]),
            ("korb %k7, %k0, %k5", &[0xc5, 0xfd, 0x45, 0xef]),
            ("korw %k1, %k2, %k3", &[0xc5, 0xec, 0x45, 0xd9]),
            ("korw %k7, %k0, %k5", &[0xc5, 0xfc, 0x45, 0xef]),
            ("kord %k1, %k2, %k3", &[0xc4, 0xe1, 0xed, 0x45, 0xd9]),
            ("kord %k7, %k0, %k5", &[0xc4, 0xe1, 0xfd, 0x45, 0xef]),
            ("korq %k1, %k2, %k3", &[0xc4, 0xe1, 0xec, 0x45, 0xd9]),
            ("korq %k7, %k0, %k5", &[0xc4, 0xe1, 0xfc, 0x45, 0xef]),
            ("kxorb %k1, %k2, %k3", &[0xc5, 0xed, 0x47, 0xd9]),
            ("kxorb %k7, %k0, %k5", &[0xc5, 0xfd, 0x47, 0xef]),
            ("kxorw %k1, %k2, %k3", &[0xc5, 0xec, 0x47, 0xd9]),
            ("kxorw %k7, %k0, %k5", &[0xc5, 0xfc, 0x47, 0xef]),
            ("kxord %k1, %k2, %k3", &[0xc4, 0xe1, 0xed, 0x47, 0xd9]),
            ("kxord %k7, %k0, %k5", &[0xc4, 0xe1, 0xfd, 0x47, 0xef]),
            ("kxorq %k1, %k2, %k3", &[0xc4, 0xe1, 0xec, 0x47, 0xd9]),
            ("kxorq %k7, %k0, %k5", &[0xc4, 0xe1, 0xfc, 0x47, 0xef]),
            ("kxnorb %k1, %k2, %k3", &[0xc5, 0xed, 0x46, 0xd9]),
            ("kxnorb %k7, %k0, %k5", &[0xc5, 0xfd, 0x46, 0xef]),
            ("kxnorw %k1, %k2, %k3", &[0xc5, 0xec, 0x46, 0xd9]),
            ("kxnorw %k7, %k0, %k5", &[0xc5, 0xfc, 0x46, 0xef]),
            ("kxnord %k1, %k2, %k3", &[0xc4, 0xe1, 0xed, 0x46, 0xd9]),
            ("kxnord %k7, %k0, %k5", &[0xc4, 0xe1, 0xfd, 0x46, 0xef]),
            ("kxnorq %k1, %k2, %k3", &[0xc4, 0xe1, 0xec, 0x46, 0xd9]),
            ("kxnorq %k7, %k0, %k5", &[0xc4, 0xe1, 0xfc, 0x46, 0xef]),
            ("kaddb %k1, %k2, %k3", &[0xc5, 0xed, 0x4a, 0xd9]),
            ("kaddb %k7, %k0, %k5", &[0xc5, 0xfd, 0x4a, 0xef]),
            ("kaddw %k1, %k2, %k3", &[0xc5, 0xec, 0x4a, 0xd9]),
            ("kaddw %k7, %k0, %k5", &[0xc5, 0xfc, 0x4a, 0xef]),
            ("kaddd %k1, %k2, %k3", &[0xc4, 0xe1, 0xed, 0x4a, 0xd9]),
            ("kaddd %k7, %k0, %k5", &[0xc4, 0xe1, 0xfd, 0x4a, 0xef]),
            ("kaddq %k1, %k2, %k3", &[0xc4, 0xe1, 0xec, 0x4a, 0xd9]),
            ("kaddq %k7, %k0, %k5", &[0xc4, 0xe1, 0xfc, 0x4a, 0xef]),
            ("knotb %k1, %k2", &[0xc5, 0xf9, 0x44, 0xd1]),
            ("knotb %k7, %k0", &[0xc5, 0xf9, 0x44, 0xc7]),
            ("knotw %k1, %k2", &[0xc5, 0xf8, 0x44, 0xd1]),
            ("knotw %k7, %k0", &[0xc5, 0xf8, 0x44, 0xc7]),
            ("knotd %k1, %k2", &[0xc4, 0xe1, 0xf9, 0x44, 0xd1]),
            ("knotd %k7, %k0", &[0xc4, 0xe1, 0xf9, 0x44, 0xc7]),
            ("knotq %k1, %k2", &[0xc4, 0xe1, 0xf8, 0x44, 0xd1]),
            ("knotq %k7, %k0", &[0xc4, 0xe1, 0xf8, 0x44, 0xc7]),
            ("kortestb %k1, %k2", &[0xc5, 0xf9, 0x98, 0xd1]),
            ("kortestb %k7, %k0", &[0xc5, 0xf9, 0x98, 0xc7]),
            ("kortestw %k1, %k2", &[0xc5, 0xf8, 0x98, 0xd1]),
            ("kortestw %k7, %k0", &[0xc5, 0xf8, 0x98, 0xc7]),
            ("kortestd %k1, %k2", &[0xc4, 0xe1, 0xf9, 0x98, 0xd1]),
            ("kortestd %k7, %k0", &[0xc4, 0xe1, 0xf9, 0x98, 0xc7]),
            ("kortestq %k1, %k2", &[0xc4, 0xe1, 0xf8, 0x98, 0xd1]),
            ("kortestq %k7, %k0", &[0xc4, 0xe1, 0xf8, 0x98, 0xc7]),
            ("ktestb %k1, %k2", &[0xc5, 0xf9, 0x99, 0xd1]),
            ("ktestb %k7, %k0", &[0xc5, 0xf9, 0x99, 0xc7]),
            ("ktestw %k1, %k2", &[0xc5, 0xf8, 0x99, 0xd1]),
            ("ktestw %k7, %k0", &[0xc5, 0xf8, 0x99, 0xc7]),
            ("ktestd %k1, %k2", &[0xc4, 0xe1, 0xf9, 0x99, 0xd1]),
            ("ktestd %k7, %k0", &[0xc4, 0xe1, 0xf9, 0x99, 0xc7]),
            ("ktestq %k1, %k2", &[0xc4, 0xe1, 0xf8, 0x99, 0xd1]),
            ("ktestq %k7, %k0", &[0xc4, 0xe1, 0xf8, 0x99, 0xc7]),
            ("kunpckbw %k1, %k2, %k3", &[0xc5, 0xed, 0x4b, 0xd9]),
            ("kunpckbw %k0, %k7, %k5", &[0xc5, 0xc5, 0x4b, 0xe8]),
            ("kunpckwd %k1, %k2, %k3", &[0xc5, 0xec, 0x4b, 0xd9]),
            ("kunpckwd %k0, %k7, %k5", &[0xc5, 0xc4, 0x4b, 0xe8]),
            ("kunpckdq %k1, %k2, %k3", &[0xc4, 0xe1, 0xec, 0x4b, 0xd9]),
            ("kunpckdq %k0, %k7, %k5", &[0xc4, 0xe1, 0xc4, 0x4b, 0xe8]),
            (
                "kshiftlb $3, %k1, %k2",
                &[0xc4, 0xe3, 0x79, 0x32, 0xd1, 0x03],
            ),
            (
                "kshiftlb $255, %k7, %k0",
                &[0xc4, 0xe3, 0x79, 0x32, 0xc7, 0xff],
            ),
            (
                "kshiftlw $3, %k1, %k2",
                &[0xc4, 0xe3, 0xf9, 0x32, 0xd1, 0x03],
            ),
            (
                "kshiftlw $255, %k7, %k0",
                &[0xc4, 0xe3, 0xf9, 0x32, 0xc7, 0xff],
            ),
            (
                "kshiftld $3, %k1, %k2",
                &[0xc4, 0xe3, 0x79, 0x33, 0xd1, 0x03],
            ),
            (
                "kshiftld $255, %k7, %k0",
                &[0xc4, 0xe3, 0x79, 0x33, 0xc7, 0xff],
            ),
            (
                "kshiftlq $3, %k1, %k2",
                &[0xc4, 0xe3, 0xf9, 0x33, 0xd1, 0x03],
            ),
            (
                "kshiftlq $255, %k7, %k0",
                &[0xc4, 0xe3, 0xf9, 0x33, 0xc7, 0xff],
            ),
            (
                "kshiftrb $3, %k1, %k2",
                &[0xc4, 0xe3, 0x79, 0x30, 0xd1, 0x03],
            ),
            (
                "kshiftrb $255, %k7, %k0",
                &[0xc4, 0xe3, 0x79, 0x30, 0xc7, 0xff],
            ),
            (
                "kshiftrw $3, %k1, %k2",
                &[0xc4, 0xe3, 0xf9, 0x30, 0xd1, 0x03],
            ),
            (
                "kshiftrw $255, %k7, %k0",
                &[0xc4, 0xe3, 0xf9, 0x30, 0xc7, 0xff],
            ),
            (
                "kshiftrd $3, %k1, %k2",
                &[0xc4, 0xe3, 0x79, 0x31, 0xd1, 0x03],
            ),
            (
                "kshiftrd $255, %k7, %k0",
                &[0xc4, 0xe3, 0x79, 0x31, 0xc7, 0xff],
            ),
            (
                "kshiftrq $3, %k1, %k2",
                &[0xc4, 0xe3, 0xf9, 0x31, 0xd1, 0x03],
            ),
            (
                "kshiftrq $255, %k7, %k0",
                &[0xc4, 0xe3, 0xf9, 0x31, 0xc7, 0xff],
            ),
            ("kmovb %k1, %k2", &[0xc5, 0xf9, 0x90, 0xd1]),
            ("kmovb %eax, %k1", &[0xc5, 0xf9, 0x92, 0xc8]),
            ("kmovb %k1, %eax", &[0xc5, 0xf9, 0x93, 0xc1]),
            ("kmovb %r15d, %k7", &[0xc4, 0xc1, 0x79, 0x92, 0xff]),
            ("kmovb %k7, %r12d", &[0xc5, 0x79, 0x93, 0xe7]),
            ("kmovb (%rax), %k1", &[0xc5, 0xf9, 0x90, 0x08]),
            ("kmovb %k1, (%rax)", &[0xc5, 0xf9, 0x91, 0x08]),
            (
                "kmovb -8(%r13,%rbx,4), %k2",
                &[0xc4, 0xc1, 0x79, 0x90, 0x54, 0x9d, 0xf8],
            ),
            (
                "kmovb %k2, 16(%r12)",
                &[0xc4, 0xc1, 0x79, 0x91, 0x54, 0x24, 0x10],
            ),
            (
                "kmovb sym(%rip), %k3",
                &[0xc5, 0xf9, 0x90, 0x1d, 0xf8, 0xff, 0xff, 0xff],
            ),
            (
                "kmovb %k3, (,%rbx,8)",
                &[0xc5, 0xf9, 0x91, 0x1c, 0xdd, 0x00, 0x00, 0x00, 0x00],
            ),
            ("kmovw %k1, %k2", &[0xc5, 0xf8, 0x90, 0xd1]),
            ("kmovw %eax, %k1", &[0xc5, 0xf8, 0x92, 0xc8]),
            ("kmovw %k1, %eax", &[0xc5, 0xf8, 0x93, 0xc1]),
            ("kmovw %r15d, %k7", &[0xc4, 0xc1, 0x78, 0x92, 0xff]),
            ("kmovw %k7, %r12d", &[0xc5, 0x78, 0x93, 0xe7]),
            ("kmovw (%rax), %k1", &[0xc5, 0xf8, 0x90, 0x08]),
            ("kmovw %k1, (%rax)", &[0xc5, 0xf8, 0x91, 0x08]),
            (
                "kmovw -8(%r13,%rbx,4), %k2",
                &[0xc4, 0xc1, 0x78, 0x90, 0x54, 0x9d, 0xf8],
            ),
            (
                "kmovw %k2, 16(%r12)",
                &[0xc4, 0xc1, 0x78, 0x91, 0x54, 0x24, 0x10],
            ),
            (
                "kmovw sym(%rip), %k3",
                &[0xc5, 0xf8, 0x90, 0x1d, 0xf8, 0xff, 0xff, 0xff],
            ),
            (
                "kmovw %k3, (,%rbx,8)",
                &[0xc5, 0xf8, 0x91, 0x1c, 0xdd, 0x00, 0x00, 0x00, 0x00],
            ),
            ("kmovd %k1, %k2", &[0xc4, 0xe1, 0xf9, 0x90, 0xd1]),
            ("kmovd %eax, %k1", &[0xc5, 0xfb, 0x92, 0xc8]),
            ("kmovd %k1, %eax", &[0xc5, 0xfb, 0x93, 0xc1]),
            ("kmovd %r15d, %k7", &[0xc4, 0xc1, 0x7b, 0x92, 0xff]),
            ("kmovd %k7, %r12d", &[0xc5, 0x7b, 0x93, 0xe7]),
            ("kmovd (%rax), %k1", &[0xc4, 0xe1, 0xf9, 0x90, 0x08]),
            ("kmovd %k1, (%rax)", &[0xc4, 0xe1, 0xf9, 0x91, 0x08]),
            (
                "kmovd -8(%r13,%rbx,4), %k2",
                &[0xc4, 0xc1, 0xf9, 0x90, 0x54, 0x9d, 0xf8],
            ),
            (
                "kmovd %k2, 16(%r12)",
                &[0xc4, 0xc1, 0xf9, 0x91, 0x54, 0x24, 0x10],
            ),
            (
                "kmovd sym(%rip), %k3",
                &[0xc4, 0xe1, 0xf9, 0x90, 0x1d, 0xf7, 0xff, 0xff, 0xff],
            ),
            (
                "kmovd %k3, (,%rbx,8)",
                &[0xc4, 0xe1, 0xf9, 0x91, 0x1c, 0xdd, 0x00, 0x00, 0x00, 0x00],
            ),
            ("kmovq %k1, %k2", &[0xc4, 0xe1, 0xf8, 0x90, 0xd1]),
            ("kmovq %rax, %k1", &[0xc4, 0xe1, 0xfb, 0x92, 0xc8]),
            ("kmovq %k1, %rax", &[0xc4, 0xe1, 0xfb, 0x93, 0xc1]),
            ("kmovq %r15, %k7", &[0xc4, 0xc1, 0xfb, 0x92, 0xff]),
            ("kmovq %k7, %r12", &[0xc4, 0x61, 0xfb, 0x93, 0xe7]),
            ("kmovq (%rax), %k1", &[0xc4, 0xe1, 0xf8, 0x90, 0x08]),
            ("kmovq %k1, (%rax)", &[0xc4, 0xe1, 0xf8, 0x91, 0x08]),
            (
                "kmovq -8(%r13,%rbx,4), %k2",
                &[0xc4, 0xc1, 0xf8, 0x90, 0x54, 0x9d, 0xf8],
            ),
            (
                "kmovq %k2, 16(%r12)",
                &[0xc4, 0xc1, 0xf8, 0x91, 0x54, 0x24, 0x10],
            ),
            (
                "kmovq sym(%rip), %k3",
                &[0xc4, 0xe1, 0xf8, 0x90, 0x1d, 0xf7, 0xff, 0xff, 0xff],
            ),
            (
                "kmovq %k3, (,%rbx,8)",
                &[0xc4, 0xe1, 0xf8, 0x91, 0x1c, 0xdd, 0x00, 0x00, 0x00, 0x00],
            ),
        ];
        const COMPRESS_FORMS: &[(&str, &[u8])] = &[
            (
                "vcompressps %zmm1, %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x8a, 0xca],
            ),
            (
                "vcompressps %zmm31, %zmm16{%k7}{z}",
                &[0x62, 0x22, 0x7d, 0xcf, 0x8a, 0xf8],
            ),
            (
                "vcompressps %ymm1, %ymm2{%k1}",
                &[0x62, 0xf2, 0x7d, 0x29, 0x8a, 0xca],
            ),
            (
                "vcompressps %xmm1, %xmm2{%k1}",
                &[0x62, 0xf2, 0x7d, 0x09, 0x8a, 0xca],
            ),
            (
                "vcompressps %zmm1, 64(%rax){%k1}",
                &[0x62, 0xf2, 0x7d, 0x49, 0x8a, 0x48, 0x10],
            ),
            (
                "vcompressps %ymm17, -64(%rbp){%k2}",
                &[0x62, 0xe2, 0x7d, 0x2a, 0x8a, 0x4d, 0xf0],
            ),
            (
                "vcompressps %xmm1, (%r12,%rbx,8){%k3}",
                &[0x62, 0xd2, 0x7d, 0x0b, 0x8a, 0x0c, 0xdc],
            ),
            (
                "vcompressps %zmm1, sym(%rip){%k4}",
                &[0x62, 0xf2, 0x7d, 0x4c, 0x8a, 0x0d, 0xf6, 0xff, 0xff, 0xff],
            ),
            (
                "vcompresspd %zmm1, %zmm2",
                &[0x62, 0xf2, 0xfd, 0x48, 0x8a, 0xca],
            ),
            (
                "vcompresspd %zmm31, %zmm16{%k7}{z}",
                &[0x62, 0x22, 0xfd, 0xcf, 0x8a, 0xf8],
            ),
            (
                "vcompresspd %ymm1, %ymm2{%k1}",
                &[0x62, 0xf2, 0xfd, 0x29, 0x8a, 0xca],
            ),
            (
                "vcompresspd %xmm1, %xmm2{%k1}",
                &[0x62, 0xf2, 0xfd, 0x09, 0x8a, 0xca],
            ),
            (
                "vcompresspd %zmm1, 64(%rax){%k1}",
                &[0x62, 0xf2, 0xfd, 0x49, 0x8a, 0x48, 0x08],
            ),
            (
                "vcompresspd %ymm17, -64(%rbp){%k2}",
                &[0x62, 0xe2, 0xfd, 0x2a, 0x8a, 0x4d, 0xf8],
            ),
            (
                "vcompresspd %xmm1, (%r12,%rbx,8){%k3}",
                &[0x62, 0xd2, 0xfd, 0x0b, 0x8a, 0x0c, 0xdc],
            ),
            (
                "vcompresspd %zmm1, sym(%rip){%k4}",
                &[0x62, 0xf2, 0xfd, 0x4c, 0x8a, 0x0d, 0xf6, 0xff, 0xff, 0xff],
            ),
            (
                "vpcompressd %zmm1, %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x8b, 0xca],
            ),
            (
                "vpcompressd %zmm31, %zmm16{%k7}{z}",
                &[0x62, 0x22, 0x7d, 0xcf, 0x8b, 0xf8],
            ),
            (
                "vpcompressd %ymm1, %ymm2{%k1}",
                &[0x62, 0xf2, 0x7d, 0x29, 0x8b, 0xca],
            ),
            (
                "vpcompressd %xmm1, %xmm2{%k1}",
                &[0x62, 0xf2, 0x7d, 0x09, 0x8b, 0xca],
            ),
            (
                "vpcompressd %zmm1, 64(%rax){%k1}",
                &[0x62, 0xf2, 0x7d, 0x49, 0x8b, 0x48, 0x10],
            ),
            (
                "vpcompressd %ymm17, -64(%rbp){%k2}",
                &[0x62, 0xe2, 0x7d, 0x2a, 0x8b, 0x4d, 0xf0],
            ),
            (
                "vpcompressd %xmm1, (%r12,%rbx,8){%k3}",
                &[0x62, 0xd2, 0x7d, 0x0b, 0x8b, 0x0c, 0xdc],
            ),
            (
                "vpcompressd %zmm1, sym(%rip){%k4}",
                &[0x62, 0xf2, 0x7d, 0x4c, 0x8b, 0x0d, 0xf6, 0xff, 0xff, 0xff],
            ),
            (
                "vpcompressq %zmm1, %zmm2",
                &[0x62, 0xf2, 0xfd, 0x48, 0x8b, 0xca],
            ),
            (
                "vpcompressq %zmm31, %zmm16{%k7}{z}",
                &[0x62, 0x22, 0xfd, 0xcf, 0x8b, 0xf8],
            ),
            (
                "vpcompressq %ymm1, %ymm2{%k1}",
                &[0x62, 0xf2, 0xfd, 0x29, 0x8b, 0xca],
            ),
            (
                "vpcompressq %xmm1, %xmm2{%k1}",
                &[0x62, 0xf2, 0xfd, 0x09, 0x8b, 0xca],
            ),
            (
                "vpcompressq %zmm1, 64(%rax){%k1}",
                &[0x62, 0xf2, 0xfd, 0x49, 0x8b, 0x48, 0x08],
            ),
            (
                "vpcompressq %ymm17, -64(%rbp){%k2}",
                &[0x62, 0xe2, 0xfd, 0x2a, 0x8b, 0x4d, 0xf8],
            ),
            (
                "vpcompressq %xmm1, (%r12,%rbx,8){%k3}",
                &[0x62, 0xd2, 0xfd, 0x0b, 0x8b, 0x0c, 0xdc],
            ),
            (
                "vpcompressq %zmm1, sym(%rip){%k4}",
                &[0x62, 0xf2, 0xfd, 0x4c, 0x8b, 0x0d, 0xf6, 0xff, 0xff, 0xff],
            ),
            (
                "vexpandps %zmm1, %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x88, 0xd1],
            ),
            (
                "vexpandps %zmm31, %zmm16{%k7}{z}",
                &[0x62, 0x82, 0x7d, 0xcf, 0x88, 0xc7],
            ),
            (
                "vexpandps 64(%rax), %zmm2{%k1}",
                &[0x62, 0xf2, 0x7d, 0x49, 0x88, 0x50, 0x10],
            ),
            (
                "vexpandps -64(%rbp), %ymm17{%k2}",
                &[0x62, 0xe2, 0x7d, 0x2a, 0x88, 0x4d, 0xf0],
            ),
            (
                "vexpandps (%r13,%rbx,2), %xmm1{%k3}",
                &[0x62, 0xd2, 0x7d, 0x0b, 0x88, 0x4c, 0x5d, 0x00],
            ),
            (
                "vexpandps sym(%rip), %zmm1{%k4}",
                &[0x62, 0xf2, 0x7d, 0x4c, 0x88, 0x0d, 0xf6, 0xff, 0xff, 0xff],
            ),
            (
                "vexpandpd %zmm1, %zmm2",
                &[0x62, 0xf2, 0xfd, 0x48, 0x88, 0xd1],
            ),
            (
                "vexpandpd %zmm31, %zmm16{%k7}{z}",
                &[0x62, 0x82, 0xfd, 0xcf, 0x88, 0xc7],
            ),
            (
                "vexpandpd 64(%rax), %zmm2{%k1}",
                &[0x62, 0xf2, 0xfd, 0x49, 0x88, 0x50, 0x08],
            ),
            (
                "vexpandpd -64(%rbp), %ymm17{%k2}",
                &[0x62, 0xe2, 0xfd, 0x2a, 0x88, 0x4d, 0xf8],
            ),
            (
                "vexpandpd (%r13,%rbx,2), %xmm1{%k3}",
                &[0x62, 0xd2, 0xfd, 0x0b, 0x88, 0x4c, 0x5d, 0x00],
            ),
            (
                "vexpandpd sym(%rip), %zmm1{%k4}",
                &[0x62, 0xf2, 0xfd, 0x4c, 0x88, 0x0d, 0xf6, 0xff, 0xff, 0xff],
            ),
            (
                "vpexpandd %zmm1, %zmm2",
                &[0x62, 0xf2, 0x7d, 0x48, 0x89, 0xd1],
            ),
            (
                "vpexpandd %zmm31, %zmm16{%k7}{z}",
                &[0x62, 0x82, 0x7d, 0xcf, 0x89, 0xc7],
            ),
            (
                "vpexpandd 64(%rax), %zmm2{%k1}",
                &[0x62, 0xf2, 0x7d, 0x49, 0x89, 0x50, 0x10],
            ),
            (
                "vpexpandd -64(%rbp), %ymm17{%k2}",
                &[0x62, 0xe2, 0x7d, 0x2a, 0x89, 0x4d, 0xf0],
            ),
            (
                "vpexpandd (%r13,%rbx,2), %xmm1{%k3}",
                &[0x62, 0xd2, 0x7d, 0x0b, 0x89, 0x4c, 0x5d, 0x00],
            ),
            (
                "vpexpandd sym(%rip), %zmm1{%k4}",
                &[0x62, 0xf2, 0x7d, 0x4c, 0x89, 0x0d, 0xf6, 0xff, 0xff, 0xff],
            ),
            (
                "vpexpandq %zmm1, %zmm2",
                &[0x62, 0xf2, 0xfd, 0x48, 0x89, 0xd1],
            ),
            (
                "vpexpandq %zmm31, %zmm16{%k7}{z}",
                &[0x62, 0x82, 0xfd, 0xcf, 0x89, 0xc7],
            ),
            (
                "vpexpandq 64(%rax), %zmm2{%k1}",
                &[0x62, 0xf2, 0xfd, 0x49, 0x89, 0x50, 0x08],
            ),
            (
                "vpexpandq -64(%rbp), %ymm17{%k2}",
                &[0x62, 0xe2, 0xfd, 0x2a, 0x89, 0x4d, 0xf8],
            ),
            (
                "vpexpandq (%r13,%rbx,2), %xmm1{%k3}",
                &[0x62, 0xd2, 0xfd, 0x0b, 0x89, 0x4c, 0x5d, 0x00],
            ),
            (
                "vpexpandq sym(%rip), %zmm1{%k4}",
                &[0x62, 0xf2, 0xfd, 0x4c, 0x89, 0x0d, 0xf6, 0xff, 0xff, 0xff],
            ),
        ];
        const EVEX_SCALAR_FORMS: &[(&str, &[u8])] = &[
            (
                "vmovss %xmm1, %xmm2, %xmm3{%k1}",
                &[0x62, 0xf1, 0x6e, 0x09, 0x10, 0xd9],
            ),
            (
                "vmovss %xmm31, %xmm30, %xmm29{%k7}{z}",
                &[0x62, 0x01, 0x0e, 0x87, 0x10, 0xef],
            ),
            (
                "vmovss %xmm17, %xmm2, %xmm3",
                &[0x62, 0xb1, 0x6e, 0x08, 0x10, 0xd9],
            ),
            (
                "vmovss 4(%rax), %xmm3{%k1}",
                &[0x62, 0xf1, 0x7e, 0x09, 0x10, 0x58, 0x01],
            ),
            (
                "vmovss 512(%rbp), %xmm17{%k2}{z}",
                &[0x62, 0xe1, 0x7e, 0x8a, 0x10, 0x8d, 0x00, 0x02, 0x00, 0x00],
            ),
            (
                "vmovss %xmm3, 4(%r12){%k1}",
                &[0x62, 0xd1, 0x7e, 0x09, 0x11, 0x5c, 0x24, 0x01],
            ),
            (
                "vmovss %xmm31, -4(%r13,%rbx,8){%k7}",
                &[0x62, 0x41, 0x7e, 0x0f, 0x11, 0x7c, 0xdd, 0xff],
            ),
            (
                "vmovss sym(%rip), %xmm3{%k1}",
                &[0x62, 0xf1, 0x7e, 0x09, 0x10, 0x1d, 0xf6, 0xff, 0xff, 0xff],
            ),
            (
                "vmovsd %xmm1, %xmm2, %xmm3{%k1}",
                &[0x62, 0xf1, 0xef, 0x09, 0x10, 0xd9],
            ),
            (
                "vmovsd %xmm31, %xmm30, %xmm29{%k7}{z}",
                &[0x62, 0x01, 0x8f, 0x87, 0x10, 0xef],
            ),
            (
                "vmovsd 8(%rax), %xmm3{%k1}",
                &[0x62, 0xf1, 0xff, 0x09, 0x10, 0x58, 0x01],
            ),
            (
                "vmovsd 1016(%rsp), %xmm31{%k3}",
                &[0x62, 0x61, 0xff, 0x0b, 0x10, 0x7c, 0x24, 0x7f],
            ),
            (
                "vmovsd %xmm3, 8(%rbp){%k1}",
                &[0x62, 0xf1, 0xff, 0x09, 0x11, 0x5d, 0x01],
            ),
            (
                "vmovsd %xmm17, (,%rbx,4){%k5}",
                &[
                    0x62, 0xe1, 0xff, 0x0d, 0x11, 0x0c, 0x9d, 0x00, 0x00, 0x00, 0x00,
                ],
            ),
            (
                "vmovntdqa 64(%rax), %zmm1",
                &[0x62, 0xf2, 0x7d, 0x48, 0x2a, 0x48, 0x01],
            ),
            (
                "vmovntdqa -64(%rbp), %ymm17",
                &[0x62, 0xe2, 0x7d, 0x28, 0x2a, 0x4d, 0xfe],
            ),
            (
                "vmovntdqa (%r13,%rbx,8), %zmm31",
                &[0x62, 0x42, 0x7d, 0x48, 0x2a, 0x7c, 0xdd, 0x00],
            ),
            (
                "vmovntdqa sym(%rip), %zmm1",
                &[0x62, 0xf2, 0x7d, 0x48, 0x2a, 0x0d, 0xf6, 0xff, 0xff, 0xff],
            ),
        ];
        const SIZED_CONVERT_FORMS: &[(&str, &[u8])] = &[
            (
                "vcvtpd2psy 64(%rax), %xmm17{%k1}",
                &[0x62, 0xe1, 0xfd, 0x29, 0x5a, 0x48, 0x02],
            ),
            (
                "vcvtpd2psx 64(%rax), %xmm17{%k1}",
                &[0x62, 0xe1, 0xfd, 0x09, 0x5a, 0x48, 0x04],
            ),
            (
                "vcvttpd2dqy -32(%rbp), %xmm3{%k2}{z}",
                &[0x62, 0xf1, 0xfd, 0xaa, 0xe6, 0x5d, 0xff],
            ),
            (
                "vcvttpd2dqx -32(%rbp), %xmm3{%k2}{z}",
                &[0x62, 0xf1, 0xfd, 0x8a, 0xe6, 0x5d, 0xfe],
            ),
            (
                "vcvtpd2dqx 16(%r13), %xmm20{%k5}",
                &[0x62, 0xc1, 0xff, 0x0d, 0xe6, 0x65, 0x01],
            ),
            (
                "vcvtpd2dqy 16(%r13), %xmm20{%k5}",
                &[0x62, 0xc1, 0xff, 0x2d, 0xe6, 0xa5, 0x10, 0x00, 0x00, 0x00],
            ),
            (
                "vcvtpd2psy sym(%rip), %xmm18{%k3}",
                &[0x62, 0xe1, 0xfd, 0x2b, 0x5a, 0x15, 0xf6, 0xff, 0xff, 0xff],
            ),
            ("vcvtpd2psx %xmm2, %xmm1", &[0xc5, 0xf9, 0x5a, 0xca]),
            ("vcvtpd2psy %ymm2, %xmm1", &[0xc5, 0xfd, 0x5a, 0xca]),
            ("vcvtpd2psx (%rax), %xmm1", &[0xc5, 0xf9, 0x5a, 0x08]),
            (
                "vcvtpd2psy 64(%rbp), %xmm1",
                &[0xc5, 0xfd, 0x5a, 0x4d, 0x40],
            ),
            (
                "vcvtpd2psx -16(%r13,%rbx,4), %xmm1",
                &[0xc4, 0xc1, 0x79, 0x5a, 0x4c, 0x9d, 0xf0],
            ),
            (
                "vcvtpd2psy sym(%rip), %xmm1",
                &[0xc5, 0xfd, 0x5a, 0x0d, 0xf8, 0xff, 0xff, 0xff],
            ),
            (
                "vcvtpd2psx %xmm18, %xmm1",
                &[0x62, 0xb1, 0xfd, 0x08, 0x5a, 0xca],
            ),
            (
                "vcvtpd2psy %ymm2, %xmm17{%k1}{z}",
                &[0x62, 0xe1, 0xfd, 0xa9, 0x5a, 0xca],
            ),
            (
                "vcvtpd2psx (%rax){1to2}, %xmm1",
                &[0x62, 0xf1, 0xfd, 0x18, 0x5a, 0x08],
            ),
            (
                "vcvtpd2psy (%rax){1to4}, %xmm17{%k3}",
                &[0x62, 0xe1, 0xfd, 0x3b, 0x5a, 0x08],
            ),
            ("vcvtpd2dqx %xmm2, %xmm1", &[0xc5, 0xfb, 0xe6, 0xca]),
            ("vcvtpd2dqy %ymm2, %xmm1", &[0xc5, 0xff, 0xe6, 0xca]),
            ("vcvtpd2dqx (%rax), %xmm1", &[0xc5, 0xfb, 0xe6, 0x08]),
            (
                "vcvtpd2dqy 64(%rbp), %xmm1",
                &[0xc5, 0xff, 0xe6, 0x4d, 0x40],
            ),
            (
                "vcvtpd2dqx -16(%r13,%rbx,4), %xmm1",
                &[0xc4, 0xc1, 0x7b, 0xe6, 0x4c, 0x9d, 0xf0],
            ),
            (
                "vcvtpd2dqy sym(%rip), %xmm1",
                &[0xc5, 0xff, 0xe6, 0x0d, 0xf8, 0xff, 0xff, 0xff],
            ),
            (
                "vcvtpd2dqx %xmm18, %xmm1",
                &[0x62, 0xb1, 0xff, 0x08, 0xe6, 0xca],
            ),
            (
                "vcvtpd2dqy %ymm2, %xmm17{%k1}{z}",
                &[0x62, 0xe1, 0xff, 0xa9, 0xe6, 0xca],
            ),
            (
                "vcvtpd2dqx (%rax){1to2}, %xmm1",
                &[0x62, 0xf1, 0xff, 0x18, 0xe6, 0x08],
            ),
            (
                "vcvtpd2dqy (%rax){1to4}, %xmm17{%k3}",
                &[0x62, 0xe1, 0xff, 0x3b, 0xe6, 0x08],
            ),
            ("vcvttpd2dqx %xmm2, %xmm1", &[0xc5, 0xf9, 0xe6, 0xca]),
            ("vcvttpd2dqy %ymm2, %xmm1", &[0xc5, 0xfd, 0xe6, 0xca]),
            ("vcvttpd2dqx (%rax), %xmm1", &[0xc5, 0xf9, 0xe6, 0x08]),
            (
                "vcvttpd2dqy 64(%rbp), %xmm1",
                &[0xc5, 0xfd, 0xe6, 0x4d, 0x40],
            ),
            (
                "vcvttpd2dqx -16(%r13,%rbx,4), %xmm1",
                &[0xc4, 0xc1, 0x79, 0xe6, 0x4c, 0x9d, 0xf0],
            ),
            (
                "vcvttpd2dqy sym(%rip), %xmm1",
                &[0xc5, 0xfd, 0xe6, 0x0d, 0xf8, 0xff, 0xff, 0xff],
            ),
            (
                "vcvttpd2dqx %xmm18, %xmm1",
                &[0x62, 0xb1, 0xfd, 0x08, 0xe6, 0xca],
            ),
            (
                "vcvttpd2dqy %ymm2, %xmm17{%k1}{z}",
                &[0x62, 0xe1, 0xfd, 0xa9, 0xe6, 0xca],
            ),
            (
                "vcvttpd2dqx (%rax){1to2}, %xmm1",
                &[0x62, 0xf1, 0xfd, 0x18, 0xe6, 0x08],
            ),
            (
                "vcvttpd2dqy (%rax){1to4}, %xmm17{%k3}",
                &[0x62, 0xe1, 0xfd, 0x3b, 0xe6, 0x08],
            ),
        ];

        fn pinned(name: &str, forms: &[(&str, &[u8])]) {
            let tmp = TempDir::new(name);
            for (index, (asm, expected)) in forms.iter().enumerate() {
                let path = tmp.path().join(format!("{name}{index}.o"));
                let object = Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                    .unwrap()
                    .assemble(&format!(".text\nsym:\n{asm}\n"), &path)
                    .map(|_| std::fs::read(&path).unwrap())
                    .unwrap_or_else(|error| panic!("{asm}: {error:?}"));
                assert_eq!(text(&object), *expected, "{asm}");
            }
        }

        #[test]
        fn avx512_forms_match_gas() {
            pinned("evex", EVEX_FORMS);
        }

        #[test]
        fn vsib_gathers_match_gas() {
            pinned("gather", GATHER_FORMS);
        }

        #[test]
        fn mmx_forms_match_gas() {
            pinned("mmx", MMX_FORMS);
        }

        #[test]
        fn avx512_rejects_what_gas_rejects() {
            let tmp = TempDir::new("evex_invalid");
            for asm in [
                "vpaddd %zmm1, %zmm2, %zmm3{%k0}",
                "vpaddd %zmm1, %zmm2, %zmm3{z}",
                "vaddps %zmm1, %zmm2, %zmm3{%k1}{z}{z}",
                "vaddps {rn-sae}, %ymm1, %ymm2, %ymm3",
                "vpaddd 4(%rax){1to8}, %zmm2, %zmm3",
                "vpaddb 4(%rax){1to64}, %zmm2, %zmm3",
                "vpaddd {rn-sae}, %zmm1, %zmm2, %zmm3",
                "vmovaps %zmm1, 64(%rax){%k1}{z}",
                "vpaddd %zmm1, %zmm2, %k3",
                "vpcmpeqd %zmm1, %zmm2, %zmm3",
                "vpaddd %zmm1, %ymm2, %zmm3",
                "vpaddd %zmm1, %zmm2, %zmm32",
                "vpmovzxbq %ymm1, %zmm2",
                "vpslld $3, %zmm1, %zmm2, %zmm3",
                "vpsllw 4(%rax){1to32}, %zmm2, %zmm3",
                "vcvtpd2ps (%rax), %xmm4",
                "vpgatherdd %ymm3, (%rax,%rbx,4), %ymm1",
                "vpgatherdd %ymm3, (%rax), %ymm1",
                "vpgatherdd %ymm3, (%rax,%ymm2,4), %xmm1",
                "vpgatherdd %xmm3, (%rax,%ymm2,4), %ymm1",
                "vpgatherdd (%rax,%ymm2,4), %ymm1",
                "vpgatherdd (%rax,%ymm2,4), %ymm1{%k1}{z}",
                "vpgatherdd %ymm3, (%rax,%ymm18,4), %ymm1",
                "vgatherdps %ymm3, (%rax,%ymm2,4){1to8}, %ymm1",
                "pshufw $3, %mm1, %xmm2",
                "paddd %mm1, %xmm2",
                "paddd %xmm1, %mm2",
                "pslldq $3, %mm1",
                "movntq %mm1, %mm2",
                "pmovmskb %mm1, %ax",
                "emms %mm0",
                "pcmpeqq %mm1, %mm2",
                "pmaxsd %mm1, %mm2",
            ] {
                let path = tmp.path().join("invalid.o");
                assert!(
                    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                        .unwrap()
                        .assemble(&format!(".text\n{asm}\n"), &path)
                        .is_err(),
                    "{asm}"
                );
            }
        }

        #[test]
        fn vsib_scatters_match_gas() {
            pinned("scatter", SCATTER_FORMS);
        }

        #[test]
        fn vsib_prefetches_match_gas() {
            pinned("vsibpf", VSIB_PREFETCH_FORMS);
        }

        #[test]
        fn opmask_forms_match_gas() {
            pinned("opmask", OPMASK_FORMS);
        }

        #[test]
        fn compress_and_expand_match_gas() {
            pinned("compress", COMPRESS_FORMS);
        }

        #[test]
        fn evex_scalar_moves_match_gas() {
            pinned("evexscalar", EVEX_SCALAR_FORMS);
        }

        #[test]
        fn sized_converts_match_gas() {
            pinned("sizedcvt", SIZED_CONVERT_FORMS);
        }

        #[test]
        fn scatters_and_opmasks_reject_what_gas_rejects() {
            let tmp = TempDir::new("evex512_invalid");
            for asm in [
                "vpscatterdd %zmm1, 64(%rax,%zmm2,4)",
                "vpscatterdd %zmm1, 64(%rax,%zmm2,4){%k0}",
                "vpscatterdd %zmm1, 64(%rax,%zmm2,4){%k1}{z}",
                "vpscatterdd %zmm1, 64(%rax,%rbx,4){%k1}",
                "vpscatterdd %zmm1, 64(%rax){%k1}",
                "vpscatterdd %zmm1{%k1}, 64(%rax,%zmm2,4)",
                "vpscatterdd 64(%rax,%zmm2,4){%k1}, %zmm1",
                "vpscatterdd %zmm1, 64(%rax,%zmm2,4){1to16}{%k1}",
                "vpscatterdd %zmm1, sym(%rip){%k1}",
                "vpscatterdq %zmm1, (%rax,%zmm2,4){%k1}",
                "vpscatterqd %zmm1, (%rax,%zmm2,4){%k1}",
                "vscatterdpd %zmm1, (%rax,%zmm2,4){%k1}",
                "vgatherpf0dps 64(%rax,%zmm2,4)",
                "vgatherpf0dps 64(%rax,%zmm2,4){%k0}",
                "vgatherpf0dps 64(%rax,%zmm2,4){%k1}{z}",
                "vgatherpf0dps 64(%rax,%rbx,4){%k1}",
                "vgatherpf0dpd 64(%rax,%zmm2,4){%k1}",
                "vscatterpf1qps %zmm1, 64(%rax,%zmm2,4){%k1}",
                "kmovw %k1, %k2, %k3",
                "kmovw %k1, %ax",
                "kmovw %k1, %rax",
                "kmovq %k1, %eax",
                "kmovd %k1, %rax",
                "kmovw %eax, %ebx",
                "kmovw %xmm1, %k1",
                "kandw %k1, %k2",
                "kandw %k1, %k2, %eax",
                "knotw %k1, %k2, %k3",
                "knotw (%rax), %k2",
                "kshiftlw $-1, %k1, %k2",
                "kshiftlw $256, %k1, %k2",
                "kshiftlw %k1, %k2, %k3",
                "kshiftlw $3, (%rax), %k2",
                "vcompressps 64(%rax), %zmm1{%k1}",
                "vcompressps %zmm1, 64(%rax){%k1}{z}",
                "vcompressps %zmm1, %zmm2{%k0}",
                "vcompressps %zmm1, %zmm2{z}",
                "vcompressps {rn-sae}, %zmm1, %zmm2",
                "vcompressps (%rax){1to16}, %zmm1{%k1}",
                "vpexpandd %zmm1, 64(%rax){%k1}",
                "vexpandps %zmm1, %zmm2, %zmm3",
                "vmovntdqa %zmm1, 64(%rax)",
                "vmovntdqa 64(%rax), %zmm1{%k1}",
                "vmovntdqa %zmm1, %zmm2",
                "vmovss %xmm1, %xmm2",
                "vmovss %xmm1, %xmm2{%k1}",
                "vmovss %xmm1, %xmm2, %xmm3{z}",
                "vmovss %xmm1, %ymm2, %xmm3{%k1}",
                "vmovss 4(%rax), %xmm3, %xmm4{%k1}",
                "vmovss %xmm3, 4(%rax){%k1}{z}",
                "vmovsd {rn-sae}, %xmm1, %xmm2, %xmm3",
                "vcvtpd2psx %ymm2, %xmm1",
                "vcvtpd2psy %xmm2, %xmm1",
                "vcvtpd2psx (%rax), %ymm1",
                "vcvtpd2psy %zmm2, %xmm1",
                "vcvtsi2ssq {rn-sae}, %rax, %xmm2, %xmm3",
                "vcvtsi2sdl {rn-sae}, %eax, %xmm2, %xmm3",
            ] {
                let path = tmp.path().join("invalid.o");
                assert!(
                    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                        .unwrap()
                        .assemble(&format!(".text\n{asm}\n"), &path)
                        .is_err(),
                    "{asm}"
                );
            }
        }
    }
}
