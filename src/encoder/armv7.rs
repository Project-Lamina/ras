//! ARMv7-A (A32) binary instruction encoder.
//!
//! Every A32 instruction is 32 bits wide and stored little-endian, and the
//! condition occupies bits 31:28 of all of them, so a conditional mnemonic is
//! the base mnemonic with a condition field rather than an opcode of its own.

use crate::encoder::traits::{InstructionEncoder, ParsedInstruction};
use crate::error::RasError;

pub const COND_AL: u32 = 14;

pub struct Armv7Encoder {
    position: usize,
}

impl Default for Armv7Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Condition field for a mnemonic suffix; the empty suffix is unconditional.
pub fn condition(name: &str) -> Option<u32> {
    Some(match name {
        "eq" => 0,
        "ne" => 1,
        "cs" | "hs" => 2,
        "cc" | "lo" => 3,
        "mi" => 4,
        "pl" => 5,
        "vs" => 6,
        "vc" => 7,
        "hi" => 8,
        "ls" => 9,
        "ge" => 10,
        "lt" => 11,
        "gt" => 12,
        "le" => 13,
        "al" | "" => COND_AL,
        _ => return None,
    })
}

/// Split a mnemonic stem into the condition that follows `base`, and whether
/// the S bit is set. `movls` is `mov` with condition `ls`, not `mov` with the
/// S bit, so the condition is matched before the S suffix.
fn suffixes(stem: &str, base: &str) -> Option<(bool, u32)> {
    let rest = stem.strip_prefix(base)?;
    if let Some(cond) = condition(rest) {
        return Some((false, cond));
    }
    condition(rest.strip_prefix('s')?).map(|cond| (true, cond))
}

/// Register number for an A32 core register, by number or ABI name.
pub fn gpr(text: &str) -> Result<u32, RasError> {
    let text = text.trim();
    match text {
        "sp" => Ok(13),
        "lr" => Ok(14),
        "pc" => Ok(15),
        "fp" => Ok(11),
        "ip" => Ok(12),
        "sl" => Ok(10),
        _ => text
            .strip_prefix('r')
            .and_then(|n| n.parse::<u32>().ok())
            .filter(|n| *n < 16)
            .ok_or_else(|| RasError::EncodingError(format!("Unknown ARM register: {text}"))),
    }
}

fn is_gpr(text: &str) -> bool {
    gpr(text).is_ok()
}

/// Register number of `s0`-`s31` or `d0`-`d31`.
fn vfp(text: &str, double: bool) -> Result<u32, RasError> {
    let text = text.trim();
    let prefix = if double { 'd' } else { 's' };
    text.strip_prefix(prefix)
        .and_then(|n| n.parse::<u32>().ok())
        .filter(|n| *n < 32)
        .ok_or_else(|| RasError::EncodingError(format!("Unknown ARM {prefix}-register: {text}")))
}

fn is_vfp(text: &str, double: bool) -> bool {
    vfp(text, double).is_ok()
}

/// Split a VFP register into the high bit and the low four bits. A single
/// register numbers the low bits first, a double register the high bit first.
fn vfp_split(number: u32, double: bool) -> (u32, u32) {
    if double {
        ((number >> 4) & 1, number & 0xf)
    } else {
        (number & 1, number >> 1)
    }
}

fn immediate(text: &str) -> Result<i64, RasError> {
    let text = text.trim().strip_prefix('#').unwrap_or(text.trim());
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let value = match digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        Some(hex) => i64::from_str_radix(hex, 16),
        None => digits.parse::<i64>(),
    }
    .map_err(|_| RasError::EncodingError(format!("Invalid ARM immediate: {text}")))?;
    Ok(if negative { -value } else { value })
}

/// The rotate/value pair of an A32 modified immediate, or `None` when the
/// value is not an 8-bit constant rotated right by an even amount.
pub fn modified_immediate(value: i64) -> Option<u32> {
    if !(-(1i64 << 31)..(1i64 << 32)).contains(&value) {
        return None;
    }
    let value = value as u32;
    (0..16u32).find_map(|rotate| {
        let candidate = value.rotate_left(2 * rotate);
        (candidate < 256).then_some((rotate << 8) | candidate)
    })
}

fn shift_type(name: &str) -> Option<u32> {
    match name {
        "lsl" => Some(0),
        "lsr" => Some(1),
        "asr" => Some(2),
        "ror" => Some(3),
        _ => None,
    }
}

fn shift_amount(kind: u32, amount: i64) -> Result<u32, RasError> {
    let valid = match kind {
        0 => (0..=31).contains(&amount),
        1 | 2 => (1..=32).contains(&amount),
        _ => (1..=31).contains(&amount),
    };
    if !valid {
        return Err(RasError::EncodingError(format!(
            "ARM shift amount {amount} out of range"
        )));
    }
    Ok(amount as u32 & 31)
}

/// Rejoin operands the parser split at every comma, then split them again at
/// commas outside brackets so that `[r11, #-8]` and `{r4, r5}` stay whole.
fn regroup(operands: &[String]) -> Vec<String> {
    let text = operands.join(", ");
    if text.trim().is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (index, ch) in text.char_indices() {
        match ch {
            '[' | '{' | '(' => depth += 1,
            ']' | '}' | ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(text[start..index].trim().to_string());
                start = index + 1;
            }
            _ => {}
        }
    }
    out.push(text[start..].trim().to_string());
    out
}

struct Memory {
    base: u32,
    offset: i64,
}

fn memory(text: &str) -> Result<Memory, RasError> {
    let text = text.trim();
    if text.ends_with('!') {
        return Err(RasError::EncodingError(format!(
            "ARM write-back addressing is not encoded: {text}"
        )));
    }
    let inner = text
        .trim()
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or_else(|| {
            RasError::EncodingError(format!("ARM addressing mode is not supported: {text}"))
        })?;
    let (base, offset) = match inner.split_once(',') {
        Some((base, offset)) => (base, Some(offset.trim())),
        None => (inner, None),
    };
    let base = gpr(base)?;
    let offset = match offset {
        None => 0,
        Some(offset) if offset.starts_with('#') => immediate(offset)?,
        Some(offset) => {
            return Err(RasError::EncodingError(format!(
                "ARM register-offset addressing is not encoded: [{inner}] uses {offset}"
            )));
        }
    };
    Ok(Memory { base, offset })
}

/// The registers of a `{...}` list, in the order written.
fn register_list(text: &str) -> Result<Vec<String>, RasError> {
    let inner = text
        .trim()
        .strip_prefix('{')
        .and_then(|rest| rest.strip_suffix('}'))
        .ok_or_else(|| RasError::EncodingError(format!("Expected a register list: {text}")))?;
    Ok(inner.split(',').map(|r| r.trim().to_string()).collect())
}

/// The first register and the count of a list that must be consecutive.
fn consecutive(list: &[String], double: bool) -> Result<(u32, u32), RasError> {
    let mut numbers = Vec::with_capacity(list.len());
    for name in list {
        numbers.push(vfp(name, double)?);
    }
    let first = *numbers
        .first()
        .ok_or_else(|| RasError::EncodingError("Empty VFP register list".into()))?;
    if numbers
        .iter()
        .enumerate()
        .any(|(index, number)| *number != first + index as u32)
    {
        return Err(RasError::EncodingError(
            "VFP register list is not consecutive".into(),
        ));
    }
    Ok((first, numbers.len() as u32))
}

fn reject_post_index(opcode: &str, operands: &[String]) -> Result<(), RasError> {
    if operands.len() > 2 && operands[1].ends_with(']') {
        return Err(RasError::EncodingError(format!(
            "ARM post-indexed addressing is not encoded: {opcode} {}",
            operands.join(", ")
        )));
    }
    Ok(())
}

fn operand_count(opcode: &str, operands: &[String], wanted: usize) -> Result<(), RasError> {
    if operands.len() != wanted {
        return Err(RasError::EncodingError(format!(
            "{opcode} takes {wanted} operands, got {}",
            operands.len()
        )));
    }
    Ok(())
}

/// Data-processing opcode field for a base mnemonic.
fn data_processing(base: &str) -> Option<u32> {
    Some(match base {
        "and" => 0,
        "eor" => 1,
        "sub" => 2,
        "rsb" => 3,
        "add" => 4,
        "adc" => 5,
        "sbc" => 6,
        "rsc" => 7,
        "tst" => 8,
        "teq" => 9,
        "cmp" => 10,
        "cmn" => 11,
        "orr" => 12,
        "mov" => 13,
        "bic" => 14,
        "mvn" => 15,
        _ => return None,
    })
}

/// The I bit and the twelve-bit operand2 field of a data-processing shifter
/// operand, which is either an immediate, a register, or a shifted register.
fn shifter_operand(operands: &[String]) -> Result<(u32, u32), RasError> {
    let first = operands[0].as_str();
    if first.starts_with('#') {
        if operands.len() != 1 {
            return Err(RasError::EncodingError(
                "An immediate shifter operand takes no shift".into(),
            ));
        }
        let value = immediate(first)?;
        let field = modified_immediate(value).ok_or_else(|| {
            RasError::EncodingError(format!(
                "ARM immediate {value} is not an 8-bit value rotated by an even amount"
            ))
        })?;
        return Ok((1, field));
    }
    let rm = gpr(first)?;
    match operands.len() {
        1 => Ok((0, rm)),
        2 => {
            let (name, amount) = operands[1].split_once(char::is_whitespace).ok_or_else(|| {
                RasError::EncodingError(format!("Invalid ARM shift: {}", operands[1]))
            })?;
            let kind = shift_type(name.trim()).ok_or_else(|| {
                RasError::EncodingError(format!("Unsupported ARM shift: {}", name.trim()))
            })?;
            let amount = amount.trim();
            if let Some(amount) = amount.strip_prefix('#') {
                let amount = shift_amount(kind, immediate(amount)?)?;
                Ok((0, (amount << 7) | (kind << 5) | rm))
            } else {
                let rs = gpr(amount)?;
                Ok((0, (rs << 8) | (kind << 5) | (1 << 4) | rm))
            }
        }
        _ => Err(RasError::EncodingError(
            "Too many operands for an ARM shifter operand".into(),
        )),
    }
}

/// A `vcvt` type suffix: the size in bits and whether it is floating point,
/// signed, or unsigned.
#[derive(PartialEq, Eq, Clone, Copy)]
enum CvtType {
    F32,
    F64,
    S32,
    U32,
}

fn cvt_type(name: &str) -> Option<CvtType> {
    Some(match name {
        "f32" => CvtType::F32,
        "f64" => CvtType::F64,
        "s32" => CvtType::S32,
        "u32" => CvtType::U32,
        _ => return None,
    })
}

impl Armv7Encoder {
    pub fn new() -> Self {
        Self { position: 0 }
    }

    fn encode(&self, inst: &ParsedInstruction) -> Result<u32, RasError> {
        let opcode = inst.opcode.to_ascii_lowercase();
        let operands = regroup(&inst.operands);
        let (stem, suffix) = match opcode.split_once('.') {
            Some((stem, suffix)) => (stem, suffix),
            None => (opcode.as_str(), ""),
        };
        if let Some(word) = self.vfp_instruction(stem, suffix, &opcode, &operands)? {
            return Ok(word);
        }
        if !suffix.is_empty() {
            return Err(RasError::EncodingError(format!(
                "Unsupported ARM instruction: {opcode}"
            )));
        }
        self.core_instruction(stem, &opcode, &operands)
    }

    fn core_instruction(
        &self,
        stem: &str,
        opcode: &str,
        operands: &[String],
    ) -> Result<u32, RasError> {
        // movw/movt before the data-processing table, so that `mov` does not
        // swallow them through its condition suffix.
        for (base, bits) in [("movw", 0x0300_0000u32), ("movt", 0x0340_0000u32)] {
            if let Some((false, cond)) = suffixes(stem, base) {
                operand_count(opcode, operands, 2)?;
                let rd = gpr(&operands[0])?;
                let value = immediate(&operands[1])?;
                if !(0..=0xffff).contains(&value) {
                    return Err(RasError::EncodingError(format!(
                        "{base} immediate {value} does not fit in 16 bits"
                    )));
                }
                let value = value as u32;
                return Ok((cond << 28)
                    | bits
                    | ((value >> 12) << 16)
                    | (rd << 12)
                    | (value & 0xfff));
            }
        }

        for base in ["lsl", "lsr", "asr", "ror"] {
            if let Some((set, cond)) = suffixes(stem, base) {
                operand_count(opcode, operands, 3)?;
                let rd = gpr(&operands[0])?;
                let kind = shift_type(base).unwrap_or_default();
                let rm = gpr(&operands[1])?;
                let amount = operands[2].trim();
                let field = if let Some(amount) = amount.strip_prefix('#') {
                    (shift_amount(kind, immediate(amount)?)? << 7) | (kind << 5) | rm
                } else {
                    (gpr(amount)? << 8) | (kind << 5) | (1 << 4) | rm
                };
                return Ok((cond << 28) | (13 << 21) | (u32::from(set) << 20) | (rd << 12) | field);
            }
        }

        for base in [
            "and", "eor", "sub", "rsb", "add", "adc", "sbc", "rsc", "tst", "teq", "cmp", "cmn",
            "orr", "mov", "bic", "mvn",
        ] {
            let Some((set, cond)) = suffixes(stem, base) else {
                continue;
            };
            let op = data_processing(base).unwrap_or_default();
            let compare = matches!(base, "tst" | "teq" | "cmp" | "cmn");
            let move_form = matches!(base, "mov" | "mvn");
            if operands.len() < 2 {
                return Err(RasError::EncodingError(format!(
                    "{opcode} takes at least two operands"
                )));
            }
            let (rd, rn, rest) = if compare {
                (0, gpr(&operands[0])?, &operands[1..])
            } else if move_form {
                (gpr(&operands[0])?, 0, &operands[1..])
            } else {
                (gpr(&operands[0])?, gpr(&operands[1])?, &operands[2..])
            };
            if rest.is_empty() {
                return Err(RasError::EncodingError(format!(
                    "{opcode} is missing its shifter operand"
                )));
            }
            let (i, field) = shifter_operand(rest)?;
            let set = set || compare;
            return Ok((cond << 28)
                | (i << 25)
                | (op << 21)
                | (u32::from(set) << 20)
                | (rn << 16)
                | (rd << 12)
                | field);
        }

        if let Some((set, cond)) = suffixes(stem, "mul") {
            operand_count(opcode, operands, 3)?;
            let rd = gpr(&operands[0])?;
            let rm = gpr(&operands[1])?;
            let rs = gpr(&operands[2])?;
            return Ok((cond << 28) | (u32::from(set) << 20) | (rd << 16) | (rs << 8) | 0x90 | rm);
        }

        for (base, bits) in [("sdiv", 0x0710_f010u32), ("udiv", 0x0730_f010u32)] {
            if let Some((false, cond)) = suffixes(stem, base) {
                operand_count(opcode, operands, 3)?;
                let rd = gpr(&operands[0])?;
                let rn = gpr(&operands[1])?;
                let rm = gpr(&operands[2])?;
                return Ok((cond << 28) | bits | (rd << 16) | (rm << 8) | rn);
            }
        }

        for (base, bits) in [
            ("clz", 0x016f_0f10u32),
            ("rbit", 0x06ff_0f30),
            ("rev16", 0x06bf_0fb0),
            ("rev", 0x06bf_0f30),
            ("sxtb", 0x06af_0070),
            ("sxth", 0x06bf_0070),
            ("uxtb", 0x06ef_0070),
            ("uxth", 0x06ff_0070),
        ] {
            if let Some((false, cond)) = suffixes(stem, base) {
                operand_count(opcode, operands, 2)?;
                let rd = gpr(&operands[0])?;
                let rm = gpr(&operands[1])?;
                return Ok((cond << 28) | bits | (rd << 12) | rm);
            }
        }

        // Word and byte loads and stores with an immediate offset.
        for (base, bits) in [
            ("ldrb", 0x0450_0000u32),
            ("ldr", 0x0410_0000),
            ("strb", 0x0440_0000),
            ("str", 0x0400_0000),
        ] {
            if let Some((false, cond)) = suffixes(stem, base) {
                reject_post_index(opcode, operands)?;
                operand_count(opcode, operands, 2)?;
                let rt = gpr(&operands[0])?;
                let address = memory(&operands[1])?;
                let magnitude = address.offset.unsigned_abs();
                if magnitude > 0xfff {
                    return Err(RasError::EncodingError(format!(
                        "{base} offset {} does not fit in 12 bits",
                        address.offset
                    )));
                }
                let up = u32::from(address.offset >= 0);
                return Ok((cond << 28)
                    | bits
                    | (1 << 24)
                    | (up << 23)
                    | (address.base << 16)
                    | (rt << 12)
                    | magnitude as u32);
            }
        }

        // Halfword and signed byte loads and stores, whose immediate offset is
        // split across two four-bit fields.
        for (base, load, kind) in [
            ("ldrsb", true, 2u32),
            ("ldrsh", true, 3),
            ("ldrh", true, 1),
            ("strh", false, 1),
        ] {
            if let Some((false, cond)) = suffixes(stem, base) {
                reject_post_index(opcode, operands)?;
                operand_count(opcode, operands, 2)?;
                let rt = gpr(&operands[0])?;
                let address = memory(&operands[1])?;
                let magnitude = address.offset.unsigned_abs();
                if magnitude > 0xff {
                    return Err(RasError::EncodingError(format!(
                        "{base} offset {} does not fit in 8 bits",
                        address.offset
                    )));
                }
                let magnitude = magnitude as u32;
                let up = u32::from(address.offset >= 0);
                return Ok((cond << 28)
                    | (1 << 24)
                    | (up << 23)
                    | (1 << 22)
                    | (u32::from(load) << 20)
                    | (address.base << 16)
                    | (rt << 12)
                    | ((magnitude >> 4) << 8)
                    | (1 << 7)
                    | (kind << 5)
                    | (1 << 4)
                    | (magnitude & 0xf));
            }
        }

        for (base, bits) in [("push", 0x092d_0000u32), ("pop", 0x08bd_0000)] {
            if let Some((false, cond)) = suffixes(stem, base) {
                operand_count(opcode, operands, 1)?;
                let mut mask = 0u32;
                for name in register_list(&operands[0])? {
                    mask |= 1 << gpr(&name)?;
                }
                if mask == 0 {
                    return Err(RasError::EncodingError(format!("{base} list is empty")));
                }
                return Ok((cond << 28) | bits | mask);
            }
        }

        for (base, bits) in [("bx", 0x012f_ff10u32), ("blx", 0x012f_ff30)] {
            if let Some((false, cond)) = suffixes(stem, base) {
                operand_count(opcode, operands, 1)?;
                if !is_gpr(&operands[0]) {
                    return Err(RasError::EncodingError(format!(
                        "{base} to a label is not encoded; only the register form is"
                    )));
                }
                return Ok((cond << 28) | bits | gpr(&operands[0])?);
            }
        }

        if stem == "udf" {
            operand_count(opcode, operands, 1)?;
            let value = immediate(&operands[0])?;
            if !(0..=0xffff).contains(&value) {
                return Err(RasError::EncodingError(format!(
                    "udf immediate {value} does not fit in 16 bits"
                )));
            }
            let value = value as u32;
            return Ok(0xe7f0_00f0 | ((value >> 4) << 8) | (value & 0xf));
        }

        for base in ["b", "bl"] {
            if suffixes(stem, base).is_some() {
                return Err(RasError::EncodingError(format!(
                    "{opcode} needs a label target the assembler resolves"
                )));
            }
        }

        Err(RasError::EncodingError(format!(
            "Unsupported ARM instruction: {opcode}"
        )))
    }

    fn vfp_instruction(
        &self,
        stem: &str,
        suffix: &str,
        opcode: &str,
        operands: &[String],
    ) -> Result<Option<u32>, RasError> {
        if !stem.starts_with('v') {
            return Ok(None);
        }

        if stem == "vmrs" {
            operand_count(opcode, operands, 2)?;
            if operands[0].eq_ignore_ascii_case("APSR_nzcv")
                && operands[1].eq_ignore_ascii_case("fpscr")
            {
                return Ok(Some(0xeef1_fa10));
            }
            return Err(RasError::EncodingError(format!(
                "Only `vmrs APSR_nzcv, fpscr` is encoded, got {opcode} {}",
                operands.join(", ")
            )));
        }

        // Arithmetic with a .f32/.f64 size suffix.
        let size = match suffix {
            "f32" => Some(false),
            "f64" => Some(true),
            _ => None,
        };
        if let Some(double) = size {
            for (base, bits) in [
                ("vadd", 0x0e30_0a00u32),
                ("vsub", 0x0e30_0a40),
                ("vmul", 0x0e20_0a00),
                ("vdiv", 0x0e80_0a00),
            ] {
                if let Some((false, cond)) = suffixes(stem, base) {
                    operand_count(opcode, operands, 3)?;
                    let (d, vd) = vfp_split(vfp(&operands[0], double)?, double);
                    let (n, vn) = vfp_split(vfp(&operands[1], double)?, double);
                    let (m, vm) = vfp_split(vfp(&operands[2], double)?, double);
                    return Ok(Some(
                        (cond << 28)
                            | bits
                            | (u32::from(double) << 8)
                            | (d << 22)
                            | (vn << 16)
                            | (vd << 12)
                            | (n << 7)
                            | (m << 5)
                            | vm,
                    ));
                }
            }
            for (base, bits) in [
                ("vneg", 0x0eb1_0a40u32),
                ("vsqrt", 0x0eb1_0ac0),
                ("vcmp", 0x0eb4_0a40),
                ("vmov", 0x0eb0_0a40),
            ] {
                if let Some((false, cond)) = suffixes(stem, base) {
                    operand_count(opcode, operands, 2)?;
                    if operands[1].starts_with('#') {
                        return Err(RasError::EncodingError(format!(
                            "VFP immediate operands are not encoded: {opcode} {}",
                            operands.join(", ")
                        )));
                    }
                    let (d, vd) = vfp_split(vfp(&operands[0], double)?, double);
                    let (m, vm) = vfp_split(vfp(&operands[1], double)?, double);
                    return Ok(Some(
                        (cond << 28)
                            | bits
                            | (u32::from(double) << 8)
                            | (d << 22)
                            | (vd << 12)
                            | (m << 5)
                            | vm,
                    ));
                }
            }
        }

        if stem == "vcvt" {
            let (to, from) = suffix
                .split_once('.')
                .and_then(|(to, from)| Some((cvt_type(to)?, cvt_type(from)?)))
                .ok_or_else(|| {
                    RasError::EncodingError(format!("Unsupported ARM instruction: {opcode}"))
                })?;
            operand_count(opcode, operands, 2)?;
            let word = match (to, from) {
                // Floating point to integer, rounding toward zero.
                (CvtType::S32 | CvtType::U32, CvtType::F32 | CvtType::F64) => {
                    let double = from == CvtType::F64;
                    let opc2 = if to == CvtType::S32 { 5 } else { 4 };
                    let (d, vd) = vfp_split(vfp(&operands[0], false)?, false);
                    let (m, vm) = vfp_split(vfp(&operands[1], double)?, double);
                    0x0eb8_0ac0
                        | (opc2 << 16)
                        | (u32::from(double) << 8)
                        | (d << 22)
                        | (vd << 12)
                        | (m << 5)
                        | vm
                }
                // Integer to floating point.
                (CvtType::F32 | CvtType::F64, CvtType::S32 | CvtType::U32) => {
                    let double = to == CvtType::F64;
                    let signed = from == CvtType::S32;
                    let (d, vd) = vfp_split(vfp(&operands[0], double)?, double);
                    let (m, vm) = vfp_split(vfp(&operands[1], false)?, false);
                    0x0eb8_0a40
                        | (u32::from(signed) << 7)
                        | (u32::from(double) << 8)
                        | (d << 22)
                        | (vd << 12)
                        | (m << 5)
                        | vm
                }
                // Between single and double precision.
                (CvtType::F64, CvtType::F32) | (CvtType::F32, CvtType::F64) => {
                    let double = to == CvtType::F64;
                    let (d, vd) = vfp_split(vfp(&operands[0], double)?, double);
                    let (m, vm) = vfp_split(vfp(&operands[1], !double)?, !double);
                    0x0eb7_0ac0 | (u32::from(!double) << 8) | (d << 22) | (vd << 12) | (m << 5) | vm
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "Unsupported ARM instruction: {opcode}"
                    )));
                }
            };
            return Ok(Some((COND_AL << 28) | word));
        }

        if suffix.is_empty()
            && let Some((false, cond)) = suffixes(stem, "vmov")
        {
            return self.vmov(cond, opcode, operands).map(Some);
        }

        for (base, load) in [("vldr", true), ("vstr", false)] {
            if suffix.is_empty()
                && let Some((false, cond)) = suffixes(stem, base)
            {
                operand_count(opcode, operands, 2)?;
                let double = is_vfp(&operands[0], true);
                let (d, vd) = vfp_split(vfp(&operands[0], double)?, double);
                let address = memory(&operands[1])?;
                if address.offset % 4 != 0 || address.offset.unsigned_abs() > 1020 {
                    return Err(RasError::EncodingError(format!(
                        "{base} offset {} is not a multiple of four within 1020",
                        address.offset
                    )));
                }
                let up = u32::from(address.offset >= 0);
                return Ok(Some(
                    (cond << 28)
                        | 0x0d00_0a00
                        | (u32::from(double) << 8)
                        | (up << 23)
                        | (d << 22)
                        | (u32::from(load) << 20)
                        | (address.base << 16)
                        | (vd << 12)
                        | (address.offset.unsigned_abs() / 4) as u32,
                ));
            }
        }

        for (base, bits) in [("vpush", 0x0d2d_0a00u32), ("vpop", 0x0cbd_0a00)] {
            if suffix.is_empty()
                && let Some((false, cond)) = suffixes(stem, base)
            {
                operand_count(opcode, operands, 1)?;
                let list = register_list(&operands[0])?;
                let double = list.first().is_some_and(|r| is_vfp(r, true));
                let (first, count) = consecutive(&list, double)?;
                let (d, vd) = vfp_split(first, double);
                let words = if double { count * 2 } else { count };
                if words > 255 || (double && count > 16) {
                    return Err(RasError::EncodingError(format!("{base} list is too long")));
                }
                return Ok(Some(
                    (cond << 28) | bits | (u32::from(double) << 8) | (d << 22) | (vd << 12) | words,
                ));
            }
        }

        Ok(None)
    }

    /// `vmov` between core registers and VFP registers.
    fn vmov(&self, cond: u32, opcode: &str, operands: &[String]) -> Result<u32, RasError> {
        match operands.len() {
            2 if is_vfp(&operands[0], false) && is_gpr(&operands[1]) => {
                let (n, vn) = vfp_split(vfp(&operands[0], false)?, false);
                let rt = gpr(&operands[1])?;
                Ok((cond << 28) | 0x0e00_0a10 | (vn << 16) | (rt << 12) | (n << 7))
            }
            2 if is_gpr(&operands[0]) && is_vfp(&operands[1], false) => {
                let rt = gpr(&operands[0])?;
                let (n, vn) = vfp_split(vfp(&operands[1], false)?, false);
                Ok((cond << 28) | 0x0e10_0a10 | (vn << 16) | (rt << 12) | (n << 7))
            }
            3 if is_vfp(&operands[0], true) => {
                let (m, vm) = vfp_split(vfp(&operands[0], true)?, true);
                let rt = gpr(&operands[1])?;
                let rt2 = gpr(&operands[2])?;
                Ok((cond << 28) | 0x0c40_0b10 | (rt2 << 16) | (rt << 12) | (m << 5) | vm)
            }
            3 if is_vfp(&operands[2], true) => {
                let rt = gpr(&operands[0])?;
                let rt2 = gpr(&operands[1])?;
                let (m, vm) = vfp_split(vfp(&operands[2], true)?, true);
                Ok((cond << 28) | 0x0c50_0b10 | (rt2 << 16) | (rt << 12) | (m << 5) | vm)
            }
            _ => Err(RasError::EncodingError(format!(
                "Unsupported vmov form: {opcode} {}",
                operands.join(", ")
            ))),
        }
    }
}

impl InstructionEncoder for Armv7Encoder {
    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let word = self.encode(inst)?;
        self.position += 4;
        Ok(word.to_le_bytes().to_vec())
    }

    fn current_position(&self) -> usize {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(text: &str) -> u32 {
        let (opcode, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        let operands = if rest.trim().is_empty() {
            Vec::new()
        } else {
            rest.split(',').map(|o| o.trim().to_string()).collect()
        };
        let mut encoder = Armv7Encoder::new();
        let bytes = encoder
            .encode_instruction(&ParsedInstruction {
                opcode: opcode.to_string(),
                operands,
            })
            .unwrap_or_else(|e| panic!("{text}: {e}"));
        u32::from_le_bytes(bytes.try_into().unwrap())
    }

    #[test]
    fn data_processing_forms() {
        assert_eq!(encode("add r4, r4, #1"), 0xe2844001);
        assert_eq!(encode("sub sp, sp, #640"), 0xe24ddd0a);
        assert_eq!(encode("bic sp, sp, #15"), 0xe3cdd00f);
        assert_eq!(encode("mov r0, r1"), 0xe1a00001);
        assert_eq!(encode("cmp r4, #0"), 0xe3540000);
        assert_eq!(encode("subs r0, r0, #1"), 0xe2500001);
        assert_eq!(encode("add r4, r4, r4, lsr #5"), 0xe08442a4);
    }

    #[test]
    fn condition_is_a_field() {
        assert_eq!(encode("moveq r4, #1") & 0xf000_0000, 0x0000_0000);
        assert_eq!(encode("movne r4, #1") & 0xf000_0000, 0x1000_0000);
        assert_eq!(encode("mvnlt r0, #0") & 0xf000_0000, 0xb000_0000);
        assert_eq!(
            encode("moveq r4, #1") & 0x0fff_ffff,
            encode("mov r4, #1") & 0x0fff_ffff
        );
    }
}
