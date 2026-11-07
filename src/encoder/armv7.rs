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

    /// The ARMv7-A (A32) encoder and the ELF32 ARM objects it writes.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    mod programs {
        use crate::Ras;
        use crate::test_support::TempDir;
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
        use std::process::Command;

        const PREAMBLE: &str = ".syntax unified\n.arch armv7-a\n.arch_extension idiv\n\
                                .fpu vfpv3-d16\n.eabi_attribute 28, 1\n.arm\n";

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
                Ras::new(TargetArchitecture::Armv7, TargetOperatingSystem::Linux)
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

            fn words(&self, name: &str) -> Vec<u32> {
                self.section(name)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32_at(word, 0))
                    .collect()
            }

            /// Every relocation of `name` as (offset, symbol, type).
            fn relocations(&self, name: &str) -> Vec<(u32, String, u32)> {
                let strings = self.section(".strtab").to_vec();
                let symtab = self.section(".symtab").to_vec();
                self.section(name)
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|entry| {
                        let info = u32_at(entry, 4);
                        let at = u32_at(&symtab[(info >> 8) as usize * 16..], 0) as usize;
                        let symbol = strings[at..].split(|b| *b == 0).next().unwrap();
                        (
                            u32_at(entry, 0),
                            String::from_utf8_lossy(symbol).into_owned(),
                            info & 0xff,
                        )
                    })
                    .collect()
            }
        }

        fn encode(line: &str) -> u32 {
            let object = Object::assemble(
                "armv7-encode",
                &format!("{PREAMBLE}.text\n.globl m\nm:\n    {line}\n"),
            );
            object.words(".text")[0]
        }

        fn error(line: &str) -> String {
            let tmp = TempDir::new("armv7-refuse");
            Ras::new(TargetArchitecture::Armv7, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(
                    &format!("{PREAMBLE}.text\n.globl m\nm:\n    {line}\n"),
                    &tmp.path().join("out.o"),
                )
                .unwrap_err()
                .to_string()
        }

        /// Every mnemonic lamina's armv7 backend emits, with the A32 word it encodes
        /// to. The condition occupies bits 31:28 of all of them.
        const ENCODINGS: &[(&str, u32)] = &[
            ("add r4, r4, #1", 0xe2844001),
            ("add r4, r4, r5", 0xe0844005),
            ("add r4, r4, r4, lsr #5", 0xe08442a4),
            ("and r5, r5, #31", 0xe205501f),
            ("and r5, r6, r4, lsr #3", 0xe00651a4),
            ("asr r6, r4, r5", 0xe1a06554),
            ("bic sp, sp, #15", 0xe3cdd00f),
            ("bx lr", 0xe12fff1e),
            ("blx r4", 0xe12fff34),
            ("clz r4, r5", 0xe16f4f15),
            ("cmn r5, #1", 0xe3750001),
            ("cmp r4, #0", 0xe3540000),
            ("eor r6, r4, r5", 0xe0246005),
            ("ldr r4, [r11, #-8]", 0xe51b4008),
            ("ldr r5, [r4]", 0xe5945000),
            ("ldrb r4, [sp]", 0xe5dd4000),
            ("ldrsb r5, [r4]", 0xe1d450d0),
            ("ldrsh r5, [r4]", 0xe1d450f0),
            ("lsl r6, r4, r5", 0xe1a06514),
            ("lsr r4, r4, #5", 0xe1a042a4),
            ("lsr r6, r4, r5", 0xe1a06534),
            ("mov r0, r1", 0xe1a00001),
            ("mov r4, #7", 0xe3a04007),
            ("moveq r4, #1", 0x03a04001),
            ("movge r4, #1", 0xa3a04001),
            ("movgt r4, #1", 0xc3a04001),
            ("movhi r4, #1", 0x83a04001),
            ("movlo r4, #1", 0x33a04001),
            ("movls r4, #1", 0x93a04001),
            ("movlt r4, #1", 0xb3a04001),
            ("movmi r4, #1", 0x43a04001),
            ("movne r4, #1", 0x13a04001),
            ("movt r5, #65535", 0xe34f5fff),
            ("movw r5, #65535", 0xe30f5fff),
            ("mul r4, r4, r5", 0xe0040594),
            ("mvnlt r0, #0", 0xb3e00000),
            ("orr r4, r4, r5", 0xe1844005),
            (
                "pop {r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}",
                0xe8bd5ff0,
            ),
            (
                "push {r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}",
                0xe92d5ff0,
            ),
            ("rbit r4, r5", 0xe6ff4f35),
            ("rev r4, r5", 0xe6bf4f35),
            ("rev16 r4, r5", 0xe6bf4fb5),
            ("ror r4, r4, r5", 0xe1a04574),
            ("rsb r6, r5, #0", 0xe2656000),
            ("sdiv r6, r4, r5", 0xe716f514),
            ("str r4, [r11, #-8]", 0xe50b4008),
            ("str r5, [r4]", 0xe5845000),
            ("strb r5, [r4]", 0xe5c45000),
            ("strh r5, [r4]", 0xe1c450b0),
            ("sub r4, r11, #16", 0xe24b4010),
            ("subs r0, r0, #1", 0xe2500001),
            ("sxtb r0, r4", 0xe6af0074),
            ("sxth r0, r4", 0xe6bf0074),
            ("udf #0", 0xe7f000f0),
            ("udiv r6, r4, r5", 0xe736f514),
            ("uxtb r4, r5", 0xe6ef4075),
            ("uxth r4, r5", 0xe6ff4075),
            ("vadd.f32 s16, s16, s17", 0xee388a28),
            ("vadd.f64 d12, d12, d13", 0xee3ccb0d),
            ("vcmp.f64 d12, d13", 0xeeb4cb4d),
            ("vcvt.f32.f64 s16, d12", 0xeeb78bcc),
            ("vcvt.f32.s32 s16, s17", 0xeeb88ae8),
            ("vcvt.f32.u32 s16, s17", 0xeeb88a68),
            ("vcvt.f64.f32 d12, s16", 0xeeb7cac8),
            ("vcvt.f64.s32 d12, s16", 0xeeb8cbc8),
            ("vcvt.f64.u32 d12, s16", 0xeeb8cb48),
            ("vcvt.s32.f32 s16, s17", 0xeebd8ae8),
            ("vcvt.s32.f64 s16, d12", 0xeebd8bcc),
            ("vcvt.u32.f32 s16, s17", 0xeebc8ae8),
            ("vcvt.u32.f64 s16, d12", 0xeebc8bcc),
            ("vdiv.f32 s16, s16, s17", 0xee888a28),
            ("vdiv.f64 d12, d12, d13", 0xee8ccb0d),
            ("vldr d12, [r11, #-8]", 0xed1bcb02),
            ("vldr s16, [r12]", 0xed9c8a00),
            ("vmov s16, r4", 0xee084a10),
            ("vmov r4, s16", 0xee184a10),
            ("vmov d12, r12, r4", 0xec44cb1c),
            ("vmov r0, r1, d0", 0xec510b10),
            ("vmoveq.f64 d12, d13", 0x0eb0cb4d),
            ("vmrs APSR_nzcv, fpscr", 0xeef1fa10),
            ("vmul.f32 s16, s16, s17", 0xee288a28),
            ("vmul.f64 d12, d12, d13", 0xee2ccb0d),
            ("vneg.f64 d12, d13", 0xeeb1cb4d),
            ("vpop {d8, d9, d10, d11, d12, d13, d14, d15}", 0xecbd8b10),
            ("vpush {d8, d9, d10, d11, d12, d13, d14, d15}", 0xed2d8b10),
            ("vsqrt.f32 s16, s17", 0xeeb18ae8),
            ("vsqrt.f64 d12, d13", 0xeeb1cbcd),
            ("vstr d12, [r11, #-8]", 0xed0bcb02),
            ("vstr s16, [r12]", 0xed8c8a00),
            ("vsub.f64 d12, d12, d13", 0xee3ccb4d),
        ];

        #[test]
        fn encodes_the_instruction_set_lamina_emits() {
            for (line, expected) in ENCODINGS {
                assert_eq!(encode(line), *expected, "{line}");
            }
        }

        /// A conditional mnemonic is the same instruction with a different top nibble,
        /// not an opcode of its own.
        #[test]
        fn the_condition_is_the_top_nibble() {
            let unconditional = encode("mov r4, #1");
            assert_eq!(unconditional >> 28, 0xe);
            for (line, condition) in [
                ("moveq r4, #1", 0x0),
                ("movne r4, #1", 0x1),
                ("movlo r4, #1", 0x3),
                ("movmi r4, #1", 0x4),
                ("movhi r4, #1", 0x8),
                ("movls r4, #1", 0x9),
                ("movge r4, #1", 0xa),
                ("movlt r4, #1", 0xb),
                ("movgt r4, #1", 0xc),
            ] {
                let word = encode(line);
                assert_eq!(word >> 28, condition, "{line}");
                assert_eq!(word & 0x0fff_ffff, unconditional & 0x0fff_ffff, "{line}");
            }
        }

        /// movw and movt carry their sixteen-bit immediate split across imm4 at bits
        /// 19:16 and imm12 at bits 11:0.
        #[test]
        fn movw_and_movt_split_the_immediate() {
            // 0xbeef splits into imm4 0xb at bits 19:16 and imm12 0xeef.
            assert_eq!(encode("movw r0, #48879"), 0xe30b_0eef);
            assert_eq!(encode("movt r0, #48879"), 0xe34b_0eef);
            assert_eq!(encode("movw r5, #0"), 0xe3005000);
            assert_eq!(encode("movw r5, #65535"), 0xe30f5fff);
            for line in ["movw r0, #65536", "movt r0, #-1"] {
                assert!(error(line).contains("16 bits"), "{line}");
            }
        }

        /// An encoding ras cannot produce correctly names itself rather than guessing
        /// a bit pattern.
        #[test]
        fn refuses_what_it_does_not_encode_by_name() {
            for (line, fragment) in [
                ("nop", "Unsupported ARM instruction: nop"),
                ("ldm r0, {r1, r2}", "Unsupported ARM instruction: ldm"),
                ("smull r0, r1, r2, r3", "Unsupported ARM instruction: smull"),
                ("bkpt #0", "Unsupported ARM instruction: bkpt"),
                (
                    "mcr p15, 0, r0, c1, c0, 0",
                    "Unsupported ARM instruction: mcr",
                ),
                (
                    "vmla.f64 d0, d1, d2",
                    "Unsupported ARM instruction: vmla.f64",
                ),
                (
                    "vcvt.f16.f32 s0, s1",
                    "Unsupported ARM instruction: vcvt.f16.f32",
                ),
                ("add r0, r0, #257", "is not an 8-bit value rotated"),
                (
                    "ldr r0, [r1, r2]",
                    "register-offset addressing is not encoded",
                ),
                ("ldr r0, [r1], #4", "post-indexed addressing is not encoded"),
                ("ldr r0, [r1, #4]!", "write-back addressing is not encoded"),
                ("ldr r0, [r1, #4096]", "does not fit in 12 bits"),
                ("ldrsb r0, [r1, #256]", "does not fit in 8 bits"),
                ("vldr s0, [r1, #3]", "is not a multiple of four within 1020"),
                (
                    "vmov.f32 s0, #1.0",
                    "VFP immediate operands are not encoded",
                ),
                ("vmrs r0, fpscr", "Only `vmrs APSR_nzcv, fpscr` is encoded"),
                ("vpush {d8, d10}", "VFP register list is not consecutive"),
                ("ror r0, r1, #0", "ARM shift amount 0 out of range"),
                ("mov r16, #1", "Unknown ARM register: r16"),
                ("vadd.f64 d12, d12, s3", "Unknown ARM d-register: s3"),
            ] {
                let message = error(line);
                assert!(message.contains(fragment), "{line}: {message}");
            }
        }

        /// ras encodes A32 only, and says so rather than assembling Thumb as ARM.
        #[test]
        fn refuses_directives_it_cannot_honour() {
            for (source, fragment) in [
                (".thumb\n", "selects Thumb"),
                (".syntax divided\n", "unified ARM syntax only"),
                (".arch armv8-a\n", "no build attributes for `.arch armv8-a`"),
                (".fpu neon\n", "no build attributes for `.fpu neon`"),
            ] {
                let tmp = TempDir::new("armv7-directive");
                let message = Ras::new(TargetArchitecture::Armv7, TargetOperatingSystem::Linux)
                    .unwrap()
                    .assemble(source, &tmp.path().join("out.o"))
                    .unwrap_err()
                    .to_string();
                assert!(message.contains(fragment), "{source}: {message}");
            }
        }

        /// A branch inside the object is resolved here, counting from the address the
        /// processor computes, which is the branch plus eight. One that leaves the
        /// object becomes a relocation whose addend is the displacement field itself,
        /// because ARM objects carry Elf32_Rel.
        #[test]
        fn branches_resolve_locally_and_relocate_outward() {
            let object = Object::assemble(
                "armv7-branch",
                &format!(
                    "{PREAMBLE}.text\n.globl m\nm:\n  b 1f\n  beq 1f\n1:\n  bl other\n  bx lr\n"
                ),
            );
            assert_eq!(
                object.words(".text"),
                [
                    // b   .+8  -> (8 - (0 + 8)) / 4 == 0
                    0xea00_0000,
                    // beq .+4  -> (8 - (4 + 8)) / 4 == -1
                    0x0aff_ffff,
                    // bl  other, with the -8 bias held in the displacement
                    0xebff_fffe,
                    0xe12f_ff1e,
                ]
            );
            assert_eq!(
                object.relocations(".rel.text"),
                [(8, "other".to_string(), 28)]
            );
        }

        /// An address is built by a movw/movt pair, which the linker completes. Both
        /// halves leave their immediate at zero, which is the Elf32_Rel addend.
        #[test]
        fn symbol_halves_become_movw_and_movt_relocations() {
            let object = Object::assemble(
                "armv7-halves",
                &format!(
                    "{PREAMBLE}.data\nvalue: .long 7\n             .text\n.globl m\nm:\n  movw r4, #:lower16:value\n  movt r4, #:upper16:value\n"
                ),
            );
            assert_eq!(object.words(".text"), [0xe300_4000, 0xe340_4000]);
            assert_eq!(
                object.relocations(".rel.text"),
                [(0, "value".to_string(), 43), (4, "value".to_string(), 44),]
            );
        }

        /// A data word pointing at a symbol is R_ARM_ABS32, and its addend is written
        /// into the word itself.
        #[test]
        fn data_pointers_keep_their_addend_in_the_field() {
            let object = Object::assemble(
                "armv7-data",
                &format!("{PREAMBLE}.data\ntext: .byte 1, 2, 3, 4\nptr: .long text+3\n"),
            );
            assert_eq!(object.section(".data"), [1, 2, 3, 4, 3, 0, 0, 0]);
            assert_eq!(
                object.relocations(".rel.data"),
                [(4, "text".to_string(), 2)]
            );
        }

        #[test]
        fn writes_an_elf32_arm_object() {
            let object = Object::assemble(
                "armv7-object",
                &format!("{PREAMBLE}.text\n.globl m\nm:\n  bl other\n  bx lr\n"),
            );
            // ELFCLASS32, ELFDATA2LSB, EM_ARM, EF_ARM_EABI_VER5.
            assert_eq!(object.bytes[4], 1);
            assert_eq!(object.bytes[5], 1);
            assert_eq!(u16_at(&object.bytes, 18), 40);
            assert_eq!(u32_at(&object.bytes, 36), 0x0500_0000);
            assert_eq!(u16_at(&object.bytes, 46), 40);
            // SHT_REL with eight-byte entries, named .rel<section>, not .rela.
            assert!(object.find(".rela.text").is_none());
            assert_eq!(object.field(".rel.text", 1), 9);
            assert_eq!(object.field(".rel.text", 9), 8);
            // SHT_ARM_ATTRIBUTES, holding the aeabi subsection the directives describe.
            assert_eq!(object.field(".ARM.attributes", 1), 0x7000_0003);
            let attributes = object.section(".ARM.attributes");
            assert_eq!(attributes[0], b'A');
            assert_eq!(u32_at(attributes, 1) as usize, attributes.len() - 1);
            assert_eq!(&attributes[5..11], b"aeabi\0");
            assert_eq!(attributes[11], 1);
            assert_eq!(u32_at(attributes, 12) as usize, attributes.len() - 11);
            assert_eq!(
                &attributes[16..],
                // Tag_CPU_name "7-A", Tag_CPU_arch v7, profile A, ARM and Thumb-2 ISA,
                // Tag_FP_arch VFPv3-D16, Tag_ABI_VFP_args VFP registers, Tag_DIV_use.
                &[
                    5, b'7', b'-', b'A', 0, 6, 10, 7, 0x41, 8, 1, 9, 2, 10, 4, 28, 1, 44, 2
                ]
            );
        }

        /// ras refuses armv7 for the object formats it has no ARM writer for.
        #[test]
        fn only_elf_targets_accept_armv7() {
            for os in [TargetOperatingSystem::MacOS, TargetOperatingSystem::Windows] {
                assert!(
                    !crate::target::is_object_file_supported(TargetArchitecture::Armv7, os),
                    "{os:?}"
                );
            }
            assert!(crate::target::is_object_file_supported(
                TargetArchitecture::Armv7,
                TargetOperatingSystem::Linux
            ));
        }

        /// The decisive check: an object ras assembled, linked and run.
        #[test]
        fn a_linked_program_runs_and_prints() {
            let have = |tool: &str| {
                Command::new(tool)
                    .arg("--version")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
            };
            if !have("arm-linux-gnueabihf-gcc") || !have("qemu-arm-static") {
                return;
            }

            let tmp = TempDir::new("armv7-run");
            let object = tmp.path().join("program.o");
            Ras::new(TargetArchitecture::Armv7, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(
                    &format!(
                        "{PREAMBLE}.section .note.GNU-stack,\"\",%progbits\n                 .data\nmessage: .asciz \"armv7 %d\\n\"\n                 .text\n.globl main\nmain:\n                 \x20 push {{r4, lr}}\n                 \x20 movw r0, #:lower16:message\n  movt r0, #:upper16:message\n                 \x20 movw r1, #20\n  movw r2, #2\n  mul r1, r1, r2\n  add r1, r1, #2\n                 \x20 bl printf\n  movw r0, #0\n  pop {{r4, pc}}\n"
                    ),
                    &object,
                )
                .unwrap();

            let program = tmp.path().join("program");
            let link = Command::new("arm-linux-gnueabihf-gcc")
                .args(["-static", "-no-pie"])
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
            let run = Command::new("qemu-arm-static")
                .arg(&program)
                .output()
                .unwrap();
            assert!(run.status.success(), "program exited {:?}", run.status);
            assert_eq!(String::from_utf8_lossy(&run.stdout), "armv7 42\n");
        }
    }
}
