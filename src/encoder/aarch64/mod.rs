pub mod fixed;
pub mod ldst_imm64;

use crate::{
    encoder::aarch64::fixed::*,
    encoder::traits::{InstructionEncoder, ParsedInstruction},
    error::RasError,
};

#[derive(Default)]
pub struct AArch64Encoder {
    position: usize,
}

fn error(message: impl Into<String>) -> RasError {
    RasError::EncodingError(message.into())
}

pub(crate) fn register(reg: &str) -> Result<u8, RasError> {
    match reg {
        "sp" | "wsp" | "xzr" | "wzr" => Ok(31),
        "fp" => Ok(29),
        "lr" => Ok(30),
        _ => {
            let fp = reg.starts_with(['s', 'd']);
            reg.strip_prefix(['x', 'w', 's', 'd'])
                .and_then(|n| n.parse::<u8>().ok())
                .filter(|n| *n < 31 || (fp && *n == 31))
                .ok_or_else(|| error(format!("Invalid AArch64 register: {reg}")))
        }
    }
}

pub(crate) fn immediate(text: &str) -> Result<i64, RasError> {
    let text = text.trim().trim_start_matches('#');
    let (negative, digits) = text.strip_prefix('-').map_or((false, text), |s| (true, s));
    let n = if let Some(hex) = digits.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
    } else {
        digits.parse::<u64>()
    }
    .map_err(|_| error(format!("Invalid immediate: {text}")))?;
    Ok(if negative {
        (n as i64).wrapping_neg()
    } else {
        n as i64
    })
}

fn add_sub_imm12(imm: i64, shift: Option<&str>) -> Result<(i64, u32), RasError> {
    let fits = |v: i64| (0..=4095).contains(&v);
    match shift {
        None if fits(imm) => Ok((imm, 0)),
        None if imm & 0xfff == 0 && fits(imm >> 12) && imm > 0 => Ok((imm >> 12, 1)),
        Some("lsl #0") if fits(imm) => Ok((imm, 0)),
        Some("lsl #12") if fits(imm) => Ok((imm, 1)),
        _ => Err(error("Arithmetic immediate out of range")),
    }
}

pub(crate) fn gpr(reg: &str) -> Result<(u32, bool), RasError> {
    let wide = match reg.as_bytes().first() {
        Some(b'x' | b'w') => reg.starts_with('x'),
        _ if matches!(reg, "sp" | "fp" | "lr") => true,
        _ => return Err(error(format!("Expected a general purpose register: {reg}"))),
    };
    Ok((u32::from(register(reg)?), wide))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DataKind {
    W,
    X,
    S,
    D,
}

/// A load or store data register and its kind; a stack pointer is not a data register.
fn data_reg(reg: &str) -> Result<(u32, DataKind), RasError> {
    if is_stack_pointer(reg) {
        return Err(error(format!("Expected a load or store register: {reg}")));
    }
    let kind = match reg.as_bytes().first() {
        Some(b'w') => DataKind::W,
        Some(b'x') => DataKind::X,
        Some(b's') => DataKind::S,
        Some(b'd') => DataKind::D,
        _ => return Err(error(format!("Expected a load or store register: {reg}"))),
    };
    Ok((u32::from(register(reg)?), kind))
}

fn exclusive_data(op: &str, reg: &str) -> Result<(u32, u32), RasError> {
    let (rt, kind) = data_reg(reg)?;
    if op.ends_with('b') || op.ends_with('h') {
        if kind != DataKind::W {
            return Err(error(format!("{op} requires a w register")));
        }
        let size = if op.ends_with('b') { 0 } else { 0x4000_0000 };
        return Ok((rt, size));
    }
    match kind {
        DataKind::W => Ok((rt, 0x8000_0000)),
        DataKind::X => Ok((rt, 0xc000_0000)),
        _ => Err(error(format!("{op} requires a general purpose register"))),
    }
}

fn exclusive_base(address: &str) -> Result<u32, RasError> {
    let inside = address
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| error(format!("Expected [base] address: {address}")))?;
    let (rn, wide) = gpr(inside.trim())?;
    if !wide {
        return Err(error("Address base must be an x register or sp"));
    }
    Ok(rn)
}

fn vector_8b(reg: &str) -> Result<u32, RasError> {
    let number = reg
        .strip_prefix('v')
        .and_then(|r| r.strip_suffix(".8b"))
        .and_then(|n| n.parse::<u32>().ok())
        .filter(|n| *n < 32)
        .ok_or_else(|| error(format!("Expected a v<n>.8b register: {reg}")))?;
    Ok(number)
}

fn scalar_b(reg: &str) -> Result<u32, RasError> {
    reg.strip_prefix('b')
        .and_then(|n| n.parse::<u32>().ok())
        .filter(|n| *n < 32)
        .ok_or_else(|| error(format!("Expected a b<n> register: {reg}")))
}

/// Register numbers of the first `N` operands, all general purpose registers of one width.
fn gpr_regs<const N: usize>(p: &[String]) -> Result<([u32; N], bool), RasError> {
    let mut regs = [0; N];
    let mut wide = false;
    for (i, slot) in regs.iter_mut().enumerate() {
        let reg = p.get(i).ok_or_else(|| error("Missing register operand"))?;
        let (number, reg_wide) = gpr(reg)?;
        if is_stack_pointer(reg) {
            return Err(error(format!("Stack pointer not allowed here: {reg}")));
        }
        if i == 0 {
            wide = reg_wide;
        } else if reg_wide != wide {
            return Err(error("Mismatched register widths"));
        }
        *slot = number;
    }
    Ok((regs, wide))
}

fn encode_add_sub(op: &str, p: &[String]) -> Result<u32, RasError> {
    let compare = op == "cmp" || op == "cmn";
    let rn_index = usize::from(!compare);
    let (min, max) = if compare { (2, 3) } else { (3, 4) };
    if !(min..=max).contains(&p.len()) {
        return Err(error(format!("Wrong operand count for {op}")));
    }
    let (lead, wide) = gpr(&p[0])?;
    let rd = if compare { 31 } else { lead };
    let (rn, rn_wide) = gpr(&p[rn_index])?;
    if rn_wide != wide {
        return Err(error(format!("Mismatched register widths for {op}")));
    }
    let sf = if wide { 0x8000_0000 } else { 0 };
    let flags = match op {
        "cmp" => 0x6000_0000,
        "sub" => 0x4000_0000,
        "subs" => 0x6000_0000,
        "cmn" => 0x2000_0000,
        _ => 0,
    };
    let src = p[rn_index + 1].as_str();
    let tail = p.get(rn_index + 2).map(|s| s.trim());
    if src.starts_with('#') {
        let (imm, shift) = add_sub_imm12(immediate(src)?, tail)?;
        return Ok(0x1100_0000
            | sf
            | flags
            | (shift << 22)
            | ((imm as u32) << 10)
            | (rn << 5)
            | rd);
    }
    if is_stack_pointer(src) {
        return Err(error(format!(
            "{op} cannot take the stack pointer as an operand"
        )));
    }
    let (rm, rm_wide) = gpr(src)?;
    if rm_wide != wide {
        return Err(error(format!("Mismatched register widths for {op}")));
    }
    if is_stack_pointer(&p[rn_index]) || (!compare && is_stack_pointer(&p[0])) {
        let amount = extended_amount(tail)?;
        let option = if wide { 0b011 } else { 0b010 };
        return Ok(0x0b20_0000
            | sf
            | flags
            | (rm << 16)
            | (option << 13)
            | (amount << 10)
            | (rn << 5)
            | rd);
    }
    if let Some(option) = tail.and_then(extend_option) {
        let amount = extend_amount(tail.unwrap_or_default())?;
        return Ok(0x0b20_0000
            | sf
            | flags
            | (rm << 16)
            | (option << 13)
            | (amount << 10)
            | (rn << 5)
            | rd);
    }
    let (kind, amount) = shifted_register(tail, wide)?;
    Ok(0x0b00_0000 | sf | flags | (kind << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd)
}

/// The option field of an extended-register operand, from the ARM ARM
/// (DDI 0487) ADD/SUB (extended register) table: UXTB=000 up to SXTX=111.
fn extend_option(shift: &str) -> Option<u32> {
    Some(match shift.split(' ').next()? {
        "uxtb" => 0,
        "uxth" => 1,
        "uxtw" => 2,
        "uxtx" => 3,
        "sxtb" => 4,
        "sxth" => 5,
        "sxtw" => 6,
        "sxtx" => 7,
        _ => return None,
    })
}

fn extend_amount(shift: &str) -> Result<u32, RasError> {
    let Some((_, amount)) = shift.split_once(' ') else {
        return Ok(0);
    };
    let amount = immediate(amount)?;
    if !(0..=4).contains(&amount) {
        return Err(error(format!("Extension amount out of range: {amount}")));
    }
    Ok(amount as u32)
}

/// Logical-immediate fields `(N, immr, imms)` for a bitmask of `size` bits, or `None` when the
/// value is not a repeated rotated run of ones.
fn logical_immediate(value: u64, wide: bool) -> Option<(u32, u32, u32)> {
    let size = if wide { 64 } else { 32 };
    let mask = if size == 64 {
        u64::MAX
    } else {
        (1 << size) - 1
    };
    let v = value & mask;
    if v == 0 || v == mask {
        return None;
    }
    let mut element = size;
    while element > 2 {
        let half = element / 2;
        let low_mask = (1u64 << half) - 1;
        if v & low_mask != (v >> half) & low_mask {
            break;
        }
        element = half;
    }
    let element_mask = if element == 64 {
        u64::MAX
    } else {
        (1u64 << element) - 1
    };
    let pattern = v & element_mask;
    let ones = pattern.count_ones();
    let rotate = |x: u64, r: u32| {
        if r == 0 {
            x & element_mask
        } else {
            ((x >> r) | (x << (element - r))) & element_mask
        }
    };
    let immr = (0..element).find(|&r| rotate((1u64 << ones) - 1, r) == pattern)?;
    let n = u32::from(element == 64);
    let imms = ((!(2 * element - 1)) & 0x3f) | (ones - 1);
    Some((n, immr, imms))
}

fn encode_logical(op: &str, p: &[String]) -> Result<u32, RasError> {
    let flag_setting = op == "ands" || op == "tst";
    let rn_index = usize::from(op != "tst");
    let (min, max) = if op == "tst" { (2, 2) } else { (3, 3) };
    if !(min..=max).contains(&p.len()) {
        return Err(error(format!("Wrong operand count for {op}")));
    }
    let (lead, wide) = gpr(&p[0])?;
    let rd = if flag_setting && op == "tst" {
        31
    } else {
        lead
    };
    let (rn, rn_wide) = gpr(&p[rn_index])?;
    if rn_wide != wide || is_stack_pointer(&p[rn_index]) {
        return Err(error(format!("Invalid register operand for {op}")));
    }
    let src = p[rn_index + 1].as_str();
    let (base_imm, base_reg) = match op {
        "and" | "ands" | "tst" => (0x1200_0000u32, 0x0a00_0000u32),
        "orr" => (0x3200_0000, 0x2a00_0000),
        _ => (0x5200_0000, 0x4a00_0000),
    };
    let base_imm = if wide {
        base_imm | 0x8000_0000
    } else {
        base_imm
    };
    let base_reg = if wide {
        base_reg | 0x8000_0000
    } else {
        base_reg
    };
    let flags_imm = if flag_setting { 0x6000_0000 } else { 0 };
    let flags_reg = if flag_setting { 0x6000_0000 } else { 0 };
    if src.starts_with('#') {
        let value = immediate(src)?;
        if !wide && !(-(1 << 31)..=u32::MAX as i64).contains(&value) {
            return Err(error(format!(
                "Immediate out of range for {op} on a w register: {src}"
            )));
        }
        let (n, immr, imms) = logical_immediate(value as u64, wide)
            .ok_or_else(|| error(format!("Immediate is not a valid logical immediate: {src}")))?;
        return Ok(base_imm | flags_imm | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd);
    }
    let (rm, rm_wide) = gpr(src)?;
    if rm_wide != wide || is_stack_pointer(src) {
        return Err(error(format!("Invalid register operand for {op}")));
    }
    if p.get(rn_index + 2).is_some() {
        return Err(error(format!(
            "Shifted logical operands are not supported for {op}"
        )));
    }
    Ok(base_reg | flags_reg | (rm << 16) | (rn << 5) | rd)
}

/// Moves `value` into `rd` with one `movz` or `movn`, a bitmask `orr` from the zero register,
/// or a `movz` and `movk` chain, in that order of preference.
fn mov_immediate_words(value: u64, wide: bool, rd: u32) -> Vec<u32> {
    let sf = if wide { 0x8000_0000 } else { 0 };
    let chunks = if wide { 4 } else { 2 };
    let chunk = |i: u32| ((value >> (i * 16)) & 0xffff) as u32;
    let ones = (0..chunks).filter(|&i| chunk(i) == 0xffff).count();
    let zeros = (0..chunks).filter(|&i| chunk(i) == 0).count();
    let invert = ones > zeros;
    let skip = if invert { 0xffff } else { 0 };
    let mut words = Vec::new();
    for i in 0..chunks {
        let c = chunk(i);
        if c == skip {
            continue;
        }
        let (base, imm) = match (words.is_empty(), invert) {
            (true, false) => (0x5280_0000, c),
            (true, true) => (0x1280_0000, !c & 0xffff),
            (false, _) => (0x7280_0000, c),
        };
        words.push(base | sf | (i << 21) | (imm << 5) | rd);
    }
    if words.is_empty() {
        let base = if invert { 0x1280_0000 } else { 0x5280_0000 };
        return vec![base | sf | rd];
    }
    if words.len() > 1
        && let Some((n, immr, imms)) = logical_immediate(value, wide)
    {
        let base = if wide { 0xb200_0000 } else { 0x3200_0000 };
        return vec![base | (n << 22) | (immr << 16) | (imms << 10) | (31 << 5) | rd];
    }
    words
}

fn sf_of(wide: bool) -> u32 {
    if wide { 0x8000_0000 } else { 0 }
}

fn is_stack_pointer(reg: &str) -> bool {
    matches!(reg, "sp" | "wsp")
}

fn extended_amount(shift: Option<&str>) -> Result<u32, RasError> {
    let Some(shift) = shift else {
        return Ok(0);
    };
    let amount = shift
        .strip_prefix("lsl ")
        .ok_or_else(|| error(format!("Invalid extension: {shift}")))
        .and_then(immediate)?;
    if !(0..=4).contains(&amount) {
        return Err(error(format!("Extension amount out of range: {amount}")));
    }
    Ok(amount as u32)
}

fn shifted_register(shift: Option<&str>, wide: bool) -> Result<(u32, u32), RasError> {
    let Some(shift) = shift else {
        return Ok((0, 0));
    };
    let (kind, amount) = shift
        .split_once(' ')
        .ok_or_else(|| error(format!("Invalid shift: {shift}")))?;
    let kind = match kind {
        "lsl" => 0,
        "lsr" => 1,
        "asr" => 2,
        _ => return Err(error(format!("Invalid shift: {shift}"))),
    };
    let amount = immediate(amount)?;
    if !(0..if wide { 64 } else { 32 }).contains(&amount) {
        return Err(error(format!("Shift amount out of range: {amount}")));
    }
    Ok((kind, amount as u32))
}

pub(crate) fn condition(text: &str) -> Result<u32, RasError> {
    [
        "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al",
        "nv",
    ]
    .iter()
    .position(|s| *s == text || (*s == "hs" && text == "cs") || (*s == "lo" && text == "cc"))
    .map(|n| n as u32)
    .ok_or_else(|| error(format!("Invalid condition: {text}")))
}

fn max_operands(op: &str) -> Option<usize> {
    Some(match op {
        "nop" | "yield" | "wfe" | "wfi" | "sev" | "sevl" => 0,
        "ret" | "br" | "blr" | "svc" | "hvc" | "smc" | "brk" | "hlt" | "dmb" | "udf" => 1,
        "mov" | "uxtb" | "sxtw" | "fmov" | "cset" | "scvtf" | "ucvtf" | "fcvtzs" | "fcvtzu" => 2,
        "movz" | "movk" | "mul" | "sdiv" | "udiv" | "fmul" | "fdiv" | "fadd" | "fsub" | "lsl"
        | "lsr" | "asr" | "ror" | "lslv" | "lsrv" | "asrv" | "rorv" => 3,
        "csel" | "msub" | "sbfx" | "ubfx" => 4,
        _ => return None,
    })
}

impl AArch64Encoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn encode(&self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let op = inst.opcode.as_str();
        let p = &inst.operands;
        if max_operands(op).is_some_and(|max| p.len() > max) {
            return Err(error(format!("Too many operands for {op}")));
        }
        let arg = |i: usize| {
            p.get(i)
                .map(String::as_str)
                .ok_or_else(|| error(format!("Missing operand for {op}")))
        };
        let r = |i| register(arg(i)?).map(u32::from);
        let wide = p.first().is_some_and(|s| s.starts_with('x') || s == "sp");
        let sf = if wide { 0x8000_0000 } else { 0 };
        let word = match op {
            "ret" if p.is_empty() => 0xd65f_03c0,
            "ret" => {
                let ([rn], wide) = gpr_regs::<1>(p)?;
                if !wide {
                    return Err(error("ret takes a 64-bit register"));
                }
                0xd65f_0000 | (rn << 5)
            }
            "nop" if p.is_empty() => 0xd503_201f,
            "yield" if p.is_empty() => 0xd503_203f,
            "wfe" if p.is_empty() => 0xd503_205f,
            "wfi" if p.is_empty() => 0xd503_207f,
            "sev" if p.is_empty() => 0xd503_209f,
            "sevl" if p.is_empty() => 0xd503_20bf,
            "udf" => {
                let imm = immediate(arg(0)?)?;
                if !(0..=0xffff).contains(&imm) {
                    return Err(error(format!("udf immediate out of range: {imm}")));
                }
                imm as u32
            }
            "svc" | "hvc" | "smc" | "brk" | "hlt" => {
                let imm = immediate(arg(0)?)?;
                if !(0..=0xffff).contains(&imm) {
                    return Err(error(format!("{op} immediate out of range: {imm}")));
                }
                let base = match op {
                    "svc" => 0xd400_0001,
                    "hvc" => 0xd400_0002,
                    "smc" => 0xd400_0003,
                    "brk" => 0xd420_0000,
                    _ => 0xd440_0000,
                };
                base | ((imm as u32) << 5)
            }
            "br" | "blr" => {
                let ([rn], wide) = gpr_regs::<1>(p)?;
                if !wide {
                    return Err(error(format!("{op} takes a 64-bit register")));
                }
                let base = if op == "br" { 0xd61f_0000 } else { 0xd63f_0000 };
                base | (rn << 5)
            }
            "dmb" | "dsb" => {
                let option = match arg(0)? {
                    "oshld" => 1,
                    "oshst" => 2,
                    "osh" => 3,
                    "nshld" => 5,
                    "nshst" => 6,
                    "nsh" => 7,
                    "ishld" => 9,
                    "ishst" => 0xa,
                    "ish" => 0xb,
                    "ld" => 0xd,
                    "st" => 0xe,
                    "sy" => 0xf,
                    other => return Err(error(format!("Invalid barrier option: {other}"))),
                };
                let base = if op == "dmb" {
                    0xd503_30bf
                } else {
                    0xd503_309f
                };
                base | (option << 8)
            }
            "isb" if p.is_empty() || matches!(p.as_slice(), [sy] if sy == "sy") => 0xd503_3fdf,
            "movz" | "movk" | "movn" => {
                if is_stack_pointer(arg(0)?) {
                    return Err(error(format!("{op} cannot write the stack pointer")));
                }
                let imm = immediate(arg(1)?)?;
                let shift = if p.len() > 2 {
                    immediate(
                        arg(2)?
                            .strip_prefix("lsl ")
                            .ok_or_else(|| error("Expected lsl"))?,
                    )?
                } else {
                    0
                };
                if !(0..=65535).contains(&imm)
                    || shift % 16 != 0
                    || !(0..if wide { 64 } else { 32 }).contains(&shift)
                {
                    return Err(error("Invalid move-wide immediate or shift"));
                }
                (match op {
                    "movz" => 0x5280_0000,
                    "movn" => 0x1280_0000,
                    _ => 0x7280_0000,
                }) | sf
                    | ((shift as u32 / 16) << 21)
                    | ((imm as u32) << 5)
                    | r(0)?
            }
            "mov" => {
                let src = arg(1)?;
                if src.starts_with('#') {
                    let raw = immediate(src)?;
                    if !wide && !(-(1 << 31)..=u32::MAX as i64).contains(&raw) {
                        return Err(error(format!(
                            "mov immediate out of range for w register: {raw}"
                        )));
                    }
                    let (dst, _) = gpr(arg(0)?)?;
                    let value = raw as u64 & if wide { u64::MAX } else { u32::MAX as u64 };
                    if is_stack_pointer(arg(0)?) {
                        let (n, immr, imms) = logical_immediate(value, wide).ok_or_else(|| {
                            error(format!(
                                "mov to the stack pointer needs a logical immediate: {raw}"
                            ))
                        })?;
                        let base = if wide { 0xb200_0000 } else { 0x3200_0000 };
                        let word = base | (n << 22) | (immr << 16) | (imms << 10) | (31 << 5) | dst;
                        return Ok(word.to_le_bytes().to_vec());
                    }
                    let words = mov_immediate_words(value, wide, dst);
                    return Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect());
                } else {
                    let (dst, dst_wide) = gpr(arg(0)?)?;
                    let (source, source_wide) = gpr(src)?;
                    if dst_wide != source_wide {
                        return Err(error("Mismatched register widths for mov"));
                    }
                    if is_stack_pointer(arg(0)?) || is_stack_pointer(src) {
                        0x1100_0000 | sf | (source << 5) | dst
                    } else {
                        0x2a00_03e0 | sf | (source << 16) | dst
                    }
                }
            }
            "add" | "sub" | "subs" | "cmp" | "cmn" => encode_add_sub(op, p)?,
            "and" | "orr" | "eor" | "ands" | "tst" => encode_logical(op, p)?,
            "mul" | "sdiv" | "udiv" => {
                let ([rd, rn, rm], wide) = gpr_regs::<3>(p)?;
                let base = match op {
                    "mul" => 0x1b00_7c00,
                    "sdiv" => 0x1ac0_0c00,
                    _ => 0x1ac0_0800,
                };
                base | (if wide { 0x8000_0000 } else { 0 }) | (rm << 16) | (rn << 5) | rd
            }
            "msub" => {
                let ([rd, rn, rm, ra], wide) = gpr_regs::<4>(p)?;
                return Ok(if wide {
                    encode_msub(rd as u8, rn as u8, rm as u8, ra as u8)
                } else {
                    encode_msub_w(rd as u8, rn as u8, rm as u8, ra as u8)
                });
            }
            "cset" => {
                let ([rd], wide) = gpr_regs::<1>(p)?;
                let cond = condition(arg(1)?)?;
                if cond >= 14 {
                    return Err(error(format!("Condition {} is invalid for cset", arg(1)?)));
                }
                0x1a9f_07e0 | (if wide { 0x8000_0000 } else { 0 }) | ((cond ^ 1) << 12) | rd
            }
            "csel" => {
                let ([rd, rn, rm], wide) = gpr_regs::<3>(p)?;
                0x1a80_0000
                    | (if wide { 0x8000_0000 } else { 0 })
                    | (rm << 16)
                    | (condition(arg(3)?)? << 12)
                    | (rn << 5)
                    | rd
            }
            "uxtb" => {
                let (rd, _) = gpr(arg(0)?)?;
                let (rn, source_wide) = gpr(arg(1)?)?;
                if source_wide || is_stack_pointer(arg(1)?) {
                    return Err(error("uxtb takes a 32-bit source register"));
                }
                0x5300_1c00 | (rn << 5) | rd
            }
            "sxtb" | "sxth" => {
                let (rd, rd_wide) = gpr(arg(0)?)?;
                let (rn, rn_wide) = gpr(arg(1)?)?;
                if rn_wide || is_stack_pointer(arg(0)?) || is_stack_pointer(arg(1)?) {
                    return Err(error(format!("{op} takes a 32-bit source register")));
                }
                let imms = if op == "sxtb" { 0x1c00 } else { 0x3c00 };
                let base = if rd_wide { 0x9340_0000 } else { 0x1300_0000 };
                base | imms | (rn << 5) | rd
            }
            "clrex" => {
                let crm = match p.first() {
                    Some(text) => immediate(text)?,
                    None => 15,
                };
                if !(0..=15).contains(&crm) {
                    return Err(error("clrex takes a barrier option from 0 to 15"));
                }
                0xd503_305f | ((crm as u32) << 8)
            }
            "sxtw" => {
                let (rd, rd_wide) = gpr(arg(0)?)?;
                let (rn, _) = gpr(arg(1)?)?;
                if !rd_wide || is_stack_pointer(arg(0)?) || is_stack_pointer(arg(1)?) {
                    return Err(error("sxtw takes a 64-bit destination register"));
                }
                0x9340_7c00 | (rn << 5) | rd
            }
            "lsl" | "lsr" | "asr" | "ror" if arg(2)?.starts_with('#') => {
                let ([rd, rn], wide) = gpr_regs::<2>(p)?;
                let size = if wide { 64 } else { 32 };
                let amount = immediate(arg(2)?)?;
                if !(0..size).contains(&amount) {
                    return Err(error(format!(
                        "Shift amount out of range for {op}: {amount}"
                    )));
                }
                let sf = if wide { 0x8000_0000u32 } else { 0 };
                let n = if wide { 0x0040_0000 } else { 0 };
                let amount = amount as u32;
                match op {
                    "ror" => {
                        let base = if wide { 0x93c0_0000u32 } else { 0x1380_0000 };
                        base | (rn << 16) | (amount << 10) | (rn << 5) | rd
                    }
                    "lsl" => {
                        let immr = (size as u32 - amount) % size as u32;
                        let imms = size as u32 - 1 - amount;
                        0x5300_0000 | sf | n | (immr << 16) | (imms << 10) | (rn << 5) | rd
                    }
                    _ => {
                        let base = if op == "asr" {
                            0x1300_0000
                        } else {
                            0x5300_0000
                        };
                        base | sf | n | (amount << 16) | ((size as u32 - 1) << 10) | (rn << 5) | rd
                    }
                }
            }
            "lsl" | "lsr" | "asr" | "ror" | "lslv" | "lsrv" | "asrv" | "rorv" => {
                let ([rd, rn, rm], wide) = gpr_regs::<3>(p)?;
                let kind = match op {
                    "lslv" | "lsl" => 0,
                    "lsrv" | "lsr" => 1,
                    "asrv" | "asr" => 2,
                    _ => 3,
                };
                0x1ac0_2000
                    | (if wide { 0x8000_0000 } else { 0 })
                    | (rm << 16)
                    | (kind << 10)
                    | (rn << 5)
                    | rd
            }
            "sbfx" | "ubfx" => {
                let (rd, rd_wide) = gpr(arg(0)?)?;
                let (rn, rn_wide) = gpr(arg(1)?)?;
                if rd_wide != rn_wide || is_stack_pointer(arg(0)?) || is_stack_pointer(arg(1)?) {
                    return Err(error(format!("Invalid register operand for {op}")));
                }
                let size = if rd_wide { 64 } else { 32 };
                let lsb = immediate(arg(2)?)?;
                let width = immediate(arg(3)?)?;
                if lsb < 0 || width < 1 || lsb + width > size {
                    return Err(error(format!("Bitfield out of range for {op}")));
                }
                let base = if op == "sbfx" {
                    0x1300_0000
                } else {
                    0x5300_0000
                };
                let n = if rd_wide { 0x8040_0000u32 } else { 0 };
                base | n
                    | ((lsb as u32) << 16)
                    | (((lsb + width - 1) as u32) << 10)
                    | (rn << 5)
                    | rd
            }
            "fmov" => {
                if is_stack_pointer(arg(0)?) || is_stack_pointer(arg(1)?) {
                    return Err(error("fmov cannot take the stack pointer"));
                }
                let base = match (arg(0)?.chars().next(), arg(1)?.chars().next()) {
                    (Some('d'), Some('x')) => 0x9e67_0000,
                    (Some('x'), Some('d')) => 0x9e66_0000,
                    (Some('s'), Some('w')) => 0x1e27_0000,
                    (Some('w'), Some('s')) => 0x1e26_0000,
                    (Some('d'), Some('d')) => 0x1e60_4000,
                    (Some('s'), Some('s')) => 0x1e20_4000,
                    _ => return Err(error("Invalid fmov register combination")),
                };
                base | (r(1)? << 5) | r(0)?
            }
            "fsqrt" => {
                let (rd, kind) = data_reg(arg(0)?)?;
                let (rn, kind_n) = data_reg(arg(1)?)?;
                if !matches!(kind, DataKind::S | DataKind::D) || kind_n != kind {
                    return Err(error("fsqrt takes two scalar registers of one size"));
                }
                let double = if kind == DataKind::D { 0x0040_0000 } else { 0 };
                0x1e21_c000 | double | (rn << 5) | rd
            }
            "fneg" => {
                let (rd, kind) = data_reg(arg(0)?)?;
                let (rn, kind_n) = data_reg(arg(1)?)?;
                if !matches!(kind, DataKind::S | DataKind::D) || kind_n != kind {
                    return Err(error("fneg takes two scalar registers of one size"));
                }
                let double = if kind == DataKind::D { 0x0040_0000 } else { 0 };
                0x1e21_4000 | double | (rn << 5) | rd
            }
            "fcmp" => {
                let (rn, kind) = data_reg(arg(0)?)?;
                let (rm, kind_m) = data_reg(arg(1)?)?;
                if !matches!(kind, DataKind::S | DataKind::D) || kind_m != kind {
                    return Err(error("fcmp takes two scalar registers of one size"));
                }
                let double = if kind == DataKind::D { 0x0040_0000 } else { 0 };
                0x1e20_2000 | double | (rm << 16) | (rn << 5)
            }
            "fcvt" => {
                let (rd, kind) = data_reg(arg(0)?)?;
                let (rn, kind_n) = data_reg(arg(1)?)?;
                match (kind, kind_n) {
                    (DataKind::D, DataKind::S) => 0x1e22_c000 | (rn << 5) | rd,
                    (DataKind::S, DataKind::D) => 0x1e62_4000 | (rn << 5) | rd,
                    _ => {
                        return Err(error(
                            "fcvt converts between a single and a double register",
                        ));
                    }
                }
            }
            "fmul" | "fdiv" | "fadd" | "fsub" => {
                let (rd, kind) = data_reg(arg(0)?)?;
                let (rn, kind_n) = data_reg(arg(1)?)?;
                let (rm, kind_m) = data_reg(arg(2)?)?;
                if !matches!(kind, DataKind::S | DataKind::D) || kind_n != kind || kind_m != kind {
                    return Err(error(format!(
                        "{op} takes three scalar registers of one size"
                    )));
                }
                let operation = match op {
                    "fmul" => 0x0800,
                    "fdiv" => 0x1800,
                    "fadd" => 0x2800,
                    _ => 0x3800,
                };
                (if kind == DataKind::D {
                    0x1e60_0000u32
                } else {
                    0x1e20_0000
                }) | operation
                    | (rm << 16)
                    | (rn << 5)
                    | rd
            }
            "scvtf" | "ucvtf" | "fcvtzs" | "fcvtzu" => {
                let to_float = op.starts_with("scvtf") || op.starts_with("ucvtf");
                let (fp, gp) = if to_float {
                    (arg(0)?, arg(1)?)
                } else {
                    (arg(1)?, arg(0)?)
                };
                let (fp_reg, fp_kind) = data_reg(fp)?;
                let (gp_reg, gp_wide) = gpr(gp)?;
                if is_stack_pointer(gp) || !matches!(fp_kind, DataKind::S | DataKind::D) {
                    return Err(error(format!(
                        "{op} takes a scalar float and a general register"
                    )));
                }
                let double = if fp_kind == DataKind::D {
                    0x0040_0000
                } else {
                    0
                };
                let (opcode, rn, rd) = match (op, to_float) {
                    ("scvtf", _) => (0x0002_0000, gp_reg, fp_reg),
                    ("ucvtf", _) => (0x0003_0000, gp_reg, fp_reg),
                    ("fcvtzs", _) => (0x0018_0000, fp_reg, gp_reg),
                    _ => (0x0019_0000, fp_reg, gp_reg),
                };
                0x1e20_0000 | sf_of(gp_wide) | double | opcode | (rn << 5) | rd
            }
            "stp" | "ldp" => {
                if p.len() < 3 {
                    return Err(error(format!("Missing address operand for {op}")));
                }
                let mem = p[2..].join(",");
                let pre = mem.ends_with('!');
                let post = mem.contains("],");
                let fields: Vec<_> = mem
                    .split(['[', ']', ',', '!'])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                let base = fields
                    .first()
                    .ok_or_else(|| error("Missing pair address"))?;
                let (rn, base_wide) = gpr(base)?;
                if !base_wide || *base == "xzr" {
                    return Err(error("Pair address must use an x register or sp"));
                }
                let offset = fields.get(1).map_or(Ok(0), |s| immediate(s))?;
                let (t1, kind) = data_reg(arg(0)?)?;
                let (t2, kind2) = data_reg(arg(1)?)?;
                if kind != kind2 {
                    return Err(error(format!("Mismatched pair register types for {op}")));
                }
                if op == "ldp" && t1 == t2 {
                    return Err(error("ldp needs two different transfer registers"));
                }
                let gpr_pair = matches!(kind, DataKind::W | DataKind::X);
                if gpr_pair && (pre || post) && rn != 31 && (t1 == rn || t2 == rn) {
                    return Err(error(format!("Unpredictable writeback for {op}")));
                }
                if wide && op == "stp" && pre {
                    enc_stp_pre_64(t1 as u8, t2 as u8, rn as u8, offset as i32)?
                } else if wide && op == "ldp" && post {
                    enc_ldp_post_64(t1 as u8, t2 as u8, rn as u8, offset as i32)?
                } else {
                    let (scale, width_bits) = match kind {
                        DataKind::W => (4, 0),
                        DataKind::X => (8, 0x8000_0000),
                        DataKind::S => (4, 0x0400_0000),
                        DataKind::D => (8, 0x4400_0000),
                    };
                    if offset % scale != 0 || !(-64..=63).contains(&(offset / scale)) {
                        return Err(error("Pair offset out of range"));
                    }
                    (if pre {
                        0x2980_0000
                    } else if post {
                        0x2880_0000
                    } else {
                        0x2900_0000
                    }) | width_bits
                        | if op == "ldp" { 1 << 22 } else { 0 }
                        | (((offset / scale) as u32 & 127) << 15)
                        | (t2 << 10)
                        | (rn << 5)
                        | t1
                }
            }
            "ldr" | "str" | "ldrb" | "strb" | "ldrh" | "strh" | "ldur" | "stur" | "ldurb"
            | "sturb" | "ldurh" | "sturh" | "ldrsb" | "ldrsh" | "ldrsw" | "ldursb" | "ldursh"
            | "ldursw" => {
                let mem = p.get(1..).unwrap_or_default().join(",");
                let (inside, post_offset) = match mem.split_once("],") {
                    Some((address, offset)) => (format!("{address}]"), Some(offset.trim())),
                    None => (mem.trim().trim_end_matches('!').to_string(), None),
                };
                let pre = mem.trim().ends_with("]!");
                let post = post_offset.is_some();
                let inside = inside
                    .trim()
                    .strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .ok_or_else(|| error("Expected base and immediate address"))?;
                let fields: Vec<_> = inside.split(',').map(str::trim).collect();
                if fields.len() > 2 {
                    return Err(error(format!("Too many address operands for {op}")));
                }
                let (rn, base_wide) = gpr(fields[0])?;
                if !base_wide || fields[0] == "xzr" {
                    return Err(error("Address base must be an x register or sp"));
                }
                let (rt, kind) = data_reg(arg(0)?)?;
                let signed = matches!(
                    op,
                    "ldrsb" | "ldrsh" | "ldrsw" | "ldursb" | "ldursh" | "ldursw"
                );
                let narrow = !signed && (op.ends_with('b') || op.ends_with('h'));
                let (size, fp) = if signed {
                    let size = match op.as_bytes()[op.len() - 1] {
                        b'b' => 0,
                        b'h' => 1,
                        _ => 2,
                    };
                    match kind {
                        DataKind::X => (size, false),
                        DataKind::W if size != 2 => (size, false),
                        _ => {
                            return Err(error(format!(
                                "{op} takes a general purpose register of the right width"
                            )));
                        }
                    }
                } else {
                    match (narrow, kind) {
                        (true, DataKind::W) => (if op.ends_with('b') { 0 } else { 1 }, false),
                        (true, _) => {
                            return Err(error(format!("{op} takes a 32-bit register")));
                        }
                        (false, DataKind::W) => (2, false),
                        (false, DataKind::X) => (3, false),
                        (false, DataKind::S) => (2, true),
                        (false, DataKind::D) => (3, true),
                    }
                };
                let offset = match post_offset {
                    Some(text) => immediate(text)?,
                    None => fields.get(1).map_or(Ok(0), |s| immediate(s))?,
                };
                let writeback = pre || post;
                if writeback && op.contains("ur") {
                    return Err(error(format!("{op} has no writeback form")));
                }
                if post && fields.len() > 1 {
                    return Err(error(format!("Post-indexed {op} takes no inner offset")));
                }
                if writeback && matches!(kind, DataKind::W | DataKind::X) && rn == rt && rn != 31 {
                    return Err(error(format!("Unpredictable writeback for {op}")));
                }
                let load = op.starts_with('l');
                let scaled = !writeback
                    && !op.contains("ur")
                    && offset >= 0
                    && offset % (1 << size) == 0
                    && offset / (1 << size) <= 4095;
                let opc: u32 = if signed {
                    if kind == DataKind::X { 2 } else { 3 }
                } else if load {
                    1
                } else {
                    0
                };
                let base = ((size as u32) << 30)
                    | if scaled { 0x3900_0000 } else { 0x3800_0000 }
                    | if fp { 0x0400_0000 } else { 0 }
                    | (opc << 22)
                    | if pre {
                        0x0C00
                    } else if post {
                        0x0400
                    } else {
                        0
                    };
                let imm = if scaled {
                    ((offset >> size) as u32) << 10
                } else {
                    if !(-256..=255).contains(&offset) {
                        return Err(error(format!("Load/store offset out of range: {offset}")));
                    }
                    ((offset as u32) & 511) << 12
                };
                base | imm | (rn << 5) | rt
            }
            "ldxr" | "ldxrb" | "ldxrh" | "ldaxr" | "ldaxrb" | "ldaxrh" => {
                let (rt, size) = exclusive_data(op, arg(0)?)?;
                let base = if op.starts_with("ldax") {
                    0x085f_fc00
                } else {
                    0x085f_7c00
                };
                base | size | (exclusive_base(arg(1)?)? << 5) | rt
            }
            "stxr" | "stxrb" | "stxrh" | "stlxr" | "stlxrb" | "stlxrh" => {
                let (rs, status_wide) = gpr(arg(0)?)?;
                if status_wide {
                    return Err(error(format!("{op} status register must be a w register")));
                }
                let (rt, size) = exclusive_data(op, arg(1)?)?;
                let base = if op.starts_with("stl") {
                    0x0800_fc00
                } else {
                    0x0800_7c00
                };
                base | size | (rs << 16) | (exclusive_base(arg(2)?)? << 5) | rt
            }
            "ldar" | "ldarb" | "ldarh" => {
                let (rt, size) = exclusive_data(op, arg(0)?)?;
                0x08df_fc00 | size | (exclusive_base(arg(1)?)? << 5) | rt
            }
            "stlr" | "stlrb" | "stlrh" => {
                let (rt, size) = exclusive_data(op, arg(0)?)?;
                0x089f_fc00 | size | (exclusive_base(arg(1)?)? << 5) | rt
            }
            "rbit" | "rev" | "rev16" | "clz" => {
                let ([rd, rn], wide) = gpr_regs::<2>(p)?;
                let base = match (op, wide) {
                    ("rbit", false) => 0x5ac0_0000,
                    ("rbit", true) => 0xdac0_0000,
                    ("rev16", false) => 0x5ac0_0400,
                    ("rev16", true) => 0xdac0_0400,
                    ("rev", false) => 0x5ac0_0800,
                    ("rev", true) => 0xdac0_0c00,
                    ("clz", false) => 0x5ac0_1000,
                    _ => 0xdac0_1000,
                };
                base | (rn << 5) | rd
            }
            "neg" => {
                let ([rd, rm], wide) = gpr_regs::<2>(p)?;
                let base = if wide { 0xcb00_03e0 } else { 0x4b00_03e0 };
                base | (rm << 16) | rd
            }
            "cnt" => {
                let (rd, rn) = (vector_8b(arg(0)?)?, vector_8b(arg(1)?)?);
                0x0e20_5800 | (rn << 5) | rd
            }
            "addv" => {
                let rd = scalar_b(arg(0)?)?;
                let rn = vector_8b(arg(1)?)?;
                0x0e31_b800 | (rn << 5) | rd
            }
            _ => return Err(error(format!("Unsupported instruction: {op}"))),
        };
        Ok(word.to_le_bytes().to_vec())
    }
}

impl InstructionEncoder for AArch64Encoder {
    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let bytes = self.encode(inst)?;
        self.position += bytes.len();
        Ok(bytes)
    }
    fn current_position(&self) -> usize {
        self.position
    }
}

#[cfg(test)]
mod exclusive_tests {
    use super::*;

    fn word(op: &str, operands: &[&str]) -> u32 {
        let inst = ParsedInstruction {
            opcode: op.to_string(),
            operands: operands.iter().map(|s| s.to_string()).collect(),
        };
        let bytes = AArch64Encoder::new().encode_instruction(&inst).unwrap();
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[test]
    fn exclusive_and_ordered_access_match_arm_arm() {
        assert_eq!(word("ldxr", &["x0", "[x1]"]), 0xC85F_7C20);
        assert_eq!(word("stlxr", &["w2", "x3", "[x4]"]), 0xC802_FC83);
        assert_eq!(word("ldaxrb", &["w5", "[sp]"]), 0x085F_FFE5);
        assert_eq!(word("ldarh", &["w0", "[x0]"]), 0x48DF_FC00);
        assert_eq!(word("stlr", &["x1", "[x2]"]), 0xC89F_FC41);
        assert_eq!(word("stxrb", &["w1", "w2", "[x3]"]), 0x0801_7C62);
    }

    /// Each word is derived from the ARM ARM (DDI 0487) fixed fields of the
    /// instruction class named in the comment.
    #[test]
    fn subs_sxth_clrex_and_scalar_float_match_arm_arm() {
        assert_eq!(word("subs", &["x0", "x0", "#1"]), 0xF100_0400);
        assert_eq!(word("subs", &["w3", "w4", "#7"]), 0x7100_1C83);
        assert_eq!(word("sxth", &["x0", "w0"]), 0x9340_3C00);
        assert_eq!(word("sxth", &["w1", "w2"]), 0x1300_3C41);
        assert_eq!(word("sxtb", &["x0", "w0"]), 0x9340_1C00);
        assert_eq!(word("sxtb", &["w1", "w2"]), 0x1300_1C41);
        assert_eq!(word("cmp", &["w9", "w12", "uxtb"]), 0x6B2C_013F);
        assert_eq!(word("fsqrt", &["d0", "d1"]), 0x1E61_C020);
        assert_eq!(word("clrex", &[]), 0xD503_3F5F);
        assert_eq!(word("clrex", &["#0"]), 0xD503_305F);
        assert_eq!(word("fcmp", &["d0", "d1"]), 0x1E61_2000);
        assert_eq!(word("fcmp", &["s2", "s3"]), 0x1E23_2040);
        assert_eq!(word("fneg", &["d0", "d0"]), 0x1E61_4000);
        assert_eq!(word("fneg", &["s0", "s1"]), 0x1E21_4020);
        assert_eq!(word("fcvt", &["d0", "s0"]), 0x1E22_C000);
        assert_eq!(word("fcvt", &["s0", "d0"]), 0x1E62_4000);
    }

    #[test]
    fn bit_count_and_reverse_match_arm_arm() {
        assert_eq!(word("rbit", &["x0", "x1"]), 0xDAC0_0020);
        assert_eq!(word("rev", &["w0", "w1"]), 0x5AC0_0820);
        assert_eq!(word("clz", &["w2", "w3"]), 0x5AC0_1062);
        assert_eq!(word("neg", &["x0", "x1"]), 0xCB01_03E0);
        assert_eq!(word("cnt", &["v0.8b", "v1.8b"]), 0x0E20_5820);
        assert_eq!(word("addv", &["b0", "v1.8b"]), 0x0E31_B820);
    }
}
