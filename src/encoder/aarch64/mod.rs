pub mod fixed;
pub mod ldst_imm64;
mod simd;

pub(crate) use simd::normalize_dotted;
mod sysreg;

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
            let fp = reg.starts_with(['s', 'd', 'q']);
            reg.strip_prefix(['x', 'w', 's', 'd', 'q'])
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
    Q,
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
        Some(b'q') => DataKind::Q,
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
        "cmn" | "adds" => 0x2000_0000,
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
    let flag_setting = matches!(op, "ands" | "bics" | "tst");
    let rn_index = usize::from(op != "tst");
    let (min, max) = if op == "tst" { (2, 3) } else { (3, 4) };
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
        "bic" | "bics" => (0, 0x0a20_0000),
        "orn" => (0, 0x2a20_0000),
        "eon" => (0, 0x4a20_0000),
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
        if matches!(op, "bic" | "bics" | "orn" | "eon") {
            return Err(error(format!("{op} has no immediate form")));
        }
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
    let (kind, amount) = shifted_register(p.get(rn_index + 2).map(|s| s.trim()), wide)?;
    Ok(base_reg | flags_reg | (kind << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd)
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

/// The condition for an alias that inverts its operand, which is never AL or NV.
fn inverted_condition(text: &str) -> Result<u32, RasError> {
    let cond = condition(text)?;
    if cond >= 14 {
        return Err(error(format!("Condition {text} is invalid here")));
    }
    Ok(cond ^ 1)
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

/// The 8-bit form of an `fmov` constant: sign, low 3 exponent bits, top 4 mantissa bits.
fn fmov_imm8(text: &str) -> Result<u32, RasError> {
    let value: f64 = text
        .parse()
        .map_err(|_| error(format!("Invalid fmov constant: {text}")))?;
    let bits = value.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    if bits & 0x0000_ffff_ffff_ffff != 0 || !(0x3fc..=0x403).contains(&exponent) {
        return Err(error(format!("fmov cannot encode constant: {text}")));
    }
    Ok((((bits >> 63) << 7) | ((exponent & 7) << 4) | ((bits >> 48) & 0xf)) as u32)
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
        "mov" | "uxtb" | "uxth" | "sxtw" | "fmov" | "cset" | "csetm" | "scvtf" | "ucvtf"
        | "fcvtzs" | "fcvtzu" | "mrs" | "msr" | "ngc" | "ngcs" => 2,
        "movz" | "movk" | "mul" | "smulh" | "umulh" | "sdiv" | "udiv" | "fmul" | "fdiv"
        | "fadd" | "fsub" | "lsl" | "lsr" | "asr" | "ror" | "lslv" | "lsrv" | "asrv" | "rorv"
        | "smull" | "umull" | "mvn" | "cinc" | "cinv" | "cneg" => 3,
        "csel" | "csinc" | "csinv" | "csneg" | "msub" | "madd" | "sbfx" | "ubfx" | "sbfiz"
        | "ubfiz" | "bfi" | "bfxil" | "extr" | "smaddl" | "umaddl" => 4,
        _ => return None,
    })
}

impl AArch64Encoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn encode(&self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        if let Some(standard) = simd::normalize_dotted(inst)? {
            return self.encode(&standard);
        }
        let op = inst.opcode.as_str();
        let p = &inst.operands;
        if max_operands(op).is_some_and(|max| p.len() > max) {
            return Err(error(format!("Too many operands for {op}")));
        }
        if let Some(word) = simd::encode(op, p)? {
            return Ok(word.to_le_bytes().to_vec());
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
            "mrs" => sysreg::encode_mrs(p)?,
            "msr" => sysreg::encode_msr(p)?,
            "add" | "sub" | "subs" | "adds" | "cmp" | "cmn" => encode_add_sub(op, p)?,
            "and" | "orr" | "eor" | "ands" | "tst" | "bic" | "bics" | "orn" | "eon" => {
                encode_logical(op, p)?
            }
            "mvn" => {
                let ([rd, rm], wide) = gpr_regs::<2>(p)?;
                let (kind, amount) = shifted_register(p.get(2).map(|s| s.trim()), wide)?;
                0x2a20_03e0 | sf_of(wide) | (kind << 22) | (rm << 16) | (amount << 10) | rd
            }
            "adc" | "adcs" | "sbc" | "sbcs" => {
                let ([rd, rn, rm], wide) = gpr_regs::<3>(p)?;
                let base = match op {
                    "adc" => 0x1a00_0000,
                    "adcs" => 0x3a00_0000,
                    "sbc" => 0x5a00_0000,
                    _ => 0x7a00_0000,
                };
                base | sf_of(wide) | (rm << 16) | (rn << 5) | rd
            }
            "ngc" | "ngcs" => {
                let ([rd, rm], wide) = gpr_regs::<2>(p)?;
                let base = if op == "ngc" {
                    0x5a00_03e0
                } else {
                    0x7a00_03e0
                };
                base | sf_of(wide) | (rm << 16) | rd
            }
            "madd" => {
                let ([rd, rn, rm, ra], wide) = gpr_regs::<4>(p)?;
                let base = if wide { 0x9b00_0000 } else { 0x1b00_0000 };
                base | (rm << 16) | (ra << 10) | (rn << 5) | rd
            }
            "smaddl" | "umaddl" | "smull" | "umull" => {
                let (rd, rd_wide) = gpr(arg(0)?)?;
                let (rn, rn_wide) = gpr(arg(1)?)?;
                let (rm, rm_wide) = gpr(arg(2)?)?;
                let narrow_ok = [arg(0)?, arg(1)?, arg(2)?]
                    .iter()
                    .all(|reg| !is_stack_pointer(reg));
                if !rd_wide || rn_wide || rm_wide || !narrow_ok {
                    return Err(error(format!("{op} takes an x destination and w sources")));
                }
                let ra = if op.ends_with("addl") {
                    let (ra, ra_wide) = gpr(arg(3)?)?;
                    if !ra_wide || is_stack_pointer(arg(3)?) {
                        return Err(error(format!("{op} takes an x accumulator")));
                    }
                    ra
                } else {
                    31
                };
                let base = if op.starts_with('s') {
                    0x9b20_0000
                } else {
                    0x9ba0_0000
                };
                base | (rm << 16) | (ra << 10) | (rn << 5) | rd
            }
            "mul" | "sdiv" | "udiv" => {
                let ([rd, rn, rm], wide) = gpr_regs::<3>(p)?;
                let base = match op {
                    "mul" => 0x1b00_7c00,
                    "sdiv" => 0x1ac0_0c00,
                    _ => 0x1ac0_0800,
                };
                base | (if wide { 0x8000_0000 } else { 0 }) | (rm << 16) | (rn << 5) | rd
            }
            "smulh" | "umulh" => {
                let ([rd, rn, rm], wide) = gpr_regs::<3>(p)?;
                if !wide {
                    return Err(error(format!("{op} requires 64-bit registers")));
                }
                let base = if op == "smulh" {
                    0x9b40_7c00
                } else {
                    0x9bc0_7c00
                };
                base | (rm << 16) | (rn << 5) | rd
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
            "cinc" | "cinv" | "cneg" => {
                let ([rd, rn], wide) = gpr_regs::<2>(p)?;
                let base = match op {
                    "cinc" => 0x1a80_0400,
                    "cinv" => 0x5a80_0000,
                    _ => 0x5a80_0400,
                };
                base | sf_of(wide)
                    | (rn << 16)
                    | (inverted_condition(arg(2)?)? << 12)
                    | (rn << 5)
                    | rd
            }
            "csetm" => {
                let ([rd], wide) = gpr_regs::<1>(p)?;
                0x5a80_0000
                    | sf_of(wide)
                    | (31 << 16)
                    | (inverted_condition(arg(1)?)? << 12)
                    | (31 << 5)
                    | rd
            }
            "csel" | "csinc" | "csinv" | "csneg" => {
                let ([rd, rn, rm], wide) = gpr_regs::<3>(p)?;
                let base = match op {
                    "csel" => 0x1a80_0000,
                    "csinc" => 0x1a80_0400,
                    "csinv" => 0x5a80_0000,
                    _ => 0x5a80_0400,
                };
                base | (if wide { 0x8000_0000 } else { 0 })
                    | (rm << 16)
                    | (condition(arg(3)?)? << 12)
                    | (rn << 5)
                    | rd
            }
            "uxth" => {
                let (rd, rd_wide) = gpr(arg(0)?)?;
                let (rn, rn_wide) = gpr(arg(1)?)?;
                if rd_wide || rn_wide || is_stack_pointer(arg(0)?) || is_stack_pointer(arg(1)?) {
                    return Err(error("uxth takes 32-bit registers"));
                }
                0x5300_3c00 | (rn << 5) | rd
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
            "sbfx" | "ubfx" | "sbfiz" | "ubfiz" | "bfi" | "bfxil" => {
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
                let insert = matches!(op, "sbfiz" | "ubfiz" | "bfi");
                let (immr, imms) = if insert {
                    ((size - lsb) % size, width - 1)
                } else {
                    (lsb, lsb + width - 1)
                };
                let base = match op {
                    "sbfx" | "sbfiz" => 0x1300_0000,
                    "ubfx" | "ubfiz" => 0x5300_0000,
                    _ => 0x3300_0000,
                };
                let n = if rd_wide { 0x8040_0000u32 } else { 0 };
                base | n | ((immr as u32) << 16) | ((imms as u32) << 10) | (rn << 5) | rd
            }
            "extr" => {
                let ([rd, rn, rm], wide) = gpr_regs::<3>(p)?;
                let size = if wide { 64 } else { 32 };
                let lsb = immediate(arg(3)?)?;
                if !(0..size).contains(&lsb) {
                    return Err(error(format!("extr lsb out of range: {lsb}")));
                }
                let base = if wide { 0x93c0_0000 } else { 0x1380_0000 };
                base | (rm << 16) | ((lsb as u32) << 10) | (rn << 5) | rd
            }
            "fmov" => {
                if is_stack_pointer(arg(0)?) || is_stack_pointer(arg(1)?) {
                    return Err(error("fmov cannot take the stack pointer"));
                }
                if let Some(constant) = arg(1)?.strip_prefix('#') {
                    let (rd, kind) = data_reg(arg(0)?)?;
                    let base = match kind {
                        DataKind::S => 0x1e20_1000,
                        DataKind::D => 0x1e60_1000,
                        _ => return Err(error("fmov constant takes s or d register")),
                    };
                    return Ok((base | (fmov_imm8(constant)? << 13) | rd)
                        .to_le_bytes()
                        .to_vec());
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
                        DataKind::Q => (16, 0x8400_0000),
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
                let index = fields
                    .get(1)
                    .filter(|field| **field != "sp" && gpr(field).is_ok());
                if fields.len() > if index.is_some() { 3 } else { 2 } {
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
                        (false, DataKind::Q) => (0, true),
                    }
                };
                let scale = if kind == DataKind::Q { 4 } else { size };
                let offset = match (post_offset, index) {
                    (Some(text), _) => immediate(text)?,
                    (None, Some(_)) => 0,
                    (None, None) => fields.get(1).map_or(Ok(0), |s| immediate(s))?,
                };
                let writeback = pre || post;
                if index.is_some() && (writeback || op.contains("ur")) {
                    return Err(error(format!("{op} has no register offset form here")));
                }
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
                    && offset % (1 << scale) == 0
                    && offset / (1 << scale) <= 4095;
                let opc: u32 = if signed {
                    if kind == DataKind::X { 2 } else { 3 }
                } else if kind == DataKind::Q {
                    if load { 3 } else { 2 }
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
                    ((offset >> scale) as u32) << 10
                } else {
                    if !(-256..=255).contains(&offset) {
                        return Err(error(format!("Load/store offset out of range: {offset}")));
                    }
                    ((offset as u32) & 511) << 12
                };
                if let Some(index) = index {
                    let (rm, index_wide) = gpr(index)?;
                    let (extend, amount) = match fields.get(2) {
                        Some(field) => field
                            .split_once(char::is_whitespace)
                            .map_or((*field, "#0"), |(name, amount)| (name, amount.trim())),
                        None => ("lsl", "#0"),
                    };
                    let option: u32 = match (extend, index_wide) {
                        ("lsl", true) => 0b011,
                        ("sxtx", true) => 0b111,
                        ("uxtw", false) => 0b010,
                        ("sxtw", false) => 0b110,
                        _ => return Err(error(format!("Bad index extend for {op}: {extend}"))),
                    };
                    let amount = immediate(amount)?;
                    if amount != 0 && amount != i64::from(scale) {
                        return Err(error(format!("Bad index shift for {op}: {amount}")));
                    }
                    let shifted = u32::from(amount != 0) << 12;
                    let register_form = ((size as u32) << 30)
                        | 0x3820_0800
                        | if fp { 0x0400_0000 } else { 0 }
                        | (opc << 22);
                    register_form | (rm << 16) | (option << 13) | shifted | (rn << 5) | rt
                } else {
                    base | imm | (rn << 5) | rt
                }
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
    fn high_multiply_matches_arm_arm() {
        assert_eq!(word("smulh", &["x0", "x1", "x2"]), 0x9B42_7C20);
        assert_eq!(word("umulh", &["x3", "x4", "x5"]), 0x9BC5_7C83);
        let narrow = ParsedInstruction {
            opcode: "smulh".to_string(),
            operands: vec!["w0".to_string(), "w1".to_string(), "w2".to_string()],
        };
        assert!(AArch64Encoder::new().encode_instruction(&narrow).is_err());
    }

    #[test]
    fn csinv_and_quad_register_access_match_arm_arm() {
        assert_eq!(word("csinv", &["w0", "w1", "w2", "eq"]), 0x5A82_0020);
        assert_eq!(word("csinv", &["x0", "x0", "xzr", "cc"]), 0xDA9F_3000);
        assert_eq!(word("ldr", &["q0", "[x9]"]), 0x3DC0_0120);
        assert_eq!(word("str", &["q0", "[x9]"]), 0x3D80_0120);
        assert_eq!(word("ldr", &["q1", "[x10, #16]"]), 0x3DC0_0541);
        assert_eq!(word("stp", &["q0", "q1", "[sp, #16]"]), 0xAD00_87E0);
    }

    #[test]
    fn simd_lane_moves_match_arm_arm() {
        assert_eq!(word("dup", &["v0.4s", "w12"]), 0x4E04_0D80);
        assert_eq!(word("dup", &["v1.16b", "w0"]), 0x4E01_0C01);
        assert_eq!(word("dup", &["v2.2d", "x3"]), 0x4E08_0C62);
        assert_eq!(word("ins", &["v0.s[1]", "w10"]), 0x4E0C_1D40);
        assert_eq!(word("ins", &["v0.s[3]", "w9"]), 0x4E1C_1D20);
        assert_eq!(word("ins", &["v2.d[1]", "x3"]), 0x4E18_1C62);
        assert_eq!(word("ins", &["v0.b[15]", "w1"]), 0x4E1F_1C20);
        assert_eq!(word("umov", &["w11", "v0.s[0]"]), 0x0E04_3C0B);
        assert_eq!(word("umov", &["w12", "v0.s[3]"]), 0x0E1C_3C0C);
        assert_eq!(word("umov", &["x5", "v1.d[1]"]), 0x4E18_3C25);
        assert_eq!(word("umov", &["w9", "v0.b[0]"]), 0x0E01_3C09);
    }

    #[test]
    fn simd_arithmetic_matches_arm_arm() {
        assert_eq!(word("add", &["v0.4s", "v0.4s", "v1.4s"]), 0x4EA1_8400);
        assert_eq!(word("sub", &["v2.2d", "v3.2d", "v4.2d"]), 0x6EE4_8462);
        assert_eq!(word("mul", &["v0.8h", "v1.8h", "v2.8h"]), 0x4E62_9C20);
        assert_eq!(word("and", &["v0.16b", "v1.16b", "v2.16b"]), 0x4E22_1C20);
        assert_eq!(word("eor", &["v3.8b", "v4.8b", "v5.8b"]), 0x2E25_1C83);
        assert_eq!(word("fmul", &["v0.4s", "v0.4s", "v1.4s"]), 0x6E21_DC00);
        assert_eq!(word("fmul", &["v2.2d", "v3.2d", "v4.2d"]), 0x6E64_DC62);
        assert_eq!(word("orr", &["v0.16b", "v1.16b", "v2.16b"]), 0x4EA2_1C20);
        assert_eq!(word("orr", &["v3.8b", "v4.8b", "v5.8b"]), 0x0EA5_1C83);
        assert_eq!(word("mov", &["v0.16b", "v1.16b"]), 0x4EA1_1C20);
        assert_eq!(word("fadd", &["v0.4s", "v1.4s", "v2.4s"]), 0x4E22_D420);
        assert_eq!(word("fadd", &["v0.2s", "v1.2s", "v2.2s"]), 0x0E22_D420);
        assert_eq!(word("fadd", &["v0.2d", "v1.2d", "v2.2d"]), 0x4E62_D420);
        assert_eq!(word("fsub", &["v0.4s", "v1.4s", "v2.4s"]), 0x4EA2_D420);
        assert_eq!(word("fsub", &["v0.2d", "v1.2d", "v2.2d"]), 0x4EE2_D420);
        assert_eq!(word("fdiv", &["v0.4s", "v1.4s", "v2.4s"]), 0x6E22_FC20);
        assert_eq!(word("fdiv", &["v0.2s", "v1.2s", "v2.2s"]), 0x2E22_FC20);
        assert_eq!(word("fdiv", &["v0.2d", "v1.2d", "v2.2d"]), 0x6E62_FC20);
        assert_eq!(word("fmla", &["v0.4s", "v1.4s", "v2.4s"]), 0x4E22_CC20);
        assert_eq!(word("fmla", &["v0.2s", "v1.2s", "v2.2s"]), 0x0E22_CC20);
        assert_eq!(word("fmla", &["v0.2d", "v1.2d", "v2.2d"]), 0x4E62_CC20);
        assert_eq!(word("fmls", &["v0.4s", "v1.4s", "v2.4s"]), 0x4EA2_CC20);
        assert_eq!(word("fmls", &["v0.2s", "v1.2s", "v2.2s"]), 0x0EA2_CC20);
        assert_eq!(word("fmls", &["v0.2d", "v1.2d", "v2.2d"]), 0x4EE2_CC20);
        assert_eq!(word("faddp", &["v0.4s", "v1.4s", "v2.4s"]), 0x6E22_D420);
        assert_eq!(word("faddp", &["v0.2s", "v1.2s", "v2.2s"]), 0x2E22_D420);
        assert_eq!(word("faddp", &["v0.2d", "v1.2d", "v2.2d"]), 0x6E62_D420);
        assert_eq!(word("faddp", &["s0", "v1.2s"]), 0x7E30_D820);
        assert_eq!(word("faddp", &["d0", "v1.2d"]), 0x7E70_D820);
        assert_eq!(
            word("ext", &["v0.16b", "v1.16b", "v2.16b", "#8"]),
            0x6E02_4020
        );
        assert_eq!(word("ext", &["v0.8b", "v1.8b", "v2.8b", "#3"]), 0x2E02_1820);
        assert_eq!(
            word("ext", &["v0.16b", "v1.16b", "v2.16b", "#0"]),
            0x6E02_0020
        );
        assert_eq!(word("fmov", &["x0", "v1.d[1]"]), 0x9EAE_0020);
        assert_eq!(word("fmov", &["v1.d[1]", "x0"]), 0x9EAF_0001);
        assert_eq!(word("movi", &["v0.2d", "#0"]), 0x6F00_E400);
        assert_eq!(word("movi", &["v0.16b", "#0"]), 0x4F00_E400);
        assert_eq!(word("movi", &["v0.8b", "#0xff"]), 0x0F07_E7E0);
        assert_eq!(word("movi", &["v1.4s", "#1"]), 0x4F00_0421);
        assert_eq!(word("movi", &["v1.4s", "#0x1", "lsl #8"]), 0x4F00_2421);
        assert_eq!(word("movi", &["v1.8h", "#0x2", "lsl #8"]), 0x4F00_A441);
        assert_eq!(word("movi", &["v1.4h", "#0x2"]), 0x0F00_8441);
        assert_eq!(word("movi", &["v1.2s", "#0x3"]), 0x0F00_0461);
        assert_eq!(word("movi", &["v1.2d", "#0xff00ff00ff00ff00"]), 0x6F05_E541);
        assert_eq!(word("movi", &["v1.2d", "#0xffffffffffffffff"]), 0x6F07_E7E1);
        assert_eq!(word("movi", &["v1.16b", "#0x42"]), 0x4F02_E441);
        assert_eq!(
            word("movi.4s", &["v1", "#1"]),
            word("movi", &["v1.4s", "#1"])
        );
        assert_eq!(word("ld1", &["{ v0.4s }", "[x0]"]), 0x4C40_7800);
        assert_eq!(word("ld1", &["{ v0.4s, v1.4s }", "[x0]"]), 0x4C40_A800);
        assert_eq!(
            word("ld1", &["{ v0.4s, v1.4s, v2.4s }", "[x0]"]),
            0x4C40_6800
        );
        assert_eq!(
            word("ld1", &["{ v0.4s, v1.4s, v2.4s, v3.4s }", "[x0]"]),
            0x4C40_2800
        );
        assert_eq!(word("ld1", &["{ v0.2d }", "[x0]"]), 0x4C40_7C00);
        assert_eq!(word("ld1", &["{ v0.8b }", "[x0]"]), 0x0C40_7000);
        assert_eq!(word("st1", &["{ v0.2s }", "[x0]"]), 0x0C00_7800);
        assert_eq!(word("st1", &["{ v31.4h }", "[sp]"]), 0x0C00_77FF);
        assert_eq!(word("st1", &["{ v0.2d, v1.2d }", "[x0]"]), 0x4C00_AC00);
        assert_eq!(
            word("ld1.4s", &["{ v0 }", "[x0]"]),
            word("ld1", &["{ v0.4s }", "[x0]"])
        );
        assert_eq!(
            word("fmov.d", &["x0", "v1[1]"]),
            word("fmov", &["x0", "v1.d[1]"])
        );
        assert_eq!(
            word("faddp.2s", &["s0", "v1"]),
            word("faddp", &["s0", "v1.2s"])
        );
    }

    #[test]
    fn dotted_neon_spelling_encodes_like_the_standard_form() {
        assert_eq!(word("add.4s", &["v0", "v1", "v2"]), 0x4EA2_8420);
        assert_eq!(
            word("add.4s", &["v0", "v1", "v2"]),
            word("add", &["v0.4s", "v1.4s", "v2.4s"])
        );
        assert_eq!(
            word("mul.8h", &["v0", "v1", "v2"]),
            word("mul", &["v0.8h", "v1.8h", "v2.8h"])
        );
        assert_eq!(word("eor.8b", &["v3", "v4", "v5"]), 0x2E25_1C83);
        assert_eq!(word("fmul.2d", &["v2", "v3", "v4"]), 0x6E64_DC62);
        assert_eq!(word("dup.4s", &["v0", "w12"]), 0x4E04_0D80);
        assert_eq!(word("dup.2d", &["v2", "x3"]), 0x4E08_0C62);
        assert_eq!(word("dup", &["v0.4s", "v1.s[2]"]), 0x4E14_0420);
        assert_eq!(word("dup", &["v0.2d", "v1.d[1]"]), 0x4E18_0420);
        assert_eq!(word("dup", &["v0.8b", "v1.b[7]"]), 0x0E0F_0420);
        assert_eq!(word("dup", &["v0.16b", "v1.b[15]"]), 0x4E1F_0420);
        assert_eq!(word("dup", &["v0.4h", "v1.h[0]"]), 0x0E02_0420);
        assert_eq!(
            word("dup.2s", &["v0", "v1[1]"]),
            word("dup", &["v0.2s", "v1.s[1]"])
        );
        assert_eq!(word("mov.s", &["w12", "v0[3]"]), 0x0E1C_3C0C);
        assert_eq!(word("mov.s", &["v0[1]", "w10"]), 0x4E0C_1D40);
        assert_eq!(word("mov", &["w11", "v0.s[0]"]), 0x0E04_3C0B);
    }

    #[test]
    fn dotted_spelling_is_refused_when_it_is_not_a_vector_form() {
        let refused = [
            ("add.4s", &["v0.4s", "v1", "v2"][..]),
            ("add.4s", &["x0", "x1", "x2"]),
            ("ld1.s", &["{ v0 }", "[x0]"]),
            ("ins.s", &["x0", "w1"]),
            ("fmul.2s", &["s0", "s1", "s2"]),
            ("add.3s", &["v0", "v1", "v2"]),
        ];
        for (op, operands) in refused {
            let inst = ParsedInstruction {
                opcode: op.to_string(),
                operands: operands.iter().map(|s| s.to_string()).collect(),
            };
            assert!(
                AArch64Encoder::new().encode_instruction(&inst).is_err(),
                "{op} {operands:?}"
            );
        }
        let condition = ParsedInstruction {
            opcode: "b.eq".into(),
            operands: vec!["8".into()],
        };
        assert!(normalize_dotted(&condition).unwrap().is_none());
    }

    #[test]
    fn simd_rejects_unencodable_shapes() {
        let refused = [
            ("dup", &["v0.4s", "x1"][..]),
            ("dup", &["v0.2d", "w1"]),
            ("dup", &["v0.4s", "sp"]),
            ("ins", &["v0.s[4]", "w0"]),
            ("ins", &["v0.d[0]", "w0"]),
            ("ins", &["v0.s[0]", "v1.s[0]"]),
            ("umov", &["w0", "v0.d[0]"]),
            ("umov", &["x0", "v0.s[0]"]),
            ("add", &["v0.4s", "v1.2s", "v2.4s"]),
            ("mul", &["v0.2d", "v1.2d", "v2.2d"]),
            ("and", &["v0.8h", "v1.8h", "v2.8h"]),
            ("fmul", &["v0.8b", "v1.8b", "v2.8b"]),
            ("orr", &["v0.4s", "v1.4s", "v2.4s"]),
            ("fadd", &["v0.8h", "v1.8h", "v2.8h"]),
            ("fdiv", &["v0.16b", "v1.16b", "v2.16b"]),
            ("fmla", &["v0.8h", "v1.8h", "v2.8h"]),
            ("fmls", &["v0.4s", "v1.4s", "v2.2s"]),
            ("mov", &["v0.4s", "v1.2s"]),
            ("mov", &["v0.4s", "v1.4s", "v2.4s"]),
            ("add", &["v0.4s", "v1.4s"]),
            ("addv", &["b0", "v1.2s"]),
            ("addv", &["d0", "v1.2d"]),
            ("addv", &["h0", "v1.8b"]),
            ("addv", &["s0", "v1.8h"]),
            ("addv", &["b0", "v1.8b", "v2.8b"]),
            ("faddp", &["s0", "v1.4s"]),
            ("faddp", &["d0", "v1.2s"]),
            ("faddp", &["s0", "v1.2d"]),
            ("faddp", &["v0.8h", "v1.8h", "v2.8h"]),
            ("ext", &["v0.8b", "v1.8b", "v2.8b", "#8"]),
            ("ext", &["v0.16b", "v1.16b", "v2.16b", "#16"]),
            ("ext", &["v0.4s", "v1.4s", "v2.4s", "#1"]),
            ("ext", &["v0.8b", "v1.16b", "v2.8b", "#1"]),
            ("ext", &["v0.8b", "v1.8b", "#1"]),
            ("fmov", &["x0", "v1.d[0]"]),
            ("fmov", &["w0", "v1.d[1]"]),
            ("fmov", &["x0", "v1.s[1]"]),
            ("fmov", &["v1.d[1]", "w0"]),
            ("fmov", &["d0", "v1.d[1]"]),
            ("movi", &["v0.8b", "#256"]),
            ("movi", &["v0.16b", "#0x100"]),
            ("movi", &["v0.2d", "#1"]),
            ("movi", &["v0.2d", "#-1"]),
            ("movi", &["v0.2d", "#0x1ff"]),
            ("movi", &["v0.4s", "#0x3", "lsl #4"]),
            ("movi", &["v0.4s", "#0x3", "lsl #32"]),
            ("movi", &["v0.8b", "#0x3", "lsl #8"]),
            ("movi", &["v0.4s", "#0x100"]),
            ("movi", &["v0.4s", "1"]),
            ("movi", &["v0.4s"]),
            ("dup", &["v0.2s", "v1.d[1]"]),
            ("dup", &["v0.4s", "v1.h[2]"]),
            ("dup", &["v0.4s", "v1.s[4]"]),
            ("dup", &["v0.4s", "v1.4s"]),
            ("ld1", &["{ v0.4s, v2.4s }", "[x0]"]),
            ("ld1", &["{ v0.4s, v1.8h }", "[x0]"]),
            ("ld1", &["{ v0.4s }", "[x0], #16"]),
            ("ld1", &["{ v0.4s }", "[x31]"]),
            ("ld1", &["{ v0.4s }", "[w0]"]),
            ("ld1", &["{ v0.4s }", "[x0, #16]"]),
            ("ld1", &["{ v0.4s }", "x0"]),
            ("ld1", &["{ v0.4s }"]),
            ("ld1", &["{ v0.4s, v1.4s, v2.4s, v3.4s, v4.4s }", "[x0]"]),
            ("ld1", &["{ v0.1d }", "[x0]"]),
            ("ld1", &["{ v0 }", "[x0]"]),
            ("st1", &["{ v0.4s }", "[x0], #16"]),
        ];
        for (op, operands) in refused {
            let inst = ParsedInstruction {
                opcode: op.to_string(),
                operands: operands.iter().map(|s| s.to_string()).collect(),
            };
            assert!(
                AArch64Encoder::new().encode_instruction(&inst).is_err(),
                "{op} {operands:?}"
            );
        }
    }

    #[test]
    fn bit_count_and_reverse_match_arm_arm() {
        assert_eq!(word("rbit", &["x0", "x1"]), 0xDAC0_0020);
        assert_eq!(word("rev", &["w0", "w1"]), 0x5AC0_0820);
        assert_eq!(word("clz", &["w2", "w3"]), 0x5AC0_1062);
        assert_eq!(word("neg", &["x0", "x1"]), 0xCB01_03E0);
        assert_eq!(word("cnt", &["v0.8b", "v1.8b"]), 0x0E20_5820);
        assert_eq!(word("addv", &["b0", "v1.8b"]), 0x0E31_B820);
        assert_eq!(word("addv", &["b0", "v1.16b"]), 0x4E31_B820);
        assert_eq!(word("addv", &["h0", "v1.4h"]), 0x0E71_B820);
        assert_eq!(word("addv", &["h0", "v1.8h"]), 0x4E71_B820);
        assert_eq!(word("addv", &["s0", "v1.4s"]), 0x4EB1_B820);
        assert_eq!(
            word("addv.4s", &["s0", "v1"]),
            word("addv", &["s0", "v1.4s"])
        );
    }

    /// The sysreg fields come from the ARM ARM (DDI 0487) system register table; the
    /// MRS, MSR (register) and MSR (immediate) fixed bits are from the system instruction class.
    #[test]
    fn system_register_access_matches_arm_arm() {
        assert_eq!(word("mrs", &["x9", "cntvct_el0"]), 0xD53B_E049);
        assert_eq!(word("mrs", &["x9", "s3_3_c14_c0_2"]), 0xD53B_E049);
        assert_eq!(word("mrs", &["x0", "nzcv"]), 0xD53B_4200);
        assert_eq!(word("mrs", &["x3", "s3_1_c15_c2_6"]), 0xD539_F2C3);
        assert_eq!(word("mrs", &["x4", "s2_0_c0_c0_0"]), 0xD530_0004);
        assert_eq!(word("mrs", &["xzr", "midr_el1"]), 0xD538_001F);
        assert_eq!(word("msr", &["tpidr_el0", "x1"]), 0xD51B_D041);
        assert_eq!(word("msr", &["fpcr", "x2"]), 0xD51B_4402);
        assert_eq!(word("msr", &["DAIF", "x2"]), 0xD51B_4222);
        assert_eq!(word("msr", &["daifset", "#2"]), 0xD503_42DF);
        assert_eq!(word("msr", &["daifclr", "#15"]), 0xD503_4FFF);
        assert_eq!(word("msr", &["spsel", "#1"]), 0xD500_41BF);
    }

    #[test]
    fn logical_add_carry_select_and_multiply_forms_match_arm_arm() {
        assert_eq!(word("orn", &["x0", "x1", "x2"]), 0xAA22_0020);
        assert_eq!(word("bic", &["w3", "w4", "w5", "lsl #2"]), 0x0A25_0883);
        assert_eq!(word("bics", &["x0", "x1", "x2"]), 0xEA22_0020);
        assert_eq!(word("eon", &["x0", "x1", "x2"]), 0xCA22_0020);
        assert_eq!(word("mvn", &["x0", "x1"]), 0xAA21_03E0);
        assert_eq!(word("mvn", &["w2", "w3", "asr #7"]), 0x2AA3_1FE2);
        assert_eq!(word("and", &["w0", "w1", "w2", "lsl #4"]), 0x0A02_1020);
        assert_eq!(word("adds", &["x0", "x1", "x2"]), 0xAB02_0020);
        assert_eq!(word("adc", &["x0", "x1", "x2"]), 0x9A02_0020);
        assert_eq!(word("adcs", &["w0", "w1", "w2"]), 0x3A02_0020);
        assert_eq!(word("sbc", &["x0", "x1", "x2"]), 0xDA02_0020);
        assert_eq!(word("sbcs", &["x0", "x1", "x2"]), 0xFA02_0020);
        assert_eq!(word("ngc", &["x0", "x1"]), 0xDA01_03E0);
        assert_eq!(word("csinc", &["x0", "x1", "x2", "eq"]), 0x9A82_0420);
        assert_eq!(word("csneg", &["x0", "x1", "x2", "ne"]), 0xDA82_1420);
        assert_eq!(word("cinc", &["x0", "x1", "eq"]), 0x9A81_1420);
        assert_eq!(word("cinv", &["w0", "w1", "lt"]), 0x5A81_A020);
        assert_eq!(word("cneg", &["x0", "x1", "hi"]), 0xDA81_9420);
        assert_eq!(word("csetm", &["x0", "eq"]), 0xDA9F_13E0);
        assert_eq!(word("madd", &["x0", "x1", "x2", "x3"]), 0x9B02_0C20);
        assert_eq!(word("madd", &["w0", "w1", "w2", "w3"]), 0x1B02_0C20);
        assert_eq!(word("smaddl", &["x0", "w1", "w2", "x3"]), 0x9B22_0C20);
        assert_eq!(word("umaddl", &["x0", "w1", "w2", "x3"]), 0x9BA2_0C20);
        assert_eq!(word("smull", &["x0", "w1", "w2"]), 0x9B22_7C20);
        assert_eq!(word("umull", &["x0", "w1", "w2"]), 0x9BA2_7C20);
    }

    #[test]
    fn bitfield_insert_extract_and_halfword_zero_extend_match_arm_arm() {
        assert_eq!(word("bfi", &["x0", "x1", "#4", "#8"]), 0xB37C_1C20);
        assert_eq!(word("bfxil", &["w2", "w3", "#5", "#3"]), 0x3305_1C62);
        assert_eq!(word("ubfiz", &["x0", "x1", "#2", "#3"]), 0xD37E_0820);
        assert_eq!(word("sbfiz", &["x0", "x1", "#4", "#8"]), 0x937C_1C20);
        assert_eq!(word("sbfx", &["w0", "w1", "#3", "#4"]), 0x1303_1820);
        assert_eq!(word("extr", &["x0", "x1", "x2", "#5"]), 0x93C2_1420);
        assert_eq!(word("extr", &["w0", "w1", "w2", "#31"]), 0x1382_7C20);
        assert_eq!(word("uxth", &["w0", "w1"]), 0x5300_3C20);
    }

    #[test]
    fn system_register_and_operand_errors_are_refused() {
        let refused = [
            ("mrs", &["x0", "tpidr_el9"][..]),
            ("mrs", &["x0", "s1_0_c0_c0_0"]),
            ("mrs", &["x0", "s3_8_c0_c0_0"]),
            ("mrs", &["x0", "s3_0_c16_c0_0"]),
            ("mrs", &["x0", "s3_0_c0_c0"]),
            ("mrs", &["x0", "s3_0_c0_0_0"]),
            ("mrs", &["w0", "nzcv"]),
            ("mrs", &["sp", "nzcv"]),
            ("mrs", &["x0"]),
            ("msr", &["nzcv", "w0"]),
            ("msr", &["nzcv", "sp"]),
            ("msr", &["daifset", "#16"]),
            ("msr", &["daifset", "#-1"]),
            ("msr", &["spsel", "#2"]),
            ("msr", &["nzcv", "#1"]),
            ("msr", &["daifset", "x0"]),
            ("bic", &["x0", "x1", "#1"]),
            ("orn", &["x0", "x1", "w2"]),
            ("adc", &["x0", "x1", "w2"]),
            ("cinc", &["x0", "x1", "al"]),
            ("csetm", &["x0", "nv"]),
            ("madd", &["x0", "x1", "x2", "w3"]),
            ("smaddl", &["w0", "w1", "w2", "x3"]),
            ("smull", &["x0", "sp", "w2"]),
            ("extr", &["x0", "x1", "x2", "#64"]),
            ("extr", &["w0", "w1", "w2", "#32"]),
            ("bfi", &["x0", "x1", "#60", "#8"]),
            ("bfxil", &["w0", "w1", "#30", "#4"]),
            ("uxth", &["x0", "w1"]),
            ("uxth", &["w0", "x1"]),
        ];
        for (op, operands) in refused {
            let inst = ParsedInstruction {
                opcode: op.to_string(),
                operands: operands.iter().map(|s| s.to_string()).collect(),
            };
            assert!(
                AArch64Encoder::new().encode_instruction(&inst).is_err(),
                "{op} {operands:?}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    /// These compare ras against clang on ELF objects.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
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

        #[test]
        fn integer_float_memory_and_branch_encodings_match_clang() {
            let tmp = TempDir::new("aarch64_oracle");
            let dir = tmp.path();
            let source = dir.join("input.s");
            let reference = dir.join("reference.o");
            let output = dir.join("ras.o");
            let asm = r#"
        .section .rodata
        message: .asciz "ok"
        .text
        .globl entry
        entry:
        stp x29, x30, [sp, #-16]!
        mov x29, sp
        sub sp, sp, #32
        mov x9, #42
        mov w10, #-1
        movz x11, #123, lsl #32
        movk x11, #456, lsl #48
        add x12, x9, x11
        add w12, w9, w11
        sub w12, w9, w11
        and x12, x9, x11
        and w12, w9, w11
        mul x12, x9, x11
        mul w12, w9, w11
        msub x12, x9, x11, x10
        msub w12, w9, w11, w10
        sdiv x12, x9, x11
        udiv w12, w9, w11
        cmp x12, #0
        cmp w12, w11
        csel x12, x9, x11, eq
        cset x12, ne
        uxtb x9, w9
        fmov d0, x9
        fmov x9, d0
        fmov s0, w9
        fmov w9, s0
        fmul d0, d0, d1
        fmul s0, s0, s1
        str x9, [sp, #8]
        ldr x9, [sp, #8]
        str w9, [sp, #4]
        ldr w9, [sp, #4]
        strb w9, [sp]
        ldrb w9, [sp]
        strh w9, [sp, #2]
        ldrh w9, [sp, #2]
        stur x9, [x29, #-8]
        ldur x9, [x29, #-8]
        stp w9, w10, [sp, #8]
        ldp w9, w10, [sp, #8]
        stp w9, w10, [sp, #-8]!
        ldp w9, w10, [sp], #8
        adrp x0, message
        add x0, x0, :lo12:message
        b.ne .Ldone
        bl external
        b .Ldone
        .Ldone:
        dmb sy
        add sp, sp, #32
        ldp x29, x30, [sp], #16
        ret
        "#;
            std::fs::write(&source, asm).unwrap();
            let result = Command::new("clang")
                .args(["--target=aarch64-linux-gnu", "-c"])
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
            Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(asm, &output)
                .unwrap();
            assert_eq!(
                text(&std::fs::read(&output).unwrap()),
                text(&std::fs::read(&reference).unwrap())
            );
        }

        #[test]
        fn duplicate_symbols_are_rejected() {
            let tmp = TempDir::new("duplicate_aarch64");
            let output = tmp.path().join("duplicate_aarch64.o");
            assert!(
                Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
                    .unwrap()
                    .assemble("function: ret\nfunction: ret\n", &output)
                    .is_err()
            );
        }

        #[test]
        fn bl_to_same_section_local_label_resolves_in_place() {
            let tmp = TempDir::new("bl_local");
            let output = tmp.path().join("bl_local.o");
            let asm = ".text\nentry:\n  bl local\n  ret\nlocal:\n  ret\n";
            Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(asm, &output)
                .unwrap();
            let bytes = std::fs::read(&output).unwrap();
            assert_eq!(text(&bytes)[..4], 0x9400_0002u32.to_le_bytes());
        }

        #[test]
        fn hints_system_calls_and_register_branches_encode_like_clang() {
            let tmp = TempDir::new("hints_aarch64");
            let output = tmp.path().join("hints_aarch64.o");
            let asm = ".text\nnop\nyield\nwfi\nsvc #0\nbrk #0x1234\nbr x1\nblr x30\n";
            Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(asm, &output)
                .unwrap();
            let words: Vec<u32> = text(&std::fs::read(&output).unwrap())
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(
                words,
                [
                    0xd503_201f,
                    0xd503_203f,
                    0xd503_207f,
                    0xd400_0001,
                    0xd420_0000 | (0x1234 << 5),
                    0xd61f_0020,
                    0xd63f_03c0,
                ]
            );
        }

        #[test]
        fn branches_to_global_same_section_labels_are_relocated() {
            let tmp = TempDir::new("global_b");
            let output = tmp.path().join("global_b.o");
            let asm = ".text\n.globl g\ng:\n  b g\n  b.eq g\n  ret\n";
            Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(asm, &output)
                .unwrap();
            let words: Vec<u32> = text(&std::fs::read(&output).unwrap())
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(words[..2], [0x1400_0000, 0x5400_0000]);
        }

        fn assemble_bytes(asm: &str) -> Result<Vec<u8>, String> {
            let tmp = TempDir::new("aarch64_words");
            let output = tmp.path().join("aarch64_words.o");
            Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(asm, &output)
                .map_err(|e| e.to_string())
                .and_then(|()| std::fs::read(&output).map_err(|e| e.to_string()))
        }

        fn assemble_words(asm: &str) -> Result<Vec<u32>, String> {
            assemble_bytes(asm).map(|bytes| {
                text(&bytes)
                    .chunks(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            })
        }

        /// (offset, relocation type, symbol name, addend) for every RELA entry in an object file.
        fn relocations(bytes: &[u8]) -> Vec<(u64, u32, String, i64)> {
            let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            let shoff = u64_at(40) as usize;
            let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
            let section = |i: usize| shoff + i * 64;
            let mut out = Vec::new();
            for i in 0..count {
                let header = section(i);
                if u32_at(header + 4) != 4 {
                    continue;
                }
                let symtab = section(u32_at(header + 40) as usize);
                let symbols = u64_at(symtab + 24) as usize;
                let strings = u64_at(section(u32_at(symtab + 40) as usize) + 24) as usize;
                let entries = u64_at(header + 24) as usize;
                for entry in (entries..entries + u64_at(header + 32) as usize).step_by(24) {
                    let info = u64_at(entry + 8);
                    let name_at = strings + u32_at(symbols + (info >> 32) as usize * 24) as usize;
                    let name_len = bytes[name_at..].iter().position(|&b| b == 0).unwrap();
                    let name =
                        String::from_utf8_lossy(&bytes[name_at..name_at + name_len]).into_owned();
                    out.push((u64_at(entry), info as u32, name, u64_at(entry + 16) as i64));
                }
            }
            out
        }

        #[test]
        fn add_sub_and_cmp_immediates_encode_the_lsl_12_form() {
            assert_eq!(
                assemble_words(".text\nadd x0, x1, #4, lsl #12\n").unwrap(),
                [0x9140_1020]
            );
            assert_eq!(
                assemble_words(".text\nadd x0, x1, #4096\n").unwrap(),
                [0x9140_0420]
            );
            assert_eq!(
                assemble_words(".text\nsub w0, w1, #1, lsl #12\n").unwrap(),
                [0x5140_0420]
            );
            assert_eq!(
                assemble_words(".text\ncmp x0, #1, lsl #12\n").unwrap(),
                [0xf140_041f]
            );
        }

        #[test]
        fn add_sub_and_cmp_immediates_reject_unrepresentable_shifts() {
            for asm in [
                ".text\nadd x0, x1, #1, lsl #24\n",
                ".text\nadd x0, x1, #1, lsr #12\n",
                ".text\nadd x0, x1, #4097\n",
                ".text\nadd x0, x1, #16781312\n",
                ".text\nadd x0, x1, #4096, lsl #12\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn add_sub_and_cmp_shifted_registers_encode_their_shift() {
            assert_eq!(
                assemble_words(".text\nadd x0, x1, x2, lsl #3\n").unwrap(),
                [0x8b02_0c20]
            );
            assert_eq!(
                assemble_words(".text\nsub x0, x1, x2, lsr #5\n").unwrap(),
                [0xcb42_1420]
            );
            assert_eq!(
                assemble_words(".text\ncmp x0, x1, lsl #2\n").unwrap(),
                [0xeb01_081f]
            );
        }

        #[test]
        fn add_sub_and_cmp_reject_out_of_range_or_unknown_shifts() {
            for asm in [
                ".text\nadd x0, x1, x2, lsl #64\n",
                ".text\nadd w0, w1, w2, lsl #32\n",
                ".text\nadd x0, x1, x2, ror #1\n",
                ".text\nadd x0, x1, x2, lsl\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn add_sub_and_cmp_with_stack_pointer_use_the_extended_form() {
            assert_eq!(
                assemble_words(".text\nadd sp, sp, x1\n").unwrap(),
                [0x8b21_63ff]
            );
            assert_eq!(
                assemble_words(".text\nadd x0, sp, x1\n").unwrap(),
                [0x8b21_63e0]
            );
            assert_eq!(
                assemble_words(".text\nsub sp, x1, x2\n").unwrap(),
                [0xcb22_603f]
            );
            assert_eq!(
                assemble_words(".text\ncmp sp, x0\n").unwrap(),
                [0xeb20_63ff]
            );
            assert_eq!(
                assemble_words(".text\nadd wsp, w1, w2\n").unwrap(),
                [0x0b22_403f]
            );
            assert_eq!(
                assemble_words(".text\nadd x0, sp, x1, lsl #4\n").unwrap(),
                [0x8b21_73e0]
            );
            for asm in [
                ".text\nadd x0, sp, x1, lsr #2\n",
                ".text\nadd x0, sp, x1, lsl #5\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn add_sub_and_cmp_reject_mixed_register_widths_and_fp_operands() {
            for asm in [
                ".text\nadd w0, x1, #0\n",
                ".text\nadd x0, w1, #1\n",
                ".text\nadd w0, w1, x2\n",
                ".text\nadd x0, x1, w2\n",
                ".text\nsub w0, x1, x2\n",
                ".text\ncmp w0, x1\n",
                ".text\nadd x0, x1, sp\n",
                ".text\ncmp x0, sp\n",
                ".text\nadd d0, x1, x2\n",
                ".text\nadd x0, s1, x2\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn extra_operands_on_fixed_arity_instructions_are_rejected() {
            for asm in [
                ".text\nret x0, x1\n",
                ".text\nbr x0, x1\n",
                ".text\nsvc #0, #1\n",
                ".text\ndmb sy, sy\n",
                ".text\nmul x0, x1, x2, x3\n",
                ".text\nmsub x0, x1, x2, x3, x4\n",
                ".text\ncsel x0, x1, x2, eq, junk\n",
                ".text\ncset x0, eq, x1\n",
                ".text\nuxtb w0, w1, w2\n",
                ".text\nmov x0, x1, lsl #3\n",
                ".text\nmov x0, #1, lsl #16\n",
                ".text\nmovz x0, #1, lsl #16, extra\n",
                ".text\nmovk x0, #1, lsl #16, extra\n",
                ".text\nfmul d0, d1, d2, d3\n",
                ".text\nldr x9, [sp, #8, lsl #3]\n",
                ".text\nldr w9, [sp, #8, lsl #3]\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn w_register_mov_immediates_must_fit_in_32_bits() {
            for asm in [
                ".text\nmov w0, #0x1ffff0000\n",
                ".text\nmov w0, #8589869056\n",
                ".text\nmov w0, #4294967296\n",
                ".text\nmov w0, #-2147483649\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
            assert_eq!(
                assemble_words(".text\nmov w0, #-2147483648\n").unwrap(),
                [0x52b0_0000]
            );
        }

        #[test]
        fn data_processing_registers_must_share_one_width_and_avoid_sp() {
            for asm in [
                ".text\nmul x0, w1, x2\n",
                ".text\nmul w0, x1, x2\n",
                ".text\nsdiv x0, x1, w2\n",
                ".text\nudiv w0, x1, x2\n",
                ".text\nmsub w0, w1, w2, x3\n",
                ".text\ncsel w0, x1, x2, eq\n",
                ".text\nand w0, x1, x2\n",
                ".text\nmul sp, x1, x2\n",
                ".text\ncsel sp, x1, x2, eq\n",
                ".text\ncset w0, al\n",
                ".text\ncset w0, nv\n",
                ".text\nuxtb w0, x1\n",
                ".text\nret w0\n",
                ".text\nret wzr\n",
                ".text\nbr w0\n",
                ".text\nblr w0\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
            assert_eq!(assemble_words(".text\nret\n").unwrap(), [0xd65f_03c0]);
            assert_eq!(assemble_words(".text\nret xzr\n").unwrap(), [0xd65f_03e0]);
            assert_eq!(
                assemble_words(".text\ncset w0, hs\n").unwrap(),
                [0x1a9f_37e0]
            );
        }

        #[test]
        fn fp_scalar_and_pair_loads_and_stores_encode_their_register_size() {
            assert_eq!(
                assemble_words(".text\nldr d0, [x1]\n").unwrap(),
                [0xfd40_0020]
            );
            assert_eq!(
                assemble_words(".text\nstr s0, [sp, #4]\n").unwrap(),
                [0xbd00_07e0]
            );
            assert_eq!(
                assemble_words(".text\nldp d0, d1, [sp]\n").unwrap(),
                [0x6d40_07e0]
            );
            assert_eq!(
                assemble_words(".text\nldp d0, d1, [sp, #8]\n").unwrap(),
                [0x6d40_87e0]
            );
            assert_eq!(
                assemble_words(".text\nldp d0, d1, [x0], #32\n").unwrap(),
                [0x6cc2_0400]
            );
        }

        #[test]
        fn loads_and_stores_reject_bad_base_and_data_registers() {
            for asm in [
                ".text\nldr x0, [w1]\n",
                ".text\nldr x0, [xzr, #8]\n",
                ".text\nldr x0, [wsp]\n",
                ".text\nldrb x9, [sp]\n",
                ".text\nstrh x9, [sp]\n",
                ".text\nldr sp, [x0]\n",
                ".text\nldr wsp, [x0]\n",
                ".text\nldr w9, [x0, #8, lsl #3]\n",
                ".text\nldp x0, w1, [sp]\n",
                ".text\nldp d0, s1, [sp]\n",
                ".text\nldp d0, d1, [sp, #4]\n",
                ".text\nldp x0, x1, [x0], #32\n",
                ".text\nstp x0, x1, [x0, #-16]!\n",
                ".text\nstp x0\n",
                ".text\nldr\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn symbol_operands_keep_their_addend_in_relocations() {
            let rel = relocations(
                &assemble_bytes(".text\nbl sym+4\nadrp x0, sym+8\nadd x0, x0, :lo12:sym-4\n")
                    .unwrap(),
            );
            assert!(rel.contains(&(0, 283, "sym".into(), 4)), "{rel:?}");
            assert!(rel.contains(&(4, 275, "sym".into(), 8)), "{rel:?}");
            assert!(rel.contains(&(8, 277, "sym".into(), -4)), "{rel:?}");
        }

        #[test]
        fn local_branch_targets_honor_their_addend() {
            assert_eq!(
                assemble_words(".text\nb tgt+4\ntgt:\nret\n").unwrap(),
                [0x1400_0002, 0xd65f_03c0]
            );
        }

        #[test]
        fn got_and_lo12_relocations_use_their_own_kinds() {
            let rel = relocations(
                &assemble_bytes(
                    ".text\nadrp x0, :got:sym\nldr x0, [x0, :got_lo12:sym]\nldr x1, [x2, #:lo12:sym]\n",
                )
                .unwrap(),
            );
            assert_eq!(
                rel,
                [
                    (0, 311, "sym".to_string(), 0),
                    (4, 312, "sym".to_string(), 0),
                    (8, 286, "sym".to_string(), 0),
                ]
            );
            assert_eq!(
                assemble_words(
                    ".text\nadrp x0, :got:sym\nldr x0, [x0, :got_lo12:sym]\nldr x1, [x2, :lo12:sym]\n"
                )
                .unwrap(),
                [0x9000_0000, 0xf940_0000, 0xf940_0041]
            );
        }

        #[test]
        fn unsupported_relocation_modifiers_are_rejected() {
            for asm in [
                ".text\nadrp x0, :tlsdesc:sym\n",
                ".text\nadrp x0, :tlsgd:sym\n",
                ".text\nldr w0, [x1, :lo12:sym]\n",
                ".text\nldr x0, [x0, :tprel_lo12_nc:sym]\n",
                ".text\nmovz x0, :abs_g0:sym\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn compare_and_test_branches_encode_their_label_offsets() {
            assert_eq!(
                assemble_words(".text\ncbz x0, tgt\ntgt:\nret\n").unwrap(),
                [0xb400_0020, 0xd65f_03c0]
            );
            assert_eq!(
                assemble_words(".text\ncbnz w1, tgt\ntgt:\nret\n").unwrap(),
                [0x3500_0021, 0xd65f_03c0]
            );
            assert_eq!(
                assemble_words(".text\ntbz x0, #3, tgt\ntgt:\nret\n").unwrap(),
                [0x3618_0020, 0xd65f_03c0]
            );
            assert_eq!(
                assemble_words(".text\ntbnz w0, #31, tgt\ntgt:\nret\n").unwrap(),
                [0x37f8_0020, 0xd65f_03c0]
            );
            assert_eq!(
                assemble_words(".text\ntbnz x0, #40, tgt\ntgt:\nret\n").unwrap(),
                [0xb740_0020, 0xd65f_03c0]
            );
        }

        #[test]
        fn compare_and_test_branches_reject_out_of_range_labels() {
            for asm in [
                ".text\ncbz x0, tgt\n.space 1048572\ntgt:\nret\n",
                ".text\ntbz x0, #3, tgt\n.space 32764\ntgt:\nret\n",
                ".text\ntgt:\n.space 32772\ntbz x0, #3, tgt\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
            assert!(assemble_words(".text\ncbz x0, tgt\n.space 1048568\ntgt:\nret\n").is_ok());
            assert!(assemble_words(".text\ntbz x0, #3, tgt\n.space 32760\ntgt:\nret\n").is_ok());
        }

        #[test]
        fn compare_and_test_branches_reject_bad_operands() {
            for asm in [
                ".text\ncbz x0\n",
                ".text\ncbz sp, tgt\n",
                ".text\ntbz w0, #32, tgt\n",
                ".text\ntbz x0, #64, tgt\n",
                ".text\ntbz x0, tgt\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn int_float_conversions_encode_every_width_pair() {
            for (asm, word) in [
                ("scvtf s0, w0", 0x1e22_0000u32),
                ("scvtf s0, x0", 0x9e22_0000),
                ("scvtf d0, w0", 0x1e62_0000),
                ("scvtf d0, x0", 0x9e62_0000),
                ("ucvtf s0, w0", 0x1e23_0000),
                ("ucvtf s0, x0", 0x9e23_0000),
                ("ucvtf d0, w0", 0x1e63_0000),
                ("ucvtf d0, x0", 0x9e63_0000),
                ("fcvtzs w0, s0", 0x1e38_0000),
                ("fcvtzs x0, s0", 0x9e38_0000),
                ("fcvtzs w0, d0", 0x1e78_0000),
                ("fcvtzs x0, d0", 0x9e78_0000),
                ("fcvtzu w0, s0", 0x1e39_0000),
                ("fcvtzu x0, s0", 0x9e39_0000),
                ("fcvtzu w0, d0", 0x1e79_0000),
                ("fcvtzu x0, d0", 0x9e79_0000),
            ] {
                assert_eq!(
                    assemble_words(&format!(".text\n{asm}\n")).unwrap(),
                    [word],
                    "{asm}"
                );
            }
            for asm in [
                "scvtf x0, w0",
                "fcvtzs w0, x0",
                "scvtf s0, sp",
                "scvtf s0",
                "scvtf w0, w1",
                "fcvtzs w0, w1",
            ] {
                assert!(
                    assemble_words(&format!(".text\n{asm}\n")).is_err(),
                    "accepted {asm:?}"
                );
            }
        }

        #[test]
        fn ldp_rejects_equal_transfer_registers() {
            for asm in [
                "ldp x0, x0, [sp]",
                "ldp xzr, xzr, [sp]",
                "ldp w0, w0, [sp]",
                "ldp s0, s0, [sp, #8]",
                "ldp d0, d0, [sp, #8]",
            ] {
                assert!(
                    assemble_words(&format!(".text\n{asm}\n")).is_err(),
                    "accepted {asm:?}"
                );
            }
            assert_eq!(
                assemble_words(".text\nstp x0, x0, [sp]\n").unwrap(),
                [0xa900_03e0]
            );
        }

        #[test]
        fn adrp_takes_a_64_bit_general_register_but_not_sp() {
            for asm in ["adrp w0, x", "adrp wzr, x", "adrp sp, x"] {
                assert!(
                    assemble_words(&format!(".text\n{asm}\nx:\n")).is_err(),
                    "accepted {asm:?}"
                );
            }
            assert_eq!(
                assemble_words(".text\nadrp x0, x\nx:\n").unwrap(),
                [0x9000_0000]
            );
            assert_eq!(
                assemble_words(".text\nadrp xzr, x\nx:\n").unwrap(),
                [0x9000_001f]
            );
        }

        #[test]
        fn stack_pointer_destinations_of_mov_forms() {
            for asm in [
                "movz sp, #1",
                "movk wsp, #1",
                "movn sp, #1",
                "mov sp, #0x12345678",
                "mov wsp, #0",
            ] {
                assert!(
                    assemble_words(&format!(".text\n{asm}\n")).is_err(),
                    "accepted {asm:?}"
                );
            }
            assert_eq!(
                assemble_words(".text\nmov sp, #1\n").unwrap(),
                [0xb240_03ff]
            );
            assert_eq!(
                assemble_words(".text\nmov wsp, #1\n").unwrap(),
                [0x3200_03ff]
            );
            assert_eq!(
                assemble_words(".text\nmovz xzr, #1\n").unwrap(),
                [0xd280_003f]
            );
        }

        #[test]
        fn fmov_rejects_stack_pointer_operands() {
            for asm in ["fmov s0, sp", "fmov sp, s0", "fmov s0, wsp", "fmov d0, sp"] {
                assert!(
                    assemble_words(&format!(".text\n{asm}\n")).is_err(),
                    "accepted {asm:?}"
                );
            }
            assert_eq!(
                assemble_words(".text\nfmov s0, s1\n").unwrap(),
                [0x1e20_4020]
            );
        }

        #[test]
        fn branch_targets_must_be_symbols() {
            for asm in [
                ".text\nbl x0\n",
                ".text\nb w0\n",
                ".text\nb sp\n",
                ".text\nb lr\n",
                ".text\nbl xzr\n",
                ".text\nbl 4\n",
                ".text\nb 4\n",
                ".text\nb :got:x\nx:\n",
                ".text\nb.eq x0\nx:\n",
                ".text\nbl sp, x\nx:\n",
                ".text\nb x, x\nx:\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
            assert_eq!(assemble_words(".text\nbl x\nx:\n").unwrap(), [0x9400_0001]);
        }

        #[test]
        fn logical_immediates_encode_as_bitmask_fields() {
            assert_eq!(
                assemble_words(".text\nand x0, x1, #0xff\n").unwrap(),
                [0x9240_1c20]
            );
            assert_eq!(
                assemble_words(".text\norr w0, w1, #0x80000000\n").unwrap(),
                [0x3201_0020]
            );
            assert_eq!(
                assemble_words(".text\neor x0, x1, #0x5555555555555555\n").unwrap(),
                [0xd200_f020]
            );
            assert_eq!(
                assemble_words(".text\nands w0, w1, #0x1\n").unwrap(),
                [0x7200_0020]
            );
            assert_eq!(
                assemble_words(".text\ntst x0, #0xff\n").unwrap(),
                [0xf240_1c1f]
            );
        }

        #[test]
        fn logical_register_forms_encode_orr_eor_ands_and_tst() {
            assert_eq!(
                assemble_words(".text\norr x0, x1, x2\n").unwrap(),
                [0xaa02_0020]
            );
            assert_eq!(
                assemble_words(".text\neor w0, w1, w2\n").unwrap(),
                [0x4a02_0020]
            );
            assert_eq!(
                assemble_words(".text\nands x0, x1, x2\n").unwrap(),
                [0xea02_0020]
            );
            assert_eq!(
                assemble_words(".text\ntst w0, w1\n").unwrap(),
                [0x6a01_001f]
            );
        }

        #[test]
        fn logical_immediates_reject_values_that_cannot_be_encoded() {
            for asm in [
                ".text\nand x0, x1, #0\n",
                ".text\nand x0, x1, #-1\n",
                ".text\nand w0, w1, #0x100000000\n",
                ".text\nand w0, w1, #0x101\n",
                ".text\norr x0, x1, #0x12345\n",
                ".text\nand x0, sp, #1\n",
                ".text\nand x0, x1, w2\n",
                ".text\ntst x0, #0\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn wide_mov_immediates_lower_to_move_wide_chains() {
            assert_eq!(
                assemble_words(".text\nmov x0, #0x12345678\n").unwrap(),
                [0xd28a_cf00, 0xf2a2_4680]
            );
            assert_eq!(
                assemble_words(".text\nmov x0, #0x123456789abcdef0\n").unwrap(),
                [0xd29b_de00, 0xf2b3_5780, 0xf2ca_cf00, 0xf2e2_4680]
            );
            assert_eq!(
                assemble_words(".text\nmov x0, #0xffffffffffff1234\n").unwrap(),
                [0x929d_b960]
            );
            assert_eq!(
                assemble_words(".text\nmov w0, #-1\n").unwrap(),
                [0x1280_0000]
            );
            assert_eq!(
                assemble_words(".text\nmov w0, #0xffff0000\n").unwrap(),
                [0x52bf_ffe0]
            );
            assert_eq!(
                assemble_words(".text\nmov x0, #0x0000ffff0000ffff\n").unwrap(),
                [0xb200_3fe0]
            );
            assert_eq!(
                assemble_words(".text\nmov x0, #0xffff1234ffff5678\n").unwrap(),
                [0x9295_30e0, 0xf2c2_4680]
            );
            assert_eq!(
                assemble_words(".text\nmov w0, #74565\n").unwrap(),
                [0x5284_68a0, 0x72a0_0020]
            );
            assert_eq!(
                assemble_words(".text\nmovn x0, #1\n").unwrap(),
                [0x9280_0020]
            );
            assert_eq!(
                assemble_words(".text\nmovn w0, #0xffff, lsl #16\n").unwrap(),
                [0x12bf_ffe0]
            );
            for asm in [
                ".text\nmovn w0, #0x10000\n",
                ".text\nmovn x0, #1, lsl #3\n",
                ".text\nmov d0, #1\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn pre_and_post_indexed_loads_and_stores_encode_their_writeback() {
            assert_eq!(
                assemble_words(".text\nldr x9, [x0, #8]!\n").unwrap(),
                [0xf840_8c09]
            );
            assert_eq!(
                assemble_words(".text\nstr w9, [x0], #8\n").unwrap(),
                [0xb800_8409]
            );
            assert_eq!(
                assemble_words(".text\nldrb w9, [x0], #-1\n").unwrap(),
                [0x385f_f409]
            );
            assert_eq!(
                assemble_words(".text\nldr d0, [x1], #16\n").unwrap(),
                [0xfc41_0420]
            );
            assert_eq!(
                assemble_words(".text\nstr x9, [sp, #-16]!\n")
                    .unwrap()
                    .len(),
                1
            );
        }

        #[test]
        fn writeback_forms_reject_unencodable_offsets_and_registers() {
            for asm in [
                ".text\nldr x9, [x0, #256]!\n",
                ".text\nldr x9, [x0], #256\n",
                ".text\nldr x9, [x0], x1\n",
                ".text\nldr x0, [x0, #8]!\n",
                ".text\nstr x0, [x0], #8\n",
                ".text\nldr x9, [x0], #8, #8\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn barrier_options_and_carry_condition_aliases_encode() {
            assert_eq!(
                assemble_words(".text\ndmb ish\ndmb ishld\ndsb sy\nisb\n").unwrap(),
                [0xd503_3bbf, 0xd503_39bf, 0xd503_3f9f, 0xd503_3fdf]
            );
            assert_eq!(
                assemble_words(".text\ncset w0, cc\ncset w0, cs\ncsel w0, w1, w2, cs\n").unwrap(),
                [0x1a9f_27e0, 0x1a9f_37e0, 0x1a82_2020]
            );
            assert!(assemble_words(".text\ndmb foo\n").is_err());
        }

        #[test]
        fn fmul_takes_scalar_registers_of_one_size() {
            assert_eq!(
                assemble_words(".text\nfmul d0, d1, d2\nfmul s0, s1, s2\n").unwrap(),
                [0x1e62_0820, 0x1e22_0820]
            );
            for asm in [
                ".text\nfmul x0, x1, x2\n",
                ".text\nfmul d0, s1, d2\n",
                ".text\nfmul s0, d1, d2\n",
                ".text\nfmul d0, sp, d2\n",
                ".text\nfmul d0, d1\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn scalar_fp_register_31_is_valid_but_general_register_31_is_not() {
            assert_eq!(
                assemble_words(".text\nfmul d31, d1, d2\n").unwrap(),
                [0x1e62_083f]
            );
            assert!(assemble_words(".text\nadd x0, x31, x1\n").is_err());
            assert!(assemble_words(".text\nmov w31, w0\n").is_err());
        }

        #[test]
        fn unscaled_loads_and_stores_have_no_writeback_form() {
            for asm in [
                ".text\nldur x9, [x0, #8]!\n",
                ".text\nstur w9, [x0], #8\n",
                ".text\nldurb w9, [x0], #8\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn post_indexed_loads_and_stores_reject_an_inner_offset() {
            for asm in [
                ".text\nldr x9, [sp, #8], #8\n",
                ".text\nstr w9, [sp, #8], #8\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
            assert!(assemble_words(".text\nldr x9, [sp], #8\n").is_ok());
        }

        #[test]
        fn register_mov_requires_matching_widths_and_general_registers() {
            assert_eq!(
                assemble_words(".text\nmov x0, sp\n").unwrap(),
                [0x9100_03e0]
            );
            assert_eq!(
                assemble_words(".text\nmov x0, x1\n").unwrap(),
                [0xaa01_03e0]
            );
            assert_eq!(
                assemble_words(".text\nmov wsp, w1\n").unwrap(),
                [0x1100_003f]
            );
            for asm in [
                ".text\nmov sp, w0\n",
                ".text\nmov w0, sp\n",
                ".text\nmov x0, wsp\n",
                ".text\nmov x0, wzr\n",
                ".text\nmov w0, xzr\n",
                ".text\nmov x0, d1\n",
                ".text\nmov w0, s1\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn symbol_operands_must_name_a_symbol() {
            for asm in [
                ".text\nadd x0, x1, :lo12:\n",
                ".text\nadrp x0, :got:\n",
                ".text\nadrp x0, \n",
                ".text\nbl +4\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn udf_encodes_bare_immediate() {
            assert_eq!(
                assemble_words(".text\nudf #0\nudf #1\nudf #0xffff\nudf #0x1234\n").unwrap(),
                [0x0000_0000, 0x0000_0001, 0x0000_ffff, 0x0000_1234]
            );
            assert!(assemble_words(".text\nudf #0x10000\n").is_err());
            assert!(assemble_words(".text\nudf\n").is_err());
            assert!(assemble_words(".text\nudf #0, #1\n").is_err());
        }

        #[test]
        fn compare_negative_encodes_as_flag_setting_add() {
            assert_eq!(
                assemble_words(".text\ncmn x0, x1\ncmn w2, w3\ncmn x4, #16\ncmn x7, x8, lsl #3\n")
                    .unwrap(),
                [0xab01_001f, 0x2b03_005f, 0xb100_409f, 0xab08_0cff]
            );
            for asm in [".text\ncmn x0, w1\n", ".text\ncmn x0, #4097\n"] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn bitfield_extracts_cover_lsb_and_width() {
            assert_eq!(
                assemble_words(".text\nsbfx x0, x1, #0, #8\nubfx w2, w3, #16, #16\nsxtw x0, w1\n")
                    .unwrap(),
                [0x9340_1c20, 0x5310_7c62, 0x9340_7c20]
            );
            for asm in [
                ".text\nsbfx x0, x1, #0, #0\n",
                ".text\nsbfx x0, x1, #60, #8\n",
                ".text\nubfx x0, x1, #-1, #8\n",
                ".text\nsxtw w0, w1\n",
            ] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn shifts_take_both_immediate_and_register_amounts() {
            assert_eq!(
                assemble_words(".text\nlsl x9, x11, #56\nasr x12, x10, #56\nlsr w2, w3, #31\n")
                    .unwrap(),
                [0xd348_1d69, 0x9378_fd4c, 0x531f_7c62]
            );
            assert_eq!(
                assemble_words(".text\nlsl x0, x1, x2\nasrv w3, w4, w5\nrorv x12, x13, x14\n")
                    .unwrap(),
                [0x9ac2_2020, 0x1ac5_2883, 0x9ace_2dac]
            );
            for asm in [".text\nlsl x0, x1, #64\n", ".text\nasr w0, w1, #32\n"] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn signed_loads_pick_the_destination_width() {
            assert_eq!(
                assemble_words(".text\nldrsb w0, [x1]\nldrsb x0, [x1]\nldrsw x0, [x1]\n").unwrap(),
                [0x39c0_0020, 0x3980_0020, 0xb980_0020]
            );
            assert_eq!(
                assemble_words(".text\nldursb x2, [x3, #-256]\nldursw x6, [x7, #-4]\n").unwrap(),
                [0x3890_0062, 0xb89f_c0e6]
            );
            for asm in [".text\nldrsw w0, [x1]\n", ".text\nldrsb s0, [x1]\n"] {
                assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
            }
        }

        #[test]
        fn scalar_float_arithmetic_covers_all_four_operations() {
            assert_eq!(
                assemble_words(".text\nfadd s0, s0, s1\nfsub d8, d9, d10\nfdiv s11, s12, s13\n")
                    .unwrap(),
                [0x1e21_2800, 0x1e6a_3928, 0x1e2d_198b]
            );
            assert!(assemble_words(".text\nfadd s0, s1, d2\n").is_err());
        }
    }

    /// aarch64 thread-local relocations.
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
                        offset: u64_at(bytes, h + 24),
                        size: u64_at(bytes, h + 32),
                        link: u32_at(bytes, h + 40) as usize,
                        entsize: u64_at(bytes, h + 56),
                    }
                })
                .collect()
        }

        fn text_words(bytes: &[u8]) -> Vec<u32> {
            let text = headers(bytes)
                .into_iter()
                .find(|h| h.name == ".text")
                .unwrap();
            (0..text.size / 4)
                .map(|i| u32_at(bytes, text.offset + i * 4))
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

        const COUNTER: &str = ".section .tbss,\"awT\",@nobits\n\
                               .globl counter\n\
                               .type counter,@tls_object\n\
                               counter: .zero 8\n";

        fn try_assemble(
            name: &str,
            os: TargetOperatingSystem,
            source: &str,
        ) -> Result<Vec<u8>, String> {
            let tmp = TempDir::new(&format!("aarch64_tls_{name}"));
            let path = tmp.path().join(format!("aarch64_tls_{name}.o"));
            Ras::new(TargetArchitecture::Aarch64, os)
                .unwrap()
                .assemble(source, &path)
                .map_err(|e| format!("{e:?}"))?;
            Ok(std::fs::read(&path).unwrap())
        }

        fn assemble(name: &str, source: &str) -> Vec<u8> {
            try_assemble(name, TargetOperatingSystem::Linux, source).unwrap()
        }

        fn refusal(name: &str, source: &str) -> String {
            try_assemble(name, TargetOperatingSystem::Linux, source).unwrap_err()
        }

        #[test]
        fn local_exec_sequence_emits_tprel_relocations() {
            let bytes = assemble(
                "local_exec",
                &format!(
                    ".text\n\
                     .globl f\n\
                     f:\n\
                     mrs x9, tpidr_el0\n\
                     add x9, x9, :tprel_hi12:counter, lsl #12\n\
                     add x9, x9, :tprel_lo12_nc:counter\n\
                     ret\n{COUNTER}"
                ),
            );
            assert_eq!(
                text_words(&bytes),
                vec![0xd53b_d049, 0x9140_0129, 0x9100_0129, 0xd65f_03c0]
            );
            assert_eq!(
                text_relocations(&bytes),
                vec![
                    (4, 549, "counter".to_string(), 0),
                    (8, 551, "counter".to_string(), 0),
                ]
            );
        }

        #[test]
        fn initial_exec_sequence_emits_gottprel_relocations() {
            let bytes = assemble(
                "initial_exec",
                &format!(
                    ".text\n\
                     .globl f\n\
                     f:\n\
                     adrp x11, :gottprel:counter\n\
                     ldr x11, [x11, :gottprel_lo12:counter]\n\
                     mrs x12, tpidr_el0\n\
                     add x11, x11, x12\n\
                     ret\n{COUNTER}"
                ),
            );
            assert_eq!(
                text_words(&bytes)[..2],
                [0x9000_000b, 0xf940_016b],
                "the page and page-offset fields must be left at zero for the linker"
            );
            assert_eq!(
                text_relocations(&bytes),
                vec![
                    (0, 541, "counter".to_string(), 0),
                    (4, 542, "counter".to_string(), 0),
                ]
            );
        }

        #[test]
        fn tprel_lo12_without_nc_is_a_distinct_relocation() {
            let bytes = assemble(
                "tprel_lo12",
                &format!(
                    ".text\n\
                     f:\n\
                     add x9, x9, :tprel_lo12:counter\n{COUNTER}"
                ),
            );
            assert_eq!(text_words(&bytes), vec![0x9100_0129]);
            assert_eq!(
                text_relocations(&bytes),
                vec![(0, 550, "counter".to_string(), 0)]
            );
        }

        #[test]
        fn tprel_add_carries_an_addend() {
            let bytes = assemble(
                "tprel_addend",
                &format!(
                    ".text\n\
                     f:\n\
                     add x9, x9, :tprel_lo12_nc:counter+8\n{COUNTER}"
                ),
            );
            assert_eq!(text_words(&bytes), vec![0x9100_0129]);
            assert_eq!(
                text_relocations(&bytes),
                vec![(0, 551, "counter".to_string(), 8)]
            );
        }

        #[test]
        fn tprel_add_refuses_32_bit_registers() {
            let message = refusal(
                "tprel_w",
                &format!(
                    ".text\n\
                     f:\n\
                     add w9, w9, :tprel_lo12_nc:counter\n{COUNTER}"
                ),
            );
            assert!(
                message.contains(":tprel_lo12_nc:counter takes 64-bit registers"),
                "{message}"
            );
        }

        #[test]
        fn tprel_hi12_refuses_a_shift_other_than_12() {
            let message = refusal(
                "tprel_shift",
                &format!(
                    ".text\n\
                     f:\n\
                     add x9, x9, :tprel_hi12:counter, lsl #0\n{COUNTER}"
                ),
            );
            assert!(
                message.contains("Unsupported shift for :tprel_hi12:counter"),
                "{message}"
            );
        }

        #[test]
        fn unknown_tprel_modifier_is_refused() {
            let message = refusal(
                "tprel_unknown",
                &format!(
                    ".text\n\
                     f:\n\
                     add x9, x9, :tprel_g0:counter\n{COUNTER}"
                ),
            );
            assert!(
                message.contains("Unsupported relocation modifier: :tprel_g0:counter"),
                "{message}"
            );
        }

        #[test]
        fn unknown_gottprel_modifier_is_refused() {
            let message = refusal(
                "gottprel_unknown",
                &format!(
                    ".text\n\
                     f:\n\
                     ldr x9, [x9, :gottprel_g1:counter]\n{COUNTER}"
                ),
            );
            assert!(
                message.contains("Unsupported relocation modifier"),
                "{message}"
            );
        }

        #[test]
        fn gottprel_load_refuses_32_bit_registers() {
            let message = refusal(
                "gottprel_w",
                &format!(
                    ".text\n\
                     f:\n\
                     ldr w9, [x9, :gottprel_lo12:counter]\n{COUNTER}"
                ),
            );
            assert!(
                message.contains("Relocated ldr takes 64-bit registers"),
                "{message}"
            );
        }

        #[test]
        fn elf_tls_relocations_are_refused_for_macho() {
            let message = try_assemble(
                "macho",
                TargetOperatingSystem::MacOS,
                ".text\n\
                 _f:\n\
                 add x9, x9, :tprel_hi12:_counter, lsl #12\n",
            )
            .unwrap_err();
            assert!(
                message.contains("R_AARCH64_TLSLE_ADD_TPREL_HI12")
                    && message.contains("has no Mach-O equivalent"),
                "{message}"
            );
        }

        #[test]
        fn elf_tls_relocations_are_refused_for_coff() {
            let message = try_assemble(
                "coff",
                TargetOperatingSystem::Windows,
                ".text\n\
                 f:\n\
                 adrp x9, :gottprel:counter\n",
            )
            .unwrap_err();
            assert!(
                message.contains("R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21")
                    && message.contains("no ARM64 COFF relocation"),
                "{message}"
            );
        }
    }

    /// Run MIR through the AArch64 JIT on the host and check what the code returns.
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
    #[cfg(feature = "encoder")]
    mod jit_run {
        use crate::{ExecutableMemory, RasAssembler};
        use lamina_mir::{
            block::Block,
            function::{Function, Parameter, Signature},
            instruction::{
                AddressMode, AtomicBinOp, FloatBinOp, FloatCmpOp, FloatUnOp, Immediate,
                Instruction, IntBinOp, IntCmpOp, IntOverflowOp, IntUnOp, MemoryAttrs,
                MemoryOrdering, Operand, VectorOp,
            },
            module::{Global, Module},
            register::{Register, VirtualReg},
            types::{MirType, ScalarType, VectorLane, VectorType},
        };
        use lamina_platform::{PointerWidth, TargetArchitecture, TargetOperatingSystem};

        const I64: MirType = MirType::Scalar(ScalarType::I64);

        #[cfg(target_os = "macos")]
        const HOST_OS: TargetOperatingSystem = TargetOperatingSystem::MacOS;
        #[cfg(target_os = "linux")]
        const HOST_OS: TargetOperatingSystem = TargetOperatingSystem::Linux;

        fn v(id: u32) -> Register {
            Register::Virtual(VirtualReg::gpr(id))
        }

        fn imm(value: i64) -> Operand {
            Operand::Immediate(Immediate::I64(value))
        }

        fn reg(id: u32) -> Operand {
            Operand::Register(v(id))
        }

        /// Registers sort by name in the frame, so these push id 99 past the short STUR range.
        fn fill_frame(block: &mut Block, ids: std::ops::Range<u32>) {
            for id in ids {
                block.push(Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: I64,
                    dst: v(id),
                    lhs: imm(i64::from(id)),
                    rhs: imm(0),
                });
            }
        }

        fn run(module: &Module, entry: &str) -> i64 {
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            let (code, offsets) = assembler
                .compile_mir_to_binary_function(module, None)
                .expect("compile");
            let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
            memory.write_code(&code).expect("write");
            memory.make_executable().expect("protect");
            let start = unsafe { memory.code_start().add(offsets[entry]) };
            let function: unsafe extern "C" fn() -> i64 = unsafe { std::mem::transmute(start) };
            unsafe { function() }
        }

        fn at_v1(offset: i16) -> AddressMode {
            AddressMode::BaseOffset {
                address_space: Default::default(),
                base: v(1),
                offset,
            }
        }

        fn module_of(functions: Vec<Function>) -> Module {
            let mut module = Module::new("jit_run");
            for function in functions {
                module.add_function(function);
            }
            module
        }

        #[test]
        fn select_keeps_false_value_when_cond_slot_is_far() {
            let mut entry = Block::new("entry");
            fill_frame(&mut entry, 0..98);
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(98),
                lhs: imm(1234),
                rhs: imm(0),
            });
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(99),
                lhs: imm(0),
                rhs: imm(0),
            });
            entry.push(Instruction::Select {
                ty: I64,
                dst: v(100),
                cond: v(99),
                true_val: imm(7),
                false_val: reg(98),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(100)),
            });
            let mut function = Function::new(Signature::new("pick").with_return(I64));
            function.add_block(entry);
            assert_eq!(run(&module_of(vec![function]), "pick"), 1234);
        }

        #[test]
        fn stack_argument_lands_in_far_slot() {
            // id 99 sorts last, so the tenth argument gets the deepest slot.
            let params: Vec<Parameter> = (200..209)
                .chain([99])
                .map(|id| Parameter::new(v(id), I64))
                .collect();
            let mut callee_entry = Block::new("entry");
            fill_frame(&mut callee_entry, 0..99);
            callee_entry.push(Instruction::Ret {
                value: Some(reg(99)),
            });
            let mut callee = Function::new(
                Signature::new("callee")
                    .with_params(params)
                    .with_return(I64),
            );
            callee.add_block(callee_entry);

            let mut caller_entry = Block::new("entry");
            caller_entry.push(Instruction::Call {
                name: "callee".to_string(),
                args: (0..10).map(|n| imm(n + 500)).collect(),
                ret: Some(v(0)),
                ret_ty: Some(MirType::Scalar(ScalarType::I64)),
            });
            caller_entry.push(Instruction::Ret {
                value: Some(reg(0)),
            });
            let mut caller = Function::new(Signature::new("caller").with_return(I64));
            caller.add_block(caller_entry);

            assert_eq!(run(&module_of(vec![callee, caller]), "caller"), 509);
        }

        #[test]
        fn frame_past_imm12_keeps_values() {
            let mut entry = Block::new("entry");
            fill_frame(&mut entry, 0..800);
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(800),
                lhs: reg(799),
                rhs: reg(99),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(800)),
            });
            let mut function = Function::new(Signature::new("big").with_return(I64));
            function.add_block(entry);
            assert_eq!(run(&module_of(vec![function]), "big"), 898);
        }

        #[test]
        fn copy_and_bitcast_move_the_value() {
            let mut entry = Block::new("entry");
            entry.push(Instruction::Copy {
                ty: I64,
                dst: v(0),
                src: imm(41),
            });
            entry.push(Instruction::Copy {
                ty: I64,
                dst: v(1),
                src: reg(0),
            });
            entry.push(Instruction::Bitcast {
                from: I64,
                to: I64,
                dst: v(2),
                src: reg(1),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(2)),
            });
            let mut function = Function::new(Signature::new("copy").with_return(I64));
            function.add_block(entry);
            assert_eq!(run(&module_of(vec![function]), "copy"), 41);
        }

        #[test]
        fn stack_addr_reaches_reserved_block() {
            let slot = VirtualReg::gpr(0);
            let mut entry = Block::new("entry");
            entry.push(Instruction::StackAddr { dst: v(1), slot });
            for (offset, value) in [(0i16, 5i64), (24, 37)] {
                entry.push(Instruction::Store {
                    ty: I64,
                    src: imm(value),
                    addr: at_v1(offset),
                    attrs: MemoryAttrs::default(),
                });
            }
            entry.push(Instruction::Load {
                ty: I64,
                dst: v(2),
                addr: at_v1(0),
                attrs: MemoryAttrs::default(),
            });
            entry.push(Instruction::Load {
                ty: I64,
                dst: v(3),
                addr: at_v1(24),
                attrs: MemoryAttrs::default(),
            });
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(4),
                lhs: reg(2),
                rhs: reg(3),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(4)),
            });
            let mut function = Function::new(Signature::new("block").with_return(I64));
            function.stack_reservations.insert(slot, 3);
            function.add_block(entry);
            assert_eq!(run(&module_of(vec![function]), "block"), 42);
        }

        fn convert(from: ScalarType, to: ScalarType, signed: bool, value: i64) -> i64 {
            let mut entry = Block::new("entry");
            entry.push(Instruction::Copy {
                ty: I64,
                dst: v(0),
                src: imm(value),
            });
            entry.push(Instruction::IntConvert {
                from: MirType::Scalar(from),
                to: MirType::Scalar(to),
                signed,
                dst: v(1),
                src: reg(0),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(1)),
            });
            let mut function = Function::new(Signature::new("convert").with_return(I64));
            function.add_block(entry);
            run(&module_of(vec![function]), "convert")
        }

        #[test]
        fn int_convert_extends_and_truncates() {
            assert_eq!(convert(ScalarType::I8, ScalarType::I64, true, 0xFF), -1);
            assert_eq!(convert(ScalarType::I8, ScalarType::I64, false, 0xFF), 0xFF);
            assert_eq!(
                convert(ScalarType::I32, ScalarType::I64, true, 0xFFFF_FFFF),
                -1
            );
            assert_eq!(
                convert(ScalarType::I32, ScalarType::I64, false, 0xFFFF_FFFF),
                0xFFFF_FFFF
            );
            assert_eq!(convert(ScalarType::I64, ScalarType::I8, true, 0x1_7F), 0x7F);
            assert_eq!(
                convert(ScalarType::I64, ScalarType::I8, false, 0x1_80),
                -128
            );
        }

        #[cfg(target_os = "macos")]
        #[test]
        fn writebyte_reports_one_byte_written() {
            let mut entry = Block::new("entry");
            entry.push(Instruction::Call {
                name: "lamina.writebyte".to_string(),
                args: vec![imm(10)],
                ret: Some(v(0)),
                ret_ty: Some(MirType::Scalar(ScalarType::I64)),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(0)),
            });
            let mut function = Function::new(Signature::new("write").with_return(I64));
            function.add_block(entry);
            assert_eq!(run(&module_of(vec![function]), "write"), 1);
        }

        #[test]
        fn aggregate_abi_is_refused_not_miscompiled() {
            let mut entry = Block::new("entry");
            entry.push(Instruction::StackAddr {
                dst: v(1),
                slot: VirtualReg::gpr(0),
            });
            entry.push(Instruction::Ret {
                value: Some(imm(0)),
            });
            let mut function = Function::new(
                Signature::new("wide")
                    .with_params(vec![Parameter::new(v(0), I64)])
                    .with_return(I64),
            );
            function.add_block(entry);
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            let error = assembler
                .compile_mir_to_binary_function(&module_of(vec![function]), None)
                .expect_err("refused");
            assert!(error.to_string().contains("held in memory"), "{error}");
        }

        #[test]
        fn extern_declaration_calls_the_loaded_symbol() {
            let mut labs = Function::new(
                Signature::new("labs")
                    .with_params(vec![Parameter::new(v(0), I64)])
                    .with_return(I64),
            );
            labs.blocks.clear();
            let mut entry = Block::new("entry");
            entry.push(Instruction::Call {
                name: "labs".to_string(),
                args: vec![imm(-42)],
                ret: Some(v(0)),
                ret_ty: Some(MirType::Scalar(ScalarType::I64)),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(0)),
            });
            let mut main = Function::new(Signature::new("main").with_return(I64));
            main.add_block(entry);
            assert_eq!(run(&module_of(vec![labs, main]), "main"), 42);
        }

        #[test]
        fn extern_narrow_return_ignores_upper_bits() {
            let mut abs = Function::new(
                Signature::new("abs")
                    .with_params(vec![Parameter::new(v(0), MirType::Scalar(ScalarType::I32))])
                    .with_return(MirType::Scalar(ScalarType::I32)),
            );
            abs.blocks.clear();
            let mut entry = Block::new("entry");
            entry.push(Instruction::Call {
                name: "abs".to_string(),
                args: vec![imm(-7)],
                ret: Some(v(0)),
                ret_ty: Some(MirType::Scalar(ScalarType::I32)),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(0)),
            });
            let mut main = Function::new(Signature::new("main").with_return(I64));
            main.add_block(entry);
            assert_eq!(run(&module_of(vec![abs, main]), "main"), 7);
        }

        #[test]
        fn global_reads_initializer_and_keeps_stores() {
            let mut entry = Block::new("entry");
            entry.push(Instruction::SymbolAddress {
                dst: v(1),
                symbol: "counter".to_string(),
            });
            entry.push(Instruction::Load {
                ty: I64,
                dst: v(2),
                addr: at_v1(0),
                attrs: MemoryAttrs::default(),
            });
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(3),
                lhs: reg(2),
                rhs: imm(2),
            });
            entry.push(Instruction::Store {
                ty: I64,
                src: reg(3),
                addr: at_v1(0),
                attrs: MemoryAttrs::default(),
            });
            entry.push(Instruction::Load {
                ty: I64,
                dst: v(4),
                addr: at_v1(0),
                attrs: MemoryAttrs::default(),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(4)),
            });
            let mut function = Function::new(Signature::new("bump").with_return(I64));
            function.add_block(entry);
            let mut module = module_of(vec![function]);
            let global = Global::new("counter", I64, PointerWidth::Bits64)
                .with_initializer(40i64.to_le_bytes().to_vec());
            module.globals.insert("counter".to_string(), global);
            assert_eq!(run(&module, "bump"), 42);
        }

        fn switch_on(value: i64) -> i64 {
            let mut entry = Block::new("entry");
            entry.push(Instruction::Copy {
                ty: I64,
                dst: v(0),
                src: imm(value),
            });
            entry.push(Instruction::Switch {
                value: v(0),
                cases: vec![(1, "one".to_string()), (-70000, "far".to_string())],
                default: "other".to_string(),
            });
            let mut function = Function::new(Signature::new("switch").with_return(I64));
            function.add_block(entry);
            for (label, result) in [("one", 10), ("far", 20), ("other", 30)] {
                let mut block = Block::new(label);
                block.push(Instruction::Ret {
                    value: Some(imm(result)),
                });
                function.add_block(block);
            }
            run(&module_of(vec![function]), "switch")
        }

        #[test]
        fn switch_takes_matching_case_or_default() {
            assert_eq!(switch_on(1), 10);
            assert_eq!(switch_on(-70000), 20);
            assert_eq!(switch_on(5), 30);
        }

        fn looping_sum(name: &str, limit: i64) -> Function {
            let mut entry = Block::new("entry");
            entry.push(Instruction::Copy {
                ty: I64,
                dst: v(0),
                src: imm(0),
            });
            entry.push(Instruction::Copy {
                ty: I64,
                dst: v(1),
                src: imm(0),
            });
            entry.push(Instruction::Jmp {
                target: "head".to_string(),
            });
            let mut head = Block::new("head");
            head.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(0),
                lhs: reg(0),
                rhs: reg(1),
            });
            head.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(1),
                lhs: reg(1),
                rhs: imm(1),
            });
            head.push(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: I64,
                dst: v(2),
                lhs: reg(1),
                rhs: imm(limit),
            });
            head.push(Instruction::Br {
                cond: v(2),
                true_target: "head".to_string(),
                false_target: "done".to_string(),
            });
            let mut done = Block::new("done");
            done.push(Instruction::Ret {
                value: Some(reg(0)),
            });
            let mut function = Function::new(Signature::new(name).with_return(I64));
            function.add_block(entry);
            function.add_block(head);
            function.add_block(done);
            function
        }

        #[test]
        fn hot_values_survive_a_call_that_uses_the_same_registers() {
            let mut entry = Block::new("entry");
            entry.push(Instruction::Copy {
                ty: I64,
                dst: v(0),
                src: imm(1000),
            });
            entry.push(Instruction::Jmp {
                target: "again".to_string(),
            });
            let mut again = Block::new("again");
            again.push(Instruction::Call {
                name: "inner".to_string(),
                args: Vec::new(),
                ret: Some(v(1)),
                ret_ty: Some(MirType::Scalar(ScalarType::I64)),
            });
            again.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(0),
                lhs: reg(0),
                rhs: reg(1),
            });
            again.push(Instruction::IntCmp {
                op: IntCmpOp::SLt,
                ty: I64,
                dst: v(2),
                lhs: reg(0),
                rhs: imm(1100),
            });
            again.push(Instruction::Br {
                cond: v(2),
                true_target: "again".to_string(),
                false_target: "out".to_string(),
            });
            let mut out = Block::new("out");
            out.push(Instruction::Ret {
                value: Some(reg(0)),
            });
            let mut outer = Function::new(Signature::new("outer").with_return(I64));
            outer.add_block(entry);
            outer.add_block(again);
            outer.add_block(out);
            // inner adds 0..5 = 10, so the loop lands on 1100
            let module = module_of(vec![outer, looping_sum("inner", 5)]);
            assert_eq!(run(&module, "outer"), 1100);
        }

        #[test]
        fn narrow_value_in_a_register_reads_at_its_width() {
            let mut entry = Block::new("entry");
            entry.push(Instruction::Copy {
                ty: I64,
                dst: v(0),
                src: imm(0x1_0000_0005),
            });
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I32),
                dst: v(1),
                lhs: reg(0),
                rhs: imm(-10),
            });
            entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: MirType::Scalar(ScalarType::I32),
                dst: v(2),
                lhs: reg(1),
                rhs: reg(1),
            });
            entry.push(Instruction::Ret {
                value: Some(reg(2)),
            });
            let mut function = Function::new(
                Signature::new("narrow").with_return(MirType::Scalar(ScalarType::I32)),
            );
            function.add_block(entry);
            assert_eq!(run(&module_of(vec![function]), "narrow") as i32, -10);
        }

        const F64_TY: MirType = MirType::Scalar(ScalarType::F64);
        const F32_TY: MirType = MirType::Scalar(ScalarType::F32);

        fn f64_imm(value: f64) -> Operand {
            Operand::Immediate(Immediate::F64(value))
        }

        /// Run a block that ends by returning a register, as its raw 64 bits.
        fn block_result(build: impl FnOnce(&mut Block)) -> i64 {
            let mut entry = Block::new("entry");
            build(&mut entry);
            let mut function = Function::new(Signature::new("result").with_return(I64));
            function.add_block(entry);
            run(&module_of(vec![function]), "result")
        }

        fn ret(entry: &mut Block, id: u32) {
            entry.push(Instruction::Ret {
                value: Some(reg(id)),
            });
        }

        #[test]
        fn float_arithmetic_and_constants() {
            let bits = block_result(|entry| {
                entry.push(Instruction::FloatBinary {
                    op: FloatBinOp::FMul,
                    ty: F64_TY,
                    dst: v(0),
                    lhs: f64_imm(1.5),
                    rhs: f64_imm(4.0),
                });
                entry.push(Instruction::FloatBinary {
                    op: FloatBinOp::FSub,
                    ty: F64_TY,
                    dst: v(1),
                    lhs: reg(0),
                    rhs: f64_imm(0.25),
                });
                entry.push(Instruction::FloatUnary {
                    op: FloatUnOp::FNeg,
                    ty: F64_TY,
                    dst: v(2),
                    src: reg(1),
                });
                ret(entry, 2);
            });
            assert_eq!(f64::from_bits(bits as u64), -5.75);
        }

        fn compares(op: FloatCmpOp, lhs: f64, rhs: f64) -> i64 {
            block_result(|entry| {
                entry.push(Instruction::FloatCmp {
                    op,
                    ty: F64_TY,
                    dst: v(0),
                    lhs: f64_imm(lhs),
                    rhs: f64_imm(rhs),
                });
                ret(entry, 0);
            })
        }

        #[test]
        fn float_compares_are_false_on_nan_except_ne() {
            for (op, less_than) in [
                (FloatCmpOp::Eq, 0),
                (FloatCmpOp::Ne, 1),
                (FloatCmpOp::Lt, 1),
                (FloatCmpOp::Le, 1),
                (FloatCmpOp::Gt, 0),
                (FloatCmpOp::Ge, 0),
            ] {
                assert_eq!(compares(op, 1.0, 2.0), less_than, "{op:?}");
                let on_nan = i64::from(matches!(op, FloatCmpOp::Ne));
                assert_eq!(compares(op, f64::NAN, 2.0), on_nan, "{op:?} nan");
            }
        }

        fn to_int(to: ScalarType, signed: bool, value: f64) -> i64 {
            block_result(|entry| {
                entry.push(Instruction::FloatToInt {
                    from: F64_TY,
                    to: MirType::Scalar(to),
                    signed,
                    dst: v(0),
                    src: f64_imm(value),
                });
                ret(entry, 0);
            })
        }

        #[test]
        fn float_to_int_truncates_and_saturates() {
            assert_eq!(to_int(ScalarType::I64, true, -2.9), -2);
            assert_eq!(to_int(ScalarType::I64, true, f64::NAN), 0);
            assert_eq!(to_int(ScalarType::I8, true, 1000.0), 127);
            assert_eq!(to_int(ScalarType::I8, true, -1000.0), -128);
            assert_eq!(to_int(ScalarType::I8, false, 1000.0), 255);
            assert_eq!(to_int(ScalarType::I32, true, 3e10), i64::from(i32::MAX));
        }

        #[test]
        fn int_to_float_and_back_through_f32() {
            let bits = block_result(|entry| {
                entry.push(Instruction::IntToFloat {
                    from: I64,
                    to: F32_TY,
                    signed: true,
                    dst: v(0),
                    src: imm(-3),
                });
                entry.push(Instruction::FloatConvert {
                    from: F32_TY,
                    to: F64_TY,
                    dst: v(1),
                    src: reg(0),
                });
                ret(entry, 1);
            });
            assert_eq!(f64::from_bits(bits as u64), -3.0);
        }

        fn narrow_op(op: IntBinOp, ty: ScalarType, lhs: i64, rhs: i64) -> i64 {
            let ty = MirType::Scalar(ty);
            block_result(|entry| {
                let address = VirtualReg::gpr(50);
                entry.push(Instruction::StackAddr {
                    dst: v(1),
                    slot: address,
                });
                entry.push(Instruction::Store {
                    ty,
                    src: imm(lhs),
                    addr: at_v1(0),
                    attrs: MemoryAttrs::default(),
                });
                entry.push(Instruction::Load {
                    ty,
                    dst: v(2),
                    addr: at_v1(0),
                    attrs: MemoryAttrs::default(),
                });
                entry.push(Instruction::IntBinary {
                    op,
                    ty,
                    dst: v(3),
                    lhs: reg(2),
                    rhs: imm(rhs),
                });
                entry.push(Instruction::IntConvert {
                    from: ty,
                    to: I64,
                    signed: false,
                    dst: v(4),
                    src: reg(3),
                });
                ret(entry, 4);
            })
        }

        #[test]
        fn narrow_unsigned_division_reads_the_type_bits() {
            assert_eq!(narrow_op(IntBinOp::UDiv, ScalarType::I16, 50000, 2), 25000);
            assert_eq!(
                narrow_op(IntBinOp::URem, ScalarType::I16, 50000, 7),
                50000 % 7
            );
            assert_eq!(narrow_op(IntBinOp::UDiv, ScalarType::I8, 200, 3), 66);
        }

        #[test]
        fn narrow_shifts_read_the_type_bits_and_mask_the_amount() {
            assert_eq!(
                narrow_op(IntBinOp::LShr, ScalarType::I16, 0xF000, 4),
                0x0F00
            );
            assert_eq!(narrow_op(IntBinOp::LShr, ScalarType::I8, 0x80, 9), 0x40);
            assert_eq!(narrow_op(IntBinOp::Shl, ScalarType::I8, 1, 9), 2);
            assert_eq!(narrow_op(IntBinOp::AShr, ScalarType::I8, 0x80, 1), 0xC0);
        }

        #[test]
        fn bool_arithmetic_is_byte_wide_like_the_aot_backend() {
            assert_eq!(narrow_op(IntBinOp::Add, ScalarType::I1, 1, 1), 2);
        }

        #[test]
        fn extern_float_goes_through_float_registers() {
            let mut ldexp = Function::new(
                Signature::new("ldexp")
                    .with_params(vec![
                        Parameter::new(v(0), F64_TY),
                        Parameter::new(v(1), MirType::Scalar(ScalarType::I32)),
                    ])
                    .with_return(F64_TY),
            );
            ldexp.blocks.clear();
            let mut entry = Block::new("entry");
            entry.push(Instruction::Call {
                name: "ldexp".to_string(),
                args: vec![f64_imm(1.5), imm(3)],
                ret: Some(v(0)),
                ret_ty: Some(MirType::Scalar(ScalarType::F64)),
            });
            ret(&mut entry, 0);
            let mut main = Function::new(Signature::new("main").with_return(I64));
            main.add_block(entry);
            let bits = run(&module_of(vec![ldexp, main]), "main");
            assert_eq!(f64::from_bits(bits as u64), 12.0);
        }

        #[cfg(target_os = "macos")]
        #[test]
        fn variadic_arguments_go_on_the_stack_on_darwin() {
            let mut snprintf = Function::new(
                Signature::new("snprintf")
                    .with_params(vec![
                        Parameter::new(v(0), MirType::Scalar(ScalarType::Ptr)),
                        Parameter::new(v(1), I64),
                        Parameter::new(v(2), MirType::Scalar(ScalarType::Ptr)),
                    ])
                    .with_return(MirType::Scalar(ScalarType::I32)),
            );
            snprintf.sig.variadic = true;
            snprintf.blocks.clear();

            let mut entry = Block::new("entry");
            entry.push(Instruction::StackAddr {
                dst: v(1),
                slot: VirtualReg::gpr(50),
            });
            entry.push(Instruction::SymbolAddress {
                dst: v(2),
                symbol: "format".to_string(),
            });
            entry.push(Instruction::Call {
                name: "snprintf".to_string(),
                args: vec![reg(1), imm(32), reg(2), imm(123), imm(-45)],
                ret: Some(v(3)),
                ret_ty: Some(MirType::Scalar(ScalarType::I32)),
            });
            ret(&mut entry, 3);
            let mut main = Function::new(Signature::new("main").with_return(I64));
            main.stack_reservations.insert(VirtualReg::gpr(50), 4);
            main.add_block(entry);
            let mut module = module_of(vec![snprintf, main]);
            let format = Global::new("format", I64, PointerWidth::Bits64)
                .with_layout(10, 1)
                .with_initializer(b"%lld|%lld\0".to_vec());
            module.globals.insert("format".to_string(), format);
            // "123|-45"
            assert_eq!(run(&module, "main"), 7);
        }

        fn two_functions_with_pointer_call() -> Module {
            let mut seven = Function::new(
                Signature::new("seven")
                    .with_params(vec![Parameter::new(v(0), I64)])
                    .with_return(I64),
            );
            let mut seven_entry = Block::new("entry");
            seven_entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(1),
                lhs: reg(0),
                rhs: imm(7),
            });
            ret(&mut seven_entry, 1);
            seven.add_block(seven_entry);

            let mut entry = Block::new("entry");
            entry.push(Instruction::SymbolAddress {
                dst: v(0),
                symbol: "seven".to_string(),
            });
            entry.push(Instruction::CallIndirect {
                target: reg(0),
                arg_types: vec![I64],
                args: vec![imm(35)],
                variadic_fixed: None,
                ret: Some(v(1)),
                ret_ty: Some(I64),
                ret_aggregate: None,
                arg_aggregates: vec![None],
            });
            ret(&mut entry, 1);
            let mut main = Function::new(Signature::new("main").with_return(I64));
            main.add_block(entry);
            module_of(vec![seven, main])
        }

        #[test]
        fn function_address_and_indirect_call() {
            assert_eq!(run(&two_functions_with_pointer_call(), "main"), 42);
        }

        /// Compiles every function in `functions` into one block and returns what each one of the
        /// `names` returns, in order.
        fn run_named(functions: Vec<Function>, names: &[String]) -> Vec<i64> {
            let module = module_of(functions);
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            let (code, offsets) = assembler
                .compile_mir_to_binary_function(&module, None)
                .expect("compile");
            let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
            memory.write_code(&code).expect("write");
            memory.make_executable().expect("protect");
            names
                .iter()
                .map(|name| {
                    let start = unsafe { memory.code_start().add(offsets[name.as_str()]) };
                    let function: unsafe extern "C" fn() -> i64 =
                        unsafe { std::mem::transmute(start) };
                    unsafe { function() }
                })
                .collect()
        }

        /// Stores `value` at `ty`, reads it back, applies `op` and widens the result to i64.
        fn unary_function(name: &str, op: IntUnOp, ty: ScalarType, value: i64) -> Function {
            let ty = MirType::Scalar(ty);
            let mut entry = Block::new("entry");
            entry.push(Instruction::StackAddr {
                dst: v(1),
                slot: VirtualReg::gpr(50),
            });
            entry.push(Instruction::Store {
                ty,
                src: imm(value),
                addr: at_v1(0),
                attrs: MemoryAttrs::default(),
            });
            entry.push(Instruction::Load {
                ty,
                dst: v(2),
                addr: at_v1(0),
                attrs: MemoryAttrs::default(),
            });
            entry.push(Instruction::IntUnary {
                op,
                ty,
                dst: v(3),
                src: reg(2),
            });
            entry.push(Instruction::IntConvert {
                from: ty,
                to: I64,
                signed: true,
                dst: v(4),
                src: reg(3),
            });
            ret(&mut entry, 4);
            let mut function = Function::new(Signature::new(name).with_return(I64));
            function.add_block(entry);
            function
        }

        fn unary_result(op: IntUnOp, ty: ScalarType, value: i64) -> i64 {
            let name = "unary".to_string();
            run_named(vec![unary_function(&name, op, ty, value)], &[name])[0]
        }

        #[test]
        fn int_unary_matches_mir_evaluation_at_every_width() {
            const EDGES: [i64; 22] = [
                0,
                1,
                -1,
                2,
                3,
                0x7F,
                -0x80,
                0xFF,
                0x100,
                0x0102,
                0x8000,
                -0x8000,
                0x7FFF,
                0x0102_0304,
                -0x8000_0000,
                0x7FFF_FFFF,
                0x8000_0001,
                0x0123_4567_89AB_CDEF,
                i64::MIN,
                i64::MAX,
                0x40,
                0xF0F0,
            ];
            let widths = [
                ScalarType::I8,
                ScalarType::I16,
                ScalarType::I32,
                ScalarType::I64,
            ];
            let ops = [
                IntUnOp::Popcount,
                IntUnOp::Ctlz,
                IntUnOp::Cttz,
                IntUnOp::Bswap,
            ];
            let mut functions = Vec::new();
            let mut expected = Vec::new();
            let mut names = Vec::new();
            for op in ops {
                for ty in widths {
                    for value in EDGES {
                        let name = format!("u{}", functions.len());
                        functions.push(unary_function(&name, op, ty, value));
                        expected.push((
                            op,
                            ty,
                            value,
                            IntUnOp::evaluate(op, MirType::Scalar(ty), value, PointerWidth::Bits64),
                        ));
                        names.push(name);
                    }
                }
            }
            let got = run_named(functions, &names);
            for (index, (op, ty, value, want)) in expected.iter().enumerate() {
                assert_eq!(got[index], *want, "{op} {ty:?} of {value:#x}");
            }
        }

        #[test]
        fn int_unary_hand_checked_edges() {
            assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I8, 0), 8);
            assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I8, 1), 7);
            assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I16, 0), 16);
            assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I32, -1), 0);
            assert_eq!(unary_result(IntUnOp::Ctlz, ScalarType::I64, 0), 64);
            assert_eq!(unary_result(IntUnOp::Cttz, ScalarType::I8, 0), 8);
            assert_eq!(unary_result(IntUnOp::Cttz, ScalarType::I16, 0x8000), 15);
            assert_eq!(unary_result(IntUnOp::Cttz, ScalarType::I64, 0), 64);
            assert_eq!(unary_result(IntUnOp::Popcount, ScalarType::I8, -1), 8);
            assert_eq!(unary_result(IntUnOp::Popcount, ScalarType::I64, -1), 64);
            assert_eq!(
                unary_result(IntUnOp::Popcount, ScalarType::I64, i64::MIN),
                1
            );
            assert_eq!(
                unary_result(IntUnOp::Bswap, ScalarType::I16, 0x0102),
                0x0201
            );
            assert_eq!(
                unary_result(IntUnOp::Bswap, ScalarType::I32, 0x0102_0304),
                0x0403_0201
            );
            assert_eq!(unary_result(IntUnOp::Bswap, ScalarType::I8, -128), -128);
            assert_eq!(
                unary_result(IntUnOp::Bswap, ScalarType::I64, 0x0102_0304_0506_0708),
                0x0807_0605_0403_0201
            );
        }

        fn typed_imm(ty: ScalarType, value: i64) -> Operand {
            Operand::Immediate(match ty {
                ScalarType::I8 => Immediate::I8(value as i8),
                ScalarType::I16 => Immediate::I16(value as i16),
                ScalarType::I32 => Immediate::I32(value as i32),
                _ => Immediate::I64(value),
            })
        }

        /// Reads `lhs` and `rhs` at `ty` (from memory, or `rhs` as an immediate) and returns the
        /// IntOverflow flag widened to i64.
        fn overflow_function(
            name: &str,
            op: IntOverflowOp,
            ty: ScalarType,
            signed: bool,
            lhs: i64,
            rhs: i64,
            rhs_immediate: bool,
        ) -> Function {
            let mir_ty = MirType::Scalar(ty);
            let slot = VirtualReg::gpr(50);
            let mut entry = Block::new("entry");
            entry.push(Instruction::StackAddr { dst: v(1), slot });
            entry.push(Instruction::Store {
                ty: mir_ty,
                src: typed_imm(ty, lhs),
                addr: at_v1(0),
                attrs: MemoryAttrs::default(),
            });
            entry.push(Instruction::Load {
                ty: mir_ty,
                dst: v(2),
                addr: at_v1(0),
                attrs: MemoryAttrs::default(),
            });
            let rhs_operand = if rhs_immediate {
                typed_imm(ty, rhs)
            } else {
                entry.push(Instruction::Store {
                    ty: mir_ty,
                    src: typed_imm(ty, rhs),
                    addr: at_v1(8),
                    attrs: MemoryAttrs::default(),
                });
                entry.push(Instruction::Load {
                    ty: mir_ty,
                    dst: v(5),
                    addr: at_v1(8),
                    attrs: MemoryAttrs::default(),
                });
                reg(5)
            };
            entry.push(Instruction::IntOverflow {
                op,
                ty: mir_ty,
                signed,
                dst: v(3),
                lhs: reg(2),
                rhs: rhs_operand,
            });
            entry.push(Instruction::IntConvert {
                from: MirType::Scalar(ScalarType::I1),
                to: I64,
                signed: false,
                dst: v(4),
                src: reg(3),
            });
            ret(&mut entry, 4);
            let mut function = Function::new(Signature::new(name).with_return(I64));
            function.stack_reservations.insert(slot, 2);
            function.add_block(entry);
            function
        }

        fn overflow_result(
            op: IntOverflowOp,
            ty: ScalarType,
            signed: bool,
            lhs: i64,
            rhs: i64,
        ) -> i64 {
            let name = "overflow".to_string();
            let function = overflow_function(&name, op, ty, signed, lhs, rhs, true);
            run_named(vec![function], &[name])[0]
        }

        #[test]
        fn int_overflow_matches_mir_at_every_width_and_signedness() {
            let edges: [(ScalarType, [i64; 10]); 4] = [
                (ScalarType::I8, [-128, -127, -2, -1, 0, 1, 2, 63, 64, 127]),
                (
                    ScalarType::I16,
                    [
                        i64::from(i16::MIN),
                        -256,
                        -2,
                        -1,
                        0,
                        1,
                        255,
                        256,
                        0x4000,
                        i64::from(i16::MAX),
                    ],
                ),
                (
                    ScalarType::I32,
                    [
                        i64::from(i32::MIN),
                        -65536,
                        -2,
                        -1,
                        0,
                        1,
                        65536,
                        0x7FFF,
                        0x4000_0000,
                        i64::from(i32::MAX),
                    ],
                ),
                (
                    ScalarType::I64,
                    [
                        i64::MIN,
                        i64::MIN + 1,
                        -2,
                        -1,
                        0,
                        1,
                        2,
                        0x1_0000_0000,
                        0x4000_0000_0000_0000,
                        i64::MAX,
                    ],
                ),
            ];
            let ops = [IntOverflowOp::Add, IntOverflowOp::Sub, IntOverflowOp::Mul];
            let mut functions = Vec::new();
            let mut names = Vec::new();
            let mut expected = Vec::new();
            for (ty, values) in edges {
                for op in ops {
                    for signed in [true, false] {
                        for (i, &lhs) in values.iter().enumerate() {
                            for (j, &rhs) in values.iter().enumerate() {
                                let name = format!("o{}", functions.len());
                                functions.push(overflow_function(
                                    &name,
                                    op,
                                    ty,
                                    signed,
                                    lhs,
                                    rhs,
                                    (i + j) % 2 == 0,
                                ));
                                let want = op.overflows(
                                    MirType::Scalar(ty),
                                    signed,
                                    lhs,
                                    rhs,
                                    PointerWidth::Bits64,
                                );
                                expected.push((ty, op, signed, lhs, rhs, i64::from(want)));
                                names.push(name);
                            }
                        }
                    }
                }
            }
            let got = run_named(functions, &names);
            for (index, (ty, op, signed, lhs, rhs, want)) in expected.iter().enumerate() {
                assert_eq!(got[index], *want, "{op} {ty:?} signed={signed} {lhs} {rhs}");
            }
        }

        #[test]
        fn int_overflow_hand_checked_boundaries() {
            use IntOverflowOp::{Add, Mul, Sub};
            use ScalarType::{I8, I16, I32, I64};
            // (op, width, signed, lhs, rhs, overflows)
            let cases = [
                (Add, I8, true, 127, 1, 1),
                (Add, I8, false, 127, 1, 0),
                (Add, I8, false, 255, 1, 1),
                (Add, I8, false, 254, 1, 0),
                (Sub, I8, true, -128, 1, 1),
                (Sub, I8, false, 0, 1, 1),
                (Mul, I8, false, -1, 2, 1),
                (Mul, I8, true, -128, -1, 1),
                (Mul, I16, true, -1, i64::from(i16::MIN), 1),
                (Add, I32, false, 0xFFFF_FFFF, 1, 1),
                (Mul, I32, true, 65536, 65536, 1),
                (Mul, I32, false, 65536, 65535, 0),
                (Sub, I64, true, 0, i64::MIN, 1),
                (Sub, I64, false, 0, i64::MIN, 1),
                (Sub, I64, false, 0, -1, 1),
                (Sub, I64, true, i64::MIN, 0, 0),
                (Add, I64, true, i64::MAX, 1, 1),
                (Add, I64, false, -1, 1, 1),
                (Add, I64, false, 1, 1, 0),
                (Mul, I64, true, i64::MIN, 1, 0),
                (Mul, I64, true, -1, i64::MIN, 1),
                (Mul, I64, true, 0x1_0000_0000, 0x7FFF_FFFF, 0),
                (Mul, I64, false, i64::MAX, 2, 0),
                (Mul, I64, false, i64::MAX, 3, 1),
                (Mul, I64, false, -1, 1, 0),
                (Mul, I64, false, 0x1_0000_0000, 0x1_0000_0000, 1),
            ];
            for (op, ty, signed, lhs, rhs, want) in cases {
                assert_eq!(
                    overflow_result(op, ty, signed, lhs, rhs),
                    want,
                    "{op} {ty:?} signed={signed} {lhs} {rhs}"
                );
            }
        }

        const ALL_WIDTHS: [ScalarType; 4] = [
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
        ];

        /// The value a `ty` holds after storing `raw`, read back sign-extended.
        fn canonical(ty: ScalarType, raw: i64) -> i64 {
            MirType::Scalar(ty).signed_integer(raw, PointerWidth::Bits64)
        }

        fn mask_of(ty: ScalarType) -> u64 {
            match ty {
                ScalarType::I8 => 0xFF,
                ScalarType::I16 => 0xFFFF,
                ScalarType::I32 => 0xFFFF_FFFF,
                _ => u64::MAX,
            }
        }

        fn slot_addr(dst: u32, slot: u32) -> Instruction {
            Instruction::StackAddr {
                dst: v(dst),
                slot: VirtualReg::gpr(slot),
            }
        }

        fn at_base(base: u32) -> AddressMode {
            AddressMode::BaseOffset {
                address_space: Default::default(),
                base: v(base),
                offset: 0,
            }
        }

        fn widen(entry: &mut Block, ty: ScalarType, src: u32, dst: u32) {
            entry.push(Instruction::IntConvert {
                from: MirType::Scalar(ty),
                to: I64,
                signed: true,
                dst: v(dst),
                src: reg(src),
            });
        }

        /// A function that makes its result from `body`, which runs after the slot is set up.
        fn atomic_function(name: &str, body: impl FnOnce(&mut Block)) -> Function {
            let mut entry = Block::new("entry");
            entry.push(slot_addr(1, 50));
            body(&mut entry);
            let mut function = Function::new(Signature::new(name).with_return(I64));
            function.stack_reservations.insert(VirtualReg::gpr(50), 1);
            function.add_block(entry);
            function
        }

        #[test]
        fn atomic_load_and_store_round_trip_every_width_and_ordering() {
            use MemoryOrdering::{Acquire, Relaxed, Release, SeqCst};
            let values: [i64; 6] = [0, 1, -1, 0x7F, -0x8000, 0x1234_5678_9ABC_DEF0];
            let orderings = [(Relaxed, Relaxed), (Release, Acquire), (SeqCst, SeqCst)];
            let mut functions = Vec::new();
            let mut names = Vec::new();
            let mut expected = Vec::new();
            for ty in ALL_WIDTHS {
                for (store_order, load_order) in orderings {
                    for (index, &value) in values.iter().enumerate() {
                        let name = format!("rt{}", functions.len());
                        let from_register = index % 2 == 1;
                        functions.push(atomic_function(&name, |entry| {
                            let stored = if from_register {
                                entry.push(Instruction::Copy {
                                    ty: I64,
                                    dst: v(5),
                                    src: imm(value),
                                });
                                reg(5)
                            } else {
                                typed_imm(ty, value)
                            };
                            entry.push(Instruction::AtomicStore {
                                ty: MirType::Scalar(ty),
                                src: stored,
                                addr: at_base(1),
                                ordering: store_order,
                            });
                            entry.push(Instruction::AtomicLoad {
                                ty: MirType::Scalar(ty),
                                dst: v(2),
                                addr: at_base(1),
                                ordering: load_order,
                            });
                            widen(entry, ty, 2, 3);
                            ret(entry, 3);
                        }));
                        names.push(name);
                        expected.push((ty, store_order, load_order, value, canonical(ty, value)));
                    }
                }
            }
            let got = run_named(functions, &names);
            for (index, (ty, store_order, load_order, value, want)) in expected.iter().enumerate() {
                assert_eq!(
                    got[index], *want,
                    "{ty:?} store {store_order:?} load {load_order:?} of {value:#x}"
                );
            }
        }

        fn atomic_binop(op: AtomicBinOp, a: i64, b: i64) -> i64 {
            match op {
                AtomicBinOp::Add => a.wrapping_add(b),
                AtomicBinOp::Sub => a.wrapping_sub(b),
                AtomicBinOp::And => a & b,
                AtomicBinOp::Or => a | b,
                AtomicBinOp::Xor => a ^ b,
                AtomicBinOp::Exchange => b,
            }
        }

        #[test]
        fn atomic_binary_returns_old_and_stores_new_at_every_width() {
            use AtomicBinOp::{Add, And, Exchange, Or, Sub, Xor};
            let ops = [Add, Sub, And, Or, Xor, Exchange];
            let values: [i64; 8] = [0, 1, -1, 0x7F, -0x80, 0x100, 0x7FFF_FFFF, i64::MIN];
            let mut functions = Vec::new();
            let mut names = Vec::new();
            let mut expected = Vec::new();
            for ty in ALL_WIDTHS {
                for op in ops {
                    for (i, &a) in values.iter().enumerate() {
                        for (j, &b) in values.iter().enumerate() {
                            // Old value and stored value are read by two functions over the same setup.
                            let build = |name: &str, read_old: bool| {
                                atomic_function(name, |entry| {
                                    entry.push(Instruction::AtomicStore {
                                        ty: MirType::Scalar(ty),
                                        src: typed_imm(ty, a),
                                        addr: at_base(1),
                                        ordering: MemoryOrdering::Relaxed,
                                    });
                                    let value = if (i + j) % 2 == 0 {
                                        typed_imm(ty, b)
                                    } else {
                                        entry.push(Instruction::Copy {
                                            ty: I64,
                                            dst: v(6),
                                            src: imm(b),
                                        });
                                        reg(6)
                                    };
                                    entry.push(Instruction::AtomicBinary {
                                        op,
                                        ty: MirType::Scalar(ty),
                                        dst: v(2),
                                        addr: at_base(1),
                                        value,
                                        ordering: MemoryOrdering::SeqCst,
                                    });
                                    if read_old {
                                        widen(entry, ty, 2, 3);
                                    } else {
                                        entry.push(Instruction::AtomicLoad {
                                            ty: MirType::Scalar(ty),
                                            dst: v(4),
                                            addr: at_base(1),
                                            ordering: MemoryOrdering::Acquire,
                                        });
                                        widen(entry, ty, 4, 3);
                                    }
                                    ret(entry, 3);
                                })
                            };
                            let old_name = format!("bin{}", functions.len());
                            functions.push(build(&old_name, true));
                            names.push(old_name);
                            let new_name = format!("bin{}", functions.len());
                            functions.push(build(&new_name, false));
                            names.push(new_name);
                            let result = atomic_binop(op, a, b);
                            expected.push((ty, op, a, b, canonical(ty, a), canonical(ty, result)));
                        }
                    }
                }
            }
            let got = run_named(functions, &names);
            for (index, (ty, op, a, b, old, new)) in expected.iter().enumerate() {
                assert_eq!(got[2 * index], *old, "{op} old {ty:?} {a:#x} {b:#x}");
                assert_eq!(got[2 * index + 1], *new, "{op} new {ty:?} {a:#x} {b:#x}");
            }
        }

        #[test]
        fn compare_exchange_swaps_only_on_a_width_match() {
            let values: [i64; 5] = [0, 1, -1, 0x7F, -0x80];
            let mut functions = Vec::new();
            let mut names = Vec::new();
            let mut expected = Vec::new();
            for ty in ALL_WIDTHS {
                for &current in &values {
                    for &wanted in &values {
                        // Narrow expected values also come with bits above the width set.
                        let expected_raw = if ty == ScalarType::I64 {
                            wanted
                        } else {
                            wanted ^ 0x1234_5600_0000_0000
                        };
                        let desired = 42;
                        let build = |name: &str, read_old: bool| {
                            atomic_function(name, |entry| {
                                entry.push(Instruction::AtomicStore {
                                    ty: MirType::Scalar(ty),
                                    src: typed_imm(ty, current),
                                    addr: at_base(1),
                                    ordering: MemoryOrdering::Relaxed,
                                });
                                entry.push(Instruction::Copy {
                                    ty: I64,
                                    dst: v(5),
                                    src: imm(expected_raw),
                                });
                                entry.push(Instruction::AtomicCompareExchange {
                                    ty: MirType::Scalar(ty),
                                    dst: v(2),
                                    addr: at_base(1),
                                    expected: reg(5),
                                    desired: typed_imm(ty, desired),
                                    success_ordering: MemoryOrdering::SeqCst,
                                    failure_ordering: MemoryOrdering::Acquire,
                                });
                                if read_old {
                                    widen(entry, ty, 2, 3);
                                } else {
                                    entry.push(Instruction::AtomicLoad {
                                        ty: MirType::Scalar(ty),
                                        dst: v(4),
                                        addr: at_base(1),
                                        ordering: MemoryOrdering::SeqCst,
                                    });
                                    widen(entry, ty, 4, 3);
                                }
                                ret(entry, 3);
                            })
                        };
                        let old_name = format!("cas{}", functions.len());
                        functions.push(build(&old_name, true));
                        names.push(old_name);
                        let new_name = format!("cas{}", functions.len());
                        functions.push(build(&new_name, false));
                        names.push(new_name);
                        let matched = (canonical(ty, current) as u64 & mask_of(ty))
                            == (wanted as u64 & mask_of(ty));
                        let after = if matched {
                            canonical(ty, desired)
                        } else {
                            canonical(ty, current)
                        };
                        expected.push((ty, current, wanted, canonical(ty, current), after));
                    }
                }
            }
            let got = run_named(functions, &names);
            for (index, (ty, current, wanted, old, after)) in expected.iter().enumerate() {
                assert_eq!(got[2 * index], *old, "cas old {ty:?} {current} {wanted}");
                assert_eq!(
                    got[2 * index + 1],
                    *after,
                    "cas new {ty:?} {current} {wanted}"
                );
            }
        }

        /// A function taking an address and adding one to the cell there, atomically.
        fn bump_function(name: &str, ty: ScalarType) -> Function {
            let mut entry = Block::new("entry");
            entry.push(Instruction::AtomicBinary {
                op: AtomicBinOp::Add,
                ty: MirType::Scalar(ty),
                dst: v(1),
                addr: at_base(0),
                value: typed_imm(ty, 1),
                ordering: MemoryOrdering::SeqCst,
            });
            ret(&mut entry, 1);
            let mut function = Function::new(
                Signature::new(name)
                    .with_params(vec![Parameter::new(v(0), I64)])
                    .with_return(I64),
            );
            function.add_block(entry);
            function
        }

        #[test]
        fn atomic_add_from_several_threads_loses_no_increment() {
            let module = module_of(vec![bump_function("bump64", ScalarType::I64)]);
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            let (code, offsets) = assembler
                .compile_mir_to_binary_function(&module, None)
                .expect("compile");
            let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
            memory.write_code(&code).expect("write");
            memory.make_executable().expect("protect");
            let base = memory.code_start() as usize;
            let wide = base + offsets["bump64"];
            let wide_fn: unsafe extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(wide) };
            let cell = Box::into_raw(Box::new(0i64)) as usize;
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(move || {
                        for _ in 0..2000 {
                            unsafe { wide_fn(cell as i64) };
                        }
                    });
                }
            });
            let total = unsafe { *Box::from_raw(cell as *mut i64) };
            assert_eq!(total, 8000);
        }

        #[test]
        fn narrow_atomic_add_leaves_neighbouring_bytes_alone() {
            let module = module_of(vec![bump_function("bump8", ScalarType::I8)]);
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            let (code, offsets) = assembler
                .compile_mir_to_binary_function(&module, None)
                .expect("compile");
            let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
            memory.write_code(&code).expect("write");
            memory.make_executable().expect("protect");
            let start = memory.code_start() as usize + offsets["bump8"];
            let bump: unsafe extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(start) };
            let cell = Box::into_raw(Box::new(0xAAAA_AAAA_AAAA_AA00u64)) as usize;
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(move || {
                        for _ in 0..1000 {
                            unsafe { bump(cell as i64) };
                        }
                    });
                }
            });
            let after = unsafe { *Box::from_raw(cell as *mut u64) };
            // 4000 increments of one byte wrap to 160 and touch nothing else.
            assert_eq!(after, 0xAAAA_AAAA_AAAA_AAA0);
        }

        #[test]
        fn atomic_orderings_the_mir_rejects_are_refused() {
            let refuse = |instruction: Instruction| {
                let mut entry = Block::new("entry");
                entry.push(slot_addr(1, 50));
                entry.push(instruction);
                entry.push(Instruction::Ret { value: None });
                let mut function = Function::new(Signature::new("bad").with_return(I64));
                function.stack_reservations.insert(VirtualReg::gpr(50), 1);
                function.add_block(entry);
                let module = module_of(vec![function]);
                let mut assembler =
                    RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
                assembler
                    .compile_mir_to_binary_function(&module, None)
                    .is_err()
            };
            let i64ty = MirType::Scalar(ScalarType::I64);
            assert!(refuse(Instruction::AtomicLoad {
                ty: i64ty,
                dst: v(2),
                addr: at_v1(0),
                ordering: MemoryOrdering::Release,
            }));
            assert!(refuse(Instruction::AtomicStore {
                ty: i64ty,
                src: imm(1),
                addr: at_v1(0),
                ordering: MemoryOrdering::Acquire,
            }));
            assert!(refuse(Instruction::AtomicCompareExchange {
                ty: i64ty,
                dst: v(2),
                addr: at_v1(0),
                expected: imm(0),
                desired: imm(1),
                success_ordering: MemoryOrdering::Relaxed,
                failure_ordering: MemoryOrdering::Release,
            }));
            assert!(refuse(Instruction::AtomicBinary {
                op: AtomicBinOp::Add,
                ty: MirType::Scalar(ScalarType::F64),
                dst: v(2),
                addr: at_v1(0),
                value: Operand::Immediate(Immediate::F64(1.0)),
                ordering: MemoryOrdering::SeqCst,
            }));
        }

        /// Runs `entry`, which takes one i64 argument, and returns its result.
        fn run_with_address(module: &Module, entry: &str, argument: i64) -> i64 {
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            let (code, offsets) = assembler
                .compile_mir_to_binary_function(module, None)
                .expect("compile");
            let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
            memory.write_code(&code).expect("write");
            memory.make_executable().expect("protect");
            let start = unsafe { memory.code_start().add(offsets[entry]) };
            let function: unsafe extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(start) };
            unsafe { function(argument) }
        }

        /// Calls the address in its argument through `args` and `arg_types`, returning the i64 result.
        fn indirect_caller(arg_types: Vec<MirType>, args: Vec<Operand>) -> Module {
            let mut entry = Block::new("entry");
            entry.push(Instruction::CallIndirect {
                target: reg(0),
                arg_aggregates: vec![None; args.len()],
                arg_types,
                args,
                variadic_fixed: None,
                ret: Some(v(1)),
                ret_ty: Some(I64),
                ret_aggregate: None,
            });
            ret(&mut entry, 1);
            let mut main = Function::new(
                Signature::new("main")
                    .with_params(vec![Parameter::new(v(0), I64)])
                    .with_return(I64),
            );
            main.add_block(entry);
            module_of(vec![main])
        }

        extern "C" fn c_mixed(a: i64, c: i64, b: f64) -> i64 {
            a * 100 + c * 10 + (b * 10.0) as i64
        }

        extern "C" fn c_float_pair(x: f32, y: f64) -> i64 {
            (x * 2.0) as i64 * 1000 + y as i64
        }

        #[test]
        fn indirect_call_passes_floats_to_a_c_function() {
            let mixed = indirect_caller(vec![I64, I64, F64_TY], vec![imm(7), imm(5), f64_imm(2.5)]);
            assert_eq!(
                run_with_address(
                    &mixed,
                    "main",
                    c_mixed as extern "C" fn(i64, i64, f64) -> i64 as usize as i64
                ),
                775
            );
            let pair = indirect_caller(
                vec![MirType::Scalar(ScalarType::F32), F64_TY],
                vec![Operand::Immediate(Immediate::F32(1.5)), f64_imm(7.0)],
            );
            assert_eq!(
                run_with_address(
                    &pair,
                    "main",
                    c_float_pair as extern "C" fn(f32, f64) -> i64 as usize as i64
                ),
                3007
            );
        }

        #[test]
        fn indirect_call_passes_floats_to_a_jit_function() {
            let mut callee_entry = Block::new("entry");
            callee_entry.push(Instruction::Bitcast {
                from: F64_TY,
                to: I64,
                dst: v(2),
                src: reg(1),
            });
            callee_entry.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: I64,
                dst: v(3),
                lhs: reg(0),
                rhs: reg(2),
            });
            ret(&mut callee_entry, 3);
            let mut callee = Function::new(
                Signature::new("takes_float")
                    .with_params(vec![
                        Parameter::new(v(0), I64),
                        Parameter::new(v(1), F64_TY),
                    ])
                    .with_return(I64),
            );
            callee.add_block(callee_entry);

            let mut entry = Block::new("entry");
            entry.push(Instruction::SymbolAddress {
                dst: v(0),
                symbol: "takes_float".to_string(),
            });
            entry.push(Instruction::CallIndirect {
                target: reg(0),
                arg_types: vec![I64, F64_TY],
                args: vec![imm(40), f64_imm(2.5)],
                variadic_fixed: None,
                ret: Some(v(1)),
                ret_ty: Some(I64),
                ret_aggregate: None,
                arg_aggregates: vec![None, None],
            });
            ret(&mut entry, 1);
            let mut main = Function::new(Signature::new("main").with_return(I64));
            main.add_block(entry);
            let module = module_of(vec![callee, main]);
            assert_eq!(
                run(&module, "main"),
                40i64.wrapping_add(2.5f64.to_bits() as i64)
            );
        }

        #[test]
        fn indirect_call_refuses_a_float_before_an_integer() {
            let module = indirect_caller(vec![F64_TY, I64], vec![f64_imm(1.0), imm(2)]);
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            assert!(
                assembler
                    .compile_mir_to_binary_function(&module, None)
                    .is_err()
            );
        }

        /// DMB ISH and DMB ISHLD, as the words the fence encoder writes.
        const DMB_ISH: u32 = 0xD503_3BBF;
        const DMB_ISHLD: u32 = 0xD503_39BF;

        fn count_word(code: &[u8], word: u32) -> usize {
            code.as_chunks::<4>()
                .0
                .iter()
                .filter(|chunk| u32::from_le_bytes(**chunk) == word)
                .count()
        }

        #[test]
        fn fence_encodes_the_barrier_each_ordering_needs() {
            use MemoryOrdering::{AcqRel, Acquire, Relaxed, Release, SeqCst};
            let cases = [
                (Relaxed, 0, 0),
                (Acquire, 0, 1),
                (Release, 1, 0),
                (AcqRel, 1, 0),
                (SeqCst, 1, 0),
            ];
            for (ordering, ish, ishld) in cases {
                let name = format!("fence_{ordering:?}");
                let function = atomic_function(&name, |entry| {
                    entry.push(Instruction::Store {
                        ty: I64,
                        src: imm(42),
                        addr: at_base(1),
                        attrs: MemoryAttrs::default(),
                    });
                    entry.push(Instruction::Fence { ordering });
                    entry.push(Instruction::Load {
                        ty: I64,
                        dst: v(2),
                        addr: at_base(1),
                        attrs: MemoryAttrs::default(),
                    });
                    ret(entry, 2);
                });
                let module = module_of(vec![function]);
                let mut assembler =
                    RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
                let (code, _) = assembler
                    .compile_mir_to_binary_function(&module, None)
                    .expect("compile");
                assert_eq!(count_word(&code, DMB_ISH), ish, "{ordering:?} DMB ISH");
                assert_eq!(
                    count_word(&code, DMB_ISHLD),
                    ishld,
                    "{ordering:?} DMB ISHLD"
                );
                assert_eq!(
                    run_named(vec![module.functions[&name].clone()], &[name])[0],
                    42
                );
            }
        }

        // Vectors: 128-bit values in 16-byte frame slots.

        const LANES: [VectorLane; 6] = [
            VectorLane::I8,
            VectorLane::I16,
            VectorLane::I32,
            VectorLane::I64,
            VectorLane::F32,
            VectorLane::F64,
        ];
        const PTR: MirType = MirType::Scalar(ScalarType::Ptr);

        fn vec_ty(lane: VectorLane) -> MirType {
            MirType::Vector(VectorType::V128(lane))
        }

        fn lane_bits(lane: VectorLane) -> u32 {
            match lane {
                VectorLane::I8 => 8,
                VectorLane::I16 => 16,
                VectorLane::I32 | VectorLane::F32 => 32,
                VectorLane::I64 | VectorLane::F64 => 64,
            }
        }

        fn lane_count(lane: VectorLane) -> usize {
            128 / lane_bits(lane) as usize
        }

        fn lane_mask(lane: VectorLane) -> u64 {
            match lane_bits(lane) {
                64 => u64::MAX,
                bits => (1u64 << bits) - 1,
            }
        }

        /// An immediate of the lane's type holding the raw bits `bits`.
        fn lane_imm(lane: VectorLane, bits: u64) -> Operand {
            Operand::Immediate(match lane {
                VectorLane::I8 => Immediate::I8(bits as u8 as i8),
                VectorLane::I16 => Immediate::I16(bits as u16 as i16),
                VectorLane::I32 => Immediate::I32(bits as u32 as i32),
                VectorLane::I64 => Immediate::I64(bits as i64),
                VectorLane::F32 => Immediate::F32(f32::from_bits(bits as u32)),
                VectorLane::F64 => Immediate::F64(f64::from_bits(bits)),
            })
        }

        /// Bit patterns that matter for a lane: zero, all ones, the sign boundary, alternating bits
        /// and, for floats, signed zeros, infinities, a denormal, the largest finite value and NaNs
        /// with payloads.
        fn edge_bits(lane: VectorLane) -> Vec<u64> {
            match lane {
                VectorLane::F32 => vec![
                    0.0f32.to_bits() as u64,
                    (-0.0f32).to_bits() as u64,
                    1.5f32.to_bits() as u64,
                    (-2.25f32).to_bits() as u64,
                    f32::MAX.to_bits() as u64,
                    f32::MIN_POSITIVE.to_bits() as u64,
                    1,
                    f32::INFINITY.to_bits() as u64,
                    f32::NEG_INFINITY.to_bits() as u64,
                    0x7FC0_1234,
                    0xFFA0_0000,
                    3.0e38f32.to_bits() as u64,
                ],
                VectorLane::F64 => vec![
                    0.0f64.to_bits(),
                    (-0.0f64).to_bits(),
                    1.5f64.to_bits(),
                    (-2.25f64).to_bits(),
                    f64::MAX.to_bits(),
                    f64::MIN_POSITIVE.to_bits(),
                    1,
                    f64::INFINITY.to_bits(),
                    f64::NEG_INFINITY.to_bits(),
                    0x7FF8_0000_0000_1234,
                    0xFFF4_0000_0000_0000,
                    3.0e300f64.to_bits(),
                ],
                _ => {
                    let mask = lane_mask(lane);
                    let sign = 1u64 << (lane_bits(lane) - 1);
                    [
                        0,
                        1,
                        mask,
                        mask - 1,
                        sign,
                        sign - 1,
                        0x5555_5555_5555_5555 & mask,
                        0xAAAA_AAAA_AAAA_AAAA & mask,
                        0x7F & mask,
                        2,
                        0x0102_0304_0506_0708 & mask,
                    ]
                    .to_vec()
                }
            }
        }

        /// What one lane holds after `op`, computed the way the lane's type would.
        fn reference_lane(op: VectorOp, lane: VectorLane, a: u64, b: u64) -> u64 {
            let mask = lane_mask(lane);
            match (op, lane) {
                (VectorOp::VAnd, _) => a & b,
                (VectorOp::VOr, _) => a | b,
                (VectorOp::VXor, _) => a ^ b,
                (_, VectorLane::F32) => {
                    let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
                    match op {
                        VectorOp::VAdd => x + y,
                        VectorOp::VSub => x - y,
                        _ => x * y,
                    }
                    .to_bits() as u64
                }
                (_, VectorLane::F64) => {
                    let (x, y) = (f64::from_bits(a), f64::from_bits(b));
                    match op {
                        VectorOp::VAdd => x + y,
                        VectorOp::VSub => x - y,
                        _ => x * y,
                    }
                    .to_bits()
                }
                (VectorOp::VAdd, _) => a.wrapping_add(b) & mask,
                (VectorOp::VSub, _) => a.wrapping_sub(b) & mask,
                _ => a.wrapping_mul(b) & mask,
            }
        }

        fn is_nan_bits(lane: VectorLane, bits: u64) -> bool {
            match lane {
                VectorLane::F32 => f32::from_bits(bits as u32).is_nan(),
                VectorLane::F64 => f64::from_bits(bits).is_nan(),
                _ => false,
            }
        }

        /// Float lanes match when both are NaN, since the payload of a NaN result is not part of
        /// the arithmetic. Everything else must match bit for bit.
        fn same_lane(lane: VectorLane, got: u64, want: u64) -> bool {
            if is_nan_bits(lane, got) && is_nan_bits(lane, want) {
                return true;
            }
            got == want
        }

        fn lanes_of(lane: VectorLane, bytes: &[u8]) -> Vec<u64> {
            let width = lane_bits(lane) as usize / 8;
            bytes
                .chunks_exact(width)
                .map(|chunk| {
                    let mut word = [0u8; 8];
                    word[..width].copy_from_slice(chunk);
                    u64::from_le_bytes(word)
                })
                .collect()
        }

        fn assert_lanes(lane: VectorLane, got: &[u8], want: &[u64], context: &str) {
            let got = lanes_of(lane, got);
            assert_eq!(got.len(), want.len(), "{context}: lane count");
            for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
                assert!(
                    same_lane(lane, g, w),
                    "{context}: lane {index} is {g:#x}, want {w:#x}"
                );
            }
        }

        /// Builds the vector in `v<id>` from one value per lane, a splat then an insert per lane.
        fn build_vector(entry: &mut Block, lane: VectorLane, id: u32, values: &[u64]) {
            entry.push(Instruction::VectorOp {
                op: VectorOp::VSplat,
                ty: vec_ty(lane),
                dst: v(id),
                operands: vec![lane_imm(lane, values[0])],
            });
            for (index, value) in values.iter().enumerate() {
                entry.push(Instruction::VectorOp {
                    op: VectorOp::VInsertLane,
                    ty: vec_ty(lane),
                    dst: v(id),
                    operands: vec![reg(id), lane_imm(lane, *value), imm(index as i64)],
                });
            }
        }

        fn store_vector(entry: &mut Block, lane: VectorLane, src: u32, base: u32, offset: i16) {
            entry.push(Instruction::Store {
                ty: vec_ty(lane),
                src: reg(src),
                addr: AddressMode::BaseOffset {
                    address_space: Default::default(),
                    base: v(base),
                    offset,
                },
                attrs: MemoryAttrs::default(),
            });
        }

        /// Compiles `module` and returns the memory holding it with the offset of each function.
        fn jit(module: &Module) -> (ExecutableMemory, std::collections::HashMap<String, usize>) {
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            let (code, offsets) = assembler
                .compile_mir_to_binary_function(module, None)
                .expect("compile");
            let mut memory = ExecutableMemory::allocate_writable(code.len()).expect("allocate");
            memory.write_code(&code).expect("write");
            memory.make_executable().expect("protect");
            (memory, offsets)
        }

        /// Runs `entry`, which takes a destination pointer, and returns the bytes it wrote there.
        fn run_into(module: &Module, entry: &str, len: usize) -> Vec<u8> {
            let (memory, offsets) = jit(module);
            let start = unsafe { memory.code_start().add(offsets[entry]) };
            let function: unsafe extern "C" fn(*mut u8) = unsafe { std::mem::transmute(start) };
            let mut out = vec![0u8; len];
            unsafe { function(out.as_mut_ptr()) };
            out
        }

        /// Runs `entry`, which takes a source and a destination pointer.
        fn run_copy(module: &Module, entry: &str, src: &[u8], len: usize) -> Vec<u8> {
            let (memory, offsets) = jit(module);
            let start = unsafe { memory.code_start().add(offsets[entry]) };
            let function: unsafe extern "C" fn(*const u8, *mut u8) =
                unsafe { std::mem::transmute(start) };
            let mut out = vec![0u8; len];
            unsafe { function(src.as_ptr(), out.as_mut_ptr()) };
            out
        }

        /// `a op b` on two vectors built lane by lane, stored at the destination pointer.
        fn vector_op_function(
            name: &str,
            lane: VectorLane,
            op: VectorOp,
            a: &[u64],
            b: &[u64],
        ) -> Function {
            let mut entry = Block::new("entry");
            build_vector(&mut entry, lane, 1, a);
            build_vector(&mut entry, lane, 2, b);
            entry.push(Instruction::VectorOp {
                op,
                ty: vec_ty(lane),
                dst: v(3),
                operands: vec![reg(1), reg(2)],
            });
            store_vector(&mut entry, lane, 3, 0, 0);
            entry.push(Instruction::Ret { value: None });
            let mut function =
                Function::new(Signature::new(name).with_params(vec![Parameter::new(v(0), PTR)]));
            function.add_block(entry);
            function
        }

        #[test]
        fn vector_arithmetic_matches_scalar_lanes_at_every_width() {
            let ops = [
                VectorOp::VAdd,
                VectorOp::VSub,
                VectorOp::VMul,
                VectorOp::VAnd,
                VectorOp::VOr,
                VectorOp::VXor,
            ];
            for lane in LANES {
                let edges = edge_bits(lane);
                let lanes = lane_count(lane);
                for op in ops {
                    if op == VectorOp::VMul && lane == VectorLane::I64 {
                        continue;
                    }
                    for shift in 0..edges.len() {
                        let a: Vec<u64> = (0..lanes)
                            .map(|i| edges[(i + shift) % edges.len()])
                            .collect();
                        let b: Vec<u64> = (0..lanes)
                            .map(|i| edges[(i * 5 + shift + 3) % edges.len()])
                            .collect();
                        let module = module_of(vec![vector_op_function("vop", lane, op, &a, &b)]);
                        let got = run_into(&module, "vop", 16);
                        let want: Vec<u64> = a
                            .iter()
                            .zip(&b)
                            .map(|(&x, &y)| reference_lane(op, lane, x, y))
                            .collect();
                        assert_lanes(lane, &got, &want, &format!("{op} {lane:?} shift {shift}"));
                    }
                }
            }
        }

        #[test]
        fn vector_load_and_store_copy_bytes_at_an_offset() {
            let patterns: Vec<Vec<u8>> = vec![
                (0..48u8)
                    .map(|i| i.wrapping_mul(37).wrapping_add(0x80))
                    .collect(),
                vec![0xFF; 48],
                (0..48u8)
                    .map(|i| if i % 4 == 3 { 0x7F } else { 0xC0 })
                    .collect(),
            ];
            for lane in LANES {
                for pattern in &patterns {
                    // Load 16 bytes at offset 16 of the source, store them at offset 0 of the destination.
                    let mut entry = Block::new("entry");
                    entry.push(Instruction::Load {
                        ty: vec_ty(lane),
                        dst: v(1),
                        addr: AddressMode::BaseOffset {
                            address_space: Default::default(),
                            base: v(0),
                            offset: 16,
                        },
                        attrs: MemoryAttrs::default(),
                    });
                    store_vector(&mut entry, lane, 1, 10, 0);
                    entry.push(Instruction::Ret { value: None });
                    let mut function =
                        Function::new(Signature::new("copy").with_params(vec![
                            Parameter::new(v(0), PTR),
                            Parameter::new(v(10), PTR),
                        ]));
                    function.add_block(entry);
                    let module = module_of(vec![function]);
                    let out = run_copy(&module, "copy", pattern, 48);
                    assert_eq!(&out[..16], &pattern[16..32], "{lane:?} copy");
                    assert!(
                        out[16..].iter().all(|&byte| byte == 0),
                        "{lane:?} wrote past 16 bytes"
                    );
                }
            }
        }

        /// The value of a lane read as a narrow integer, sign-extended to 64 bits. Float lanes
        /// use the same rule as the scalar floats in their slots.
        fn sign_extended(lane: VectorLane, bits: u64) -> i64 {
            match lane_bits(lane) {
                64 => bits as i64,
                width => ((bits << (64 - width)) as i64) >> (64 - width),
            }
        }

        #[test]
        fn vector_splat_and_lane_insert_match_every_width() {
            for lane in LANES {
                let edges = edge_bits(lane);
                let lanes = lane_count(lane);
                let mask = lane_mask(lane);
                for &bits in &edges {
                    // An immediate splat, and a splat from a register whose upper bits are junk: a
                    // narrow lane must take only its own bits from the scalar.
                    let junk = bits | !mask;
                    let mut from_imm = Block::new("entry");
                    from_imm.push(Instruction::VectorOp {
                        op: VectorOp::VSplat,
                        ty: vec_ty(lane),
                        dst: v(1),
                        operands: vec![lane_imm(lane, bits)],
                    });
                    store_vector(&mut from_imm, lane, 1, 0, 0);
                    from_imm.push(Instruction::Ret { value: None });
                    let mut from_reg = Block::new("entry");
                    from_reg.push(Instruction::Copy {
                        ty: MirType::Scalar(ScalarType::I64),
                        dst: v(5),
                        src: imm(junk as i64),
                    });
                    from_reg.push(Instruction::VectorOp {
                        op: VectorOp::VSplat,
                        ty: vec_ty(lane),
                        dst: v(1),
                        operands: vec![reg(5)],
                    });
                    store_vector(&mut from_reg, lane, 1, 0, 0);
                    from_reg.push(Instruction::Ret { value: None });
                    for block in [from_imm, from_reg] {
                        let mut function = Function::new(
                            Signature::new("splat").with_params(vec![Parameter::new(v(0), PTR)]),
                        );
                        function.add_block(block);
                        let got = run_into(&module_of(vec![function]), "splat", 16);
                        assert_lanes(
                            lane,
                            &got,
                            &vec![bits & mask; lanes],
                            &format!("splat {lane:?} {bits:#x}"),
                        );
                    }

                    // Insert the scalar into every lane of a vector built from other values.
                    let base: Vec<u64> = (0..lanes).map(|i| edges[(i + 1) % edges.len()]).collect();
                    for index in 0..lanes {
                        let mut entry = Block::new("entry");
                        build_vector(&mut entry, lane, 1, &base);
                        entry.push(Instruction::VectorOp {
                            op: VectorOp::VInsertLane,
                            ty: vec_ty(lane),
                            dst: v(2),
                            operands: vec![reg(1), lane_imm(lane, bits), imm(index as i64)],
                        });
                        store_vector(&mut entry, lane, 2, 0, 0);
                        entry.push(Instruction::Ret { value: None });
                        let mut function = Function::new(
                            Signature::new("insert").with_params(vec![Parameter::new(v(0), PTR)]),
                        );
                        function.add_block(entry);
                        let got = run_into(&module_of(vec![function]), "insert", 16);
                        let mut want = base.clone();
                        want[index] = bits & mask;
                        assert_lanes(
                            lane,
                            &got,
                            &want,
                            &format!("insert {lane:?} {bits:#x} at {index}"),
                        );
                    }
                }
            }
        }

        #[test]
        fn vector_extract_lane_sign_extends_every_lane() {
            let mut functions = Vec::new();
            let mut names = Vec::new();
            let mut expected = Vec::new();
            for lane in LANES {
                let edges = edge_bits(lane);
                let lanes = lane_count(lane);
                for shift in 0..3 {
                    let values: Vec<u64> = (0..lanes)
                        .map(|i| edges[(i + shift) % edges.len()])
                        .collect();
                    for index in 0..lanes {
                        let name = format!("x{}", functions.len());
                        let mut entry = Block::new("entry");
                        build_vector(&mut entry, lane, 1, &values);
                        entry.push(Instruction::VectorOp {
                            op: VectorOp::VExtractLane,
                            ty: vec_ty(lane),
                            dst: v(2),
                            operands: vec![reg(1), imm(index as i64)],
                        });
                        ret(&mut entry, 2);
                        let mut function = Function::new(Signature::new(&name).with_return(I64));
                        function.add_block(entry);
                        functions.push(function);
                        names.push(name);
                        expected.push(sign_extended(lane, values[index]));
                    }
                }
            }
            let got = run_named(functions, &names);
            for (index, (got, want)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(got, want, "extract case {index}");
            }
        }

        #[test]
        fn vector_slots_past_the_immediate_range_keep_their_values() {
            // 300 vectors take 4800 bytes of frame, so the later ones sit past the 12-bit offsets.
            let mut entry = Block::new("entry");
            for k in 0..300u32 {
                entry.push(Instruction::VectorOp {
                    op: VectorOp::VSplat,
                    ty: vec_ty(VectorLane::I32),
                    dst: v(100 + k),
                    operands: vec![lane_imm(VectorLane::I32, u64::from(k))],
                });
            }
            entry.push(Instruction::VectorOp {
                op: VectorOp::VAdd,
                ty: vec_ty(VectorLane::I32),
                dst: v(1000),
                operands: vec![reg(100), reg(399)],
            });
            entry.push(Instruction::VectorOp {
                op: VectorOp::VAdd,
                ty: vec_ty(VectorLane::I32),
                dst: v(1001),
                operands: vec![reg(150), reg(250)],
            });
            store_vector(&mut entry, VectorLane::I32, 1000, 0, 0);
            store_vector(&mut entry, VectorLane::I32, 1001, 0, 16);
            entry.push(Instruction::Ret { value: None });
            let mut function =
                Function::new(Signature::new("deep").with_params(vec![Parameter::new(v(0), PTR)]));
            function.add_block(entry);
            let got = run_into(&module_of(vec![function]), "deep", 32);
            let want: Vec<u64> = vec![299; 4];
            assert_lanes(VectorLane::I32, &got[..16], &want, "first sum");
            // v(150) holds 50 and v(250) holds 150.
            let want: Vec<u64> = vec![200; 4];
            assert_lanes(VectorLane::I32, &got[16..], &want, "second sum");
        }

        /// Compiles `module` and returns the refusal, which must be one of this encoder's.
        #[track_caller]
        fn refusal(module: &Module) -> String {
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            match assembler.compile_mir_to_binary_function(module, None) {
                Ok(_) => panic!("compiled a vector shape that must be refused"),
                Err(error) => {
                    let text = format!("{error:?}");
                    assert!(
                        text.contains("AArch64 JIT"),
                        "not an encoder refusal: {text}"
                    );
                    text
                }
            }
        }

        /// A module of one function `main` with `signature` and the body `build` makes.
        fn refused_main(signature: Signature, build: impl FnOnce(&mut Block)) -> Module {
            let mut entry = Block::new("entry");
            build(&mut entry);
            let mut function = Function::new(signature);
            function.add_block(entry);
            module_of(vec![function])
        }

        fn splat_i32(entry: &mut Block, dst: u32, value: u64) {
            entry.push(Instruction::VectorOp {
                op: VectorOp::VSplat,
                ty: vec_ty(VectorLane::I32),
                dst: v(dst),
                operands: vec![lane_imm(VectorLane::I32, value)],
            });
        }

        #[test]
        fn vector_shapes_the_encoder_does_not_model_are_refused() {
            let no_params = || Signature::new("main");

            let refused = refusal(&refused_main(no_params(), |entry| {
                for (dst, value) in [(2, 1), (3, 2)] {
                    entry.push(Instruction::VectorOp {
                        op: VectorOp::VSplat,
                        ty: vec_ty(VectorLane::I64),
                        dst: v(dst),
                        operands: vec![lane_imm(VectorLane::I64, value)],
                    });
                }
                entry.push(Instruction::VectorOp {
                    op: VectorOp::VMul,
                    ty: vec_ty(VectorLane::I64),
                    dst: v(1),
                    operands: vec![reg(2), reg(3)],
                });
                entry.push(Instruction::Ret { value: None });
            }));
            assert!(refused.contains("vmul on 64-bit lanes"), "{refused}");

            refusal(&refused_main(no_params(), |entry| {
                entry.push(Instruction::Load {
                    ty: MirType::Vector(VectorType::V256(VectorLane::I32)),
                    dst: v(1),
                    addr: at_base(0),
                    attrs: MemoryAttrs::default(),
                });
                entry.push(Instruction::Ret { value: None });
            }));

            refusal(&refused_main(no_params(), |entry| {
                splat_i32(entry, 1, 1);
                entry.push(Instruction::VectorOp {
                    op: VectorOp::VShl,
                    ty: vec_ty(VectorLane::I32),
                    dst: v(2),
                    operands: vec![reg(1), imm(3)],
                });
                entry.push(Instruction::Ret { value: None });
            }));

            let takes = {
                let mut entry = Block::new("entry");
                entry.push(Instruction::Ret { value: None });
                let mut function = Function::new(
                    Signature::new("takes").with_params(vec![Parameter::new(v(0), I64)]),
                );
                function.add_block(entry);
                function
            };
            let mut main_entry = Block::new("entry");
            splat_i32(&mut main_entry, 1, 1);
            main_entry.push(Instruction::Call {
                name: "takes".to_string(),
                args: vec![reg(1)],
                ret: None,
                ret_ty: None,
            });
            main_entry.push(Instruction::Ret { value: None });
            let mut main = Function::new(Signature::new("main"));
            main.add_block(main_entry);
            refusal(&module_of(vec![takes, main]));

            refusal(&refused_main(
                Signature::new("main").with_return(vec_ty(VectorLane::I32)),
                |entry| {
                    splat_i32(entry, 1, 1);
                    entry.push(Instruction::Ret {
                        value: Some(reg(1)),
                    });
                },
            ));

            refusal(&refused_main(
                Signature::new("main")
                    .with_params(vec![Parameter::new(v(1), vec_ty(VectorLane::I32))]),
                |entry| entry.push(Instruction::Ret { value: None }),
            ));

            refusal(&refused_main(
                Signature::new("main").with_return(I64),
                |entry| {
                    splat_i32(entry, 1, 1);
                    entry.push(Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty: I64,
                        dst: v(2),
                        lhs: reg(1),
                        rhs: imm(0),
                    });
                    ret(entry, 2);
                },
            ));

            let refused = refusal(&refused_main(
                Signature::new("main").with_return(I64),
                |entry| {
                    splat_i32(entry, 1, 1);
                    entry.push(Instruction::VectorOp {
                        op: VectorOp::VExtractLane,
                        ty: vec_ty(VectorLane::I32),
                        dst: v(2),
                        operands: vec![reg(1), imm(4)],
                    });
                    ret(entry, 2);
                },
            ));
            assert!(refused.contains("out of range"), "{refused}");

            refusal(&refused_main(
                Signature::new("main").with_return(I64),
                |entry| {
                    entry.push(Instruction::Copy {
                        ty: MirType::Scalar(ScalarType::I64),
                        dst: v(3),
                        src: imm(1),
                    });
                    splat_i32(entry, 1, 1);
                    entry.push(Instruction::Select {
                        ty: vec_ty(VectorLane::I32),
                        dst: v(2),
                        cond: v(3),
                        true_val: reg(1),
                        false_val: reg(1),
                    });
                    ret(entry, 2);
                },
            ));

            refusal(&refused_main(no_params(), |entry| {
                splat_i32(entry, 1, 1);
                entry.push(Instruction::StackAddr {
                    dst: v(2),
                    slot: VirtualReg::gpr(1),
                });
                entry.push(Instruction::Ret { value: None });
            }));

            refusal(&refused_main(no_params(), |entry| {
                entry.push(Instruction::VectorOp {
                    op: VectorOp::VAdd,
                    ty: vec_ty(VectorLane::I32),
                    dst: v(1),
                    operands: vec![imm(1), imm(2)],
                });
                entry.push(Instruction::Ret { value: None });
            }));

            refusal(&refused_main(no_params(), |entry| {
                splat_i32(entry, 1, 1);
                entry.push(Instruction::VectorOp {
                    op: VectorOp::VInsertLane,
                    ty: vec_ty(VectorLane::I32),
                    dst: v(2),
                    operands: vec![reg(1), lane_imm(VectorLane::I32, 2), reg(3)],
                });
                entry.push(Instruction::Ret { value: None });
            }));

            refusal(&refused_main(no_params(), |entry| {
                splat_i32(entry, 1, 1);
                entry.push(Instruction::Copy {
                    ty: vec_ty(VectorLane::I32),
                    dst: v(2),
                    src: reg(1),
                });
                entry.push(Instruction::Ret { value: None });
            }));
        }

        // Float results of indirect calls, from C callees and from JIT callees.

        extern "C" fn c_echo_f64(x: f64) -> f64 {
            x
        }

        extern "C" fn c_echo_f32(x: f32) -> f32 {
            x
        }

        extern "C" fn c_scale(a: i64, b: f64) -> f64 {
            a as f64 + b
        }

        extern "C" fn c_nan_payload() -> f64 {
            f64::from_bits(0x7FF8_0000_0000_1234)
        }

        /// Calls the address in its argument with `args`, and returns the result's bits.
        fn float_indirect_caller(
            arg_types: Vec<MirType>,
            args: Vec<Operand>,
            ret_ty: MirType,
        ) -> Module {
            let mut entry = Block::new("entry");
            entry.push(Instruction::CallIndirect {
                target: reg(0),
                arg_aggregates: vec![None; args.len()],
                arg_types,
                args,
                variadic_fixed: None,
                ret: Some(v(1)),
                ret_ty: Some(ret_ty),
                ret_aggregate: None,
            });
            ret(&mut entry, 1);
            let mut main = Function::new(
                Signature::new("main")
                    .with_params(vec![Parameter::new(v(0), I64)])
                    .with_return(I64),
            );
            main.add_block(entry);
            module_of(vec![main])
        }

        fn f64_operand(bits: u64) -> Operand {
            Operand::Immediate(Immediate::F64(f64::from_bits(bits)))
        }

        fn f32_operand(bits: u32) -> Operand {
            Operand::Immediate(Immediate::F32(f32::from_bits(bits)))
        }

        const F64_EDGES: [u64; 8] = [
            0,
            0x8000_0000_0000_0000,
            0x3FF8_0000_0000_0000,
            0x7FF0_0000_0000_0000,
            0xFFF0_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x7FEF_FFFF_FFFF_FFFF,
            0x7FF8_0000_0000_1234,
        ];

        const F32_EDGES: [u32; 7] = [
            0,
            0x8000_0000,
            0x3FC0_0000,
            0x7F80_0000,
            0x0000_0001,
            0x7F7F_FFFF,
            0x7FC0_1234,
        ];

        #[test]
        fn indirect_call_returns_a_float_from_a_c_function() {
            for bits in F64_EDGES {
                let module = float_indirect_caller(vec![F64_TY], vec![f64_operand(bits)], F64_TY);
                assert_eq!(
                    run_with_address(
                        &module,
                        "main",
                        c_echo_f64 as extern "C" fn(f64) -> f64 as usize as i64
                    ),
                    bits as i64,
                    "f64 {bits:#x}"
                );
            }
            for bits in F32_EDGES {
                let module = float_indirect_caller(
                    vec![MirType::Scalar(ScalarType::F32)],
                    vec![f32_operand(bits)],
                    MirType::Scalar(ScalarType::F32),
                );
                assert_eq!(
                    run_with_address(
                        &module,
                        "main",
                        c_echo_f32 as extern "C" fn(f32) -> f32 as usize as i64
                    ),
                    i64::from(bits),
                    "f32 {bits:#x}"
                );
            }
            let mixed = float_indirect_caller(
                vec![I64, F64_TY],
                vec![imm(-3), f64_operand(2.5f64.to_bits())],
                F64_TY,
            );
            assert_eq!(
                run_with_address(
                    &mixed,
                    "main",
                    c_scale as extern "C" fn(i64, f64) -> f64 as usize as i64
                ),
                (-3.0f64 + 2.5).to_bits() as i64
            );
            let nan = float_indirect_caller(vec![], vec![], F64_TY);
            assert_eq!(
                run_with_address(
                    &nan,
                    "main",
                    c_nan_payload as extern "C" fn() -> f64 as usize as i64
                ),
                0x7FF8_0000_0000_1234
            );
        }

        /// `add_half(x) = x + 0.5`, a JIT function that returns a computed float.
        fn add_half_function() -> Function {
            let mut entry = Block::new("entry");
            entry.push(Instruction::FloatBinary {
                op: FloatBinOp::FAdd,
                ty: F64_TY,
                dst: v(1),
                lhs: reg(0),
                rhs: f64_imm(0.5),
            });
            ret(&mut entry, 1);
            let mut function = Function::new(
                Signature::new("add_half")
                    .with_params(vec![Parameter::new(v(0), F64_TY)])
                    .with_return(F64_TY),
            );
            function.add_block(entry);
            function
        }

        #[test]
        fn indirect_call_returns_a_float_from_a_jit_function() {
            let inputs = [2.25f64, -7.5, 1.0e300, -0.0];
            for input in inputs {
                let mut entry = Block::new("entry");
                entry.push(Instruction::SymbolAddress {
                    dst: v(0),
                    symbol: "add_half".to_string(),
                });
                entry.push(Instruction::CallIndirect {
                    target: reg(0),
                    arg_types: vec![F64_TY],
                    args: vec![f64_operand(input.to_bits())],
                    variadic_fixed: None,
                    ret: Some(v(1)),
                    ret_ty: Some(F64_TY),
                    ret_aggregate: None,
                    arg_aggregates: vec![None],
                });
                entry.push(Instruction::Bitcast {
                    from: F64_TY,
                    to: I64,
                    dst: v(2),
                    src: reg(1),
                });
                ret(&mut entry, 2);
                let mut main = Function::new(Signature::new("main").with_return(I64));
                main.add_block(entry);
                let module = module_of(vec![add_half_function(), main]);
                assert_eq!(
                    run(&module, "main"),
                    (input + 0.5).to_bits() as i64,
                    "add_half({input})"
                );
            }
        }

        #[test]
        fn jit_float_return_is_still_read_from_x0_by_a_direct_call() {
            let mut entry = Block::new("entry");
            entry.push(Instruction::Call {
                name: "add_half".to_string(),
                args: vec![f64_operand(2.25f64.to_bits())],
                ret: Some(v(1)),
                ret_ty: Some(MirType::Scalar(ScalarType::F64)),
            });
            entry.push(Instruction::Bitcast {
                from: F64_TY,
                to: I64,
                dst: v(2),
                src: reg(1),
            });
            ret(&mut entry, 2);
            let mut main = Function::new(Signature::new("main").with_return(I64));
            main.add_block(entry);
            let module = module_of(vec![add_half_function(), main]);
            assert_eq!(run(&module, "main"), 2.75f64.to_bits() as i64);
        }

        #[test]
        fn indirect_call_returns_a_float_immediate_from_a_jit_function() {
            let mut f64_callee = Block::new("entry");
            f64_callee.push(Instruction::Ret {
                value: Some(f64_operand(2.5f64.to_bits())),
            });
            let mut f64_fn = Function::new(Signature::new("constant64").with_return(F64_TY));
            f64_fn.add_block(f64_callee);

            let mut f32_callee = Block::new("entry");
            f32_callee.push(Instruction::Ret {
                value: Some(f32_operand(1.25f32.to_bits())),
            });
            let mut f32_fn = Function::new(
                Signature::new("constant32").with_return(MirType::Scalar(ScalarType::F32)),
            );
            f32_fn.add_block(f32_callee);

            for (name, ty, expected) in [
                ("constant64", F64_TY, 2.5f64.to_bits()),
                (
                    "constant32",
                    MirType::Scalar(ScalarType::F32),
                    u64::from(1.25f32.to_bits()),
                ),
            ] {
                let mut entry = Block::new("entry");
                entry.push(Instruction::SymbolAddress {
                    dst: v(0),
                    symbol: name.to_string(),
                });
                entry.push(Instruction::CallIndirect {
                    target: reg(0),
                    arg_types: vec![],
                    args: vec![],
                    variadic_fixed: None,
                    ret: Some(v(1)),
                    ret_ty: Some(ty),
                    ret_aggregate: None,
                    arg_aggregates: vec![],
                });
                ret(&mut entry, 1);
                let mut main = Function::new(Signature::new("main").with_return(I64));
                main.add_block(entry);
                let module = module_of(vec![f64_fn.clone(), f32_fn.clone(), main]);
                // An f32 result is its 32 bits, zero-extended, as the move from s0 leaves it.
                assert_eq!(run(&module, "main") as u64, expected, "{name}");
            }
        }

        #[test]
        fn va_start_and_va_arg_are_refused_not_miscompiled() {
            // The JIT's own calls pass every argument in a register by position. Darwin's va_list
            // reads variadic arguments from the stack, so a variadic function here is refused.
            let mut entry = Block::new("entry");
            entry.push(slot_addr(1, 50));
            entry.push(Instruction::VaStart { ap: reg(1) });
            entry.push(Instruction::VaArg {
                ty: I64,
                dst: v(2),
                ap: reg(1),
            });
            ret(&mut entry, 2);
            let mut function = Function::new(
                Signature::new("sum_args")
                    .with_params(vec![Parameter::new(v(0), I64)])
                    .with_return(I64),
            );
            function.sig.variadic = true;
            function.stack_reservations.insert(VirtualReg::gpr(50), 3);
            function.add_block(entry);
            let mut assembler =
                RasAssembler::new(TargetArchitecture::Aarch64, HOST_OS).expect("assembler");
            assert!(
                assembler
                    .compile_mir_to_binary_function(&module_of(vec![function]), None)
                    .is_err()
            );
        }
    }
}
