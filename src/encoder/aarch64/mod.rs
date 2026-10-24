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
        _ => reg
            .strip_prefix(['x', 'w', 's', 'd'])
            .and_then(|n| n.parse::<u8>().ok())
            .filter(|n| *n < 31)
            .ok_or_else(|| error(format!("Invalid AArch64 register: {reg}"))),
    }
}

fn immediate(text: &str) -> Result<i64, RasError> {
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
    let compare = op == "cmp";
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
    let (kind, amount) = shifted_register(tail, wide)?;
    Ok(0x0b00_0000 | sf | flags | (kind << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd)
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
    .position(|s| *s == text)
    .map(|n| n as u32)
    .ok_or_else(|| error(format!("Invalid condition: {text}")))
}

fn max_operands(op: &str) -> Option<usize> {
    Some(match op {
        "nop" | "yield" | "wfe" | "wfi" | "sev" | "sevl" => 0,
        "ret" | "br" | "blr" | "svc" | "hvc" | "smc" | "brk" | "hlt" | "dmb" => 1,
        "mov" | "uxtb" | "fmov" | "cset" => 2,
        "movz" | "movk" | "mul" | "sdiv" | "udiv" | "fmul" => 3,
        "csel" | "msub" => 4,
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
            "dmb" if arg(0)? == "sy" => 0xd503_3fbf,
            "movz" | "movk" => {
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
                (if op == "movz" {
                    0x5280_0000
                } else {
                    0x7280_0000
                }) | sf
                    | ((shift as u32 / 16) << 21)
                    | ((imm as u32) << 5)
                    | r(0)?
            }
            "mov" => {
                let dst = r(0)?;
                let src = arg(1)?;
                if src.starts_with('#') {
                    let raw = immediate(src)?;
                    if !wide && !(-(1 << 31)..=u32::MAX as i64).contains(&raw) {
                        return Err(error(format!(
                            "mov immediate out of range for w register: {raw}"
                        )));
                    }
                    let value = raw as u64 & if wide { u64::MAX } else { u32::MAX as u64 };
                    let chunks = if wide { 4 } else { 2 };
                    let mut encoding = None;
                    for i in 0..chunks {
                        let mask = 0xffffu64 << (i * 16);
                        if value & !mask == 0 {
                            encoding = Some(
                                0x5280_0000
                                    | sf
                                    | (i << 21)
                                    | (((value >> (i * 16)) as u32) << 5)
                                    | dst,
                            );
                            break;
                        }
                        if value | mask == if wide { u64::MAX } else { u32::MAX as u64 } {
                            encoding = Some(
                                0x1280_0000
                                    | sf
                                    | (i << 21)
                                    | (((!value >> (i * 16)) as u32 & 0xffff) << 5)
                                    | dst,
                            );
                            break;
                        }
                    }
                    encoding.ok_or_else(|| {
                        error("mov immediate requires a single move-wide instruction")
                    })?
                } else if arg(0)? == "sp" || src == "sp" || arg(0)? == "wsp" || src == "wsp" {
                    0x1100_0000 | sf | (r(1)? << 5) | dst
                } else {
                    0x2a00_03e0 | sf | (r(1)? << 16) | dst
                }
            }
            "add" | "sub" | "cmp" => encode_add_sub(op, p)?,
            "and" => {
                let ([rd, rn, rm], wide) = gpr_regs::<3>(p)?;
                return Ok(if wide {
                    encode_and_rrr(rd as u8, rn as u8, rm as u8)
                } else {
                    encode_and_w_rrr(rd as u8, rn as u8, rm as u8)
                });
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
            "fmov" => {
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
            "fmul" => {
                (if arg(0)?.starts_with('d') {
                    0x1e60_0800
                } else {
                    0x1e20_0800
                }) | (r(2)? << 16)
                    | (r(1)? << 5)
                    | r(0)?
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
            "ldr" | "str" | "ldrb" | "strb" | "ldrh" | "strh" | "ldur" | "stur" => {
                let mem = p.get(1..).unwrap_or_default().join(",");
                let inside = mem
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
                let narrow = op.ends_with('b') || op.ends_with('h');
                let (size, fp) = match (narrow, kind) {
                    (true, DataKind::W) => (if op.ends_with('b') { 0 } else { 1 }, false),
                    (true, _) => {
                        return Err(error(format!("{op} takes a 32-bit register")));
                    }
                    (false, DataKind::W) => (2, false),
                    (false, DataKind::X) => (3, false),
                    (false, DataKind::S) => (2, true),
                    (false, DataKind::D) => (3, true),
                };
                let offset = fields.get(1).map_or(Ok(0), |s| immediate(s))?;
                let load = op.starts_with('l');
                let scaled = !op.contains("ur")
                    && offset >= 0
                    && offset % (1 << size) == 0
                    && offset / (1 << size) <= 4095;
                let base = ((size as u32) << 30)
                    | if scaled { 0x3900_0000 } else { 0x3800_0000 }
                    | if fp { 0x0400_0000 } else { 0 }
                    | if load { 1 << 22 } else { 0 };
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
