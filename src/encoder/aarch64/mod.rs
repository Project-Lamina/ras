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

impl AArch64Encoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn encode(&self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let op = inst.opcode.as_str();
        let p = &inst.operands;
        let arg = |i: usize| {
            p.get(i)
                .map(String::as_str)
                .ok_or_else(|| error(format!("Missing operand for {op}")))
        };
        let r = |i| register(arg(i)?).map(u32::from);
        let wide = p.first().is_some_and(|s| s.starts_with('x') || s == "sp");
        let sf = if wide { 0x8000_0000 } else { 0 };
        let rrr = |base: u32| -> Result<u32, RasError> {
            Ok(base | sf | (r(2)? << 16) | (r(1)? << 5) | r(0)?)
        };
        let word = match op {
            "ret" => 0xd65f_0000 | (if p.is_empty() { 30 } else { r(0)? } << 5),
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
                if matches!(arg(0)?, "sp" | "wsp") {
                    return Err(error(format!("{op} cannot take the stack pointer")));
                }
                let base = if op == "br" { 0xd61f_0000 } else { 0xd63f_0000 };
                base | (r(0)? << 5)
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
                    let value =
                        immediate(src)? as u64 & if wide { u64::MAX } else { u32::MAX as u64 };
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
            "add" | "sub" | "cmp" => {
                let compare = op == "cmp";
                let rn = r(usize::from(!compare))?;
                let rd = if compare { 31 } else { r(0)? };
                let operand = arg(if compare { 1 } else { 2 })?;
                let flags = if compare {
                    0x6000_0000
                } else if op == "sub" {
                    0x4000_0000
                } else {
                    0
                };
                if operand.starts_with('#') {
                    let (imm, shift) = add_sub_imm12(
                        immediate(operand)?,
                        p.get(if compare { 2 } else { 3 }).map(|s| s.trim()),
                    )?;
                    0x1100_0000 | sf | flags | (shift << 22) | ((imm as u32) << 10) | (rn << 5) | rd
                } else {
                    let shift = p.get(if compare { 2 } else { 3 }).map(|s| s.trim());
                    let (kind, amount) = shifted_register(shift, wide)?;
                    if op == "add" && wide && shift.is_none() {
                        return Ok(encode_add_rrr(rd as u8, rn as u8, register(operand)?));
                    }
                    0x0b00_0000
                        | sf
                        | flags
                        | (kind << 22)
                        | (u32::from(register(operand)?) << 16)
                        | (amount << 10)
                        | (rn << 5)
                        | rd
                }
            }
            "and" => {
                return Ok(if wide {
                    encode_and_rrr(r(0)? as u8, r(1)? as u8, r(2)? as u8)
                } else {
                    encode_and_w_rrr(r(0)? as u8, r(1)? as u8, r(2)? as u8)
                });
            }
            "mul" => rrr(0x1b00_7c00)?,
            "msub" => {
                return Ok(if wide {
                    encode_msub(r(0)? as u8, r(1)? as u8, r(2)? as u8, r(3)? as u8)
                } else {
                    encode_msub_w(r(0)? as u8, r(1)? as u8, r(2)? as u8, r(3)? as u8)
                });
            }
            "sdiv" => rrr(0x1ac0_0c00)?,
            "udiv" => rrr(0x1ac0_0800)?,
            "cset" => 0x1a9f_07e0 | sf | ((condition(arg(1)?)? ^ 1) << 12) | r(0)?,
            "csel" => rrr(0x1a80_0000)? | (condition(arg(3)?)? << 12),
            "uxtb" => 0x5300_1c00 | (r(1)? << 5) | r(0)?,
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
                let mem = p[2..].join(",");
                let pre = mem.ends_with('!');
                let post = mem.contains("],");
                let fields: Vec<_> = mem
                    .split(['[', ']', ',', '!'])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect();
                let rn = register(
                    fields
                        .first()
                        .ok_or_else(|| error("Missing pair address"))?,
                )?;
                let offset = fields.get(1).map_or(Ok(0), |s| immediate(s))?;
                if wide && op == "stp" && pre {
                    enc_stp_pre_64(r(0)? as u8, r(1)? as u8, rn, offset as i32)?
                } else if wide && op == "ldp" && post {
                    enc_ldp_post_64(r(0)? as u8, r(1)? as u8, rn, offset as i32)?
                } else {
                    let scale = if wide { 8 } else { 4 };
                    if offset % scale != 0 || !(-64..=63).contains(&(offset / scale)) {
                        return Err(error("Pair offset out of range"));
                    }
                    (if pre {
                        0x2980_0000
                    } else if post {
                        0x2880_0000
                    } else {
                        0x2900_0000
                    }) | if wide { 0x8000_0000 } else { 0 }
                        | if op == "ldp" { 1 << 22 } else { 0 }
                        | (((offset / scale) as u32 & 127) << 15)
                        | (r(1)? << 10)
                        | (u32::from(rn) << 5)
                        | r(0)?
                }
            }
            "ldr" | "str" | "ldrb" | "strb" | "ldrh" | "strh" | "ldur" | "stur" => {
                let mem = p[1..].join(",");
                let inside = mem
                    .trim()
                    .strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .ok_or_else(|| error("Expected base and immediate address"))?;
                let fields: Vec<_> = inside.split(',').map(str::trim).collect();
                let rn = register(fields[0])?;
                let offset = fields.get(1).map_or(Ok(0), |s| immediate(s))?;
                let size = if op.ends_with('b') {
                    0
                } else if op.ends_with('h') {
                    1
                } else if wide {
                    3
                } else {
                    2
                };
                let load = op.starts_with('l');
                let scaled = !op.contains("ur")
                    && offset >= 0
                    && offset % (1 << size) == 0
                    && offset / (1 << size) <= 4095;
                let base = ((size as u32) << 30)
                    | if scaled { 0x3900_0000 } else { 0x3800_0000 }
                    | if load { 1 << 22 } else { 0 };
                let imm = if scaled {
                    ((offset >> size) as u32) << 10
                } else {
                    if !(-256..=255).contains(&offset) {
                        return Err(error(format!("Load/store offset out of range: {offset}")));
                    }
                    ((offset as u32) & 511) << 12
                };
                base | imm | (u32::from(rn) << 5) | r(0)?
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
