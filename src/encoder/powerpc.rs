//! PowerPC64 little-endian (ELFv2) instruction encoder.
//!
//! Every instruction is one 32-bit word stored little-endian. The Power ISA numbers
//! bits from the most significant end; the helpers here place each field by its
//! shift from the least significant bit, using the field layouts of the I, B, D,
//! DS, X, XL, XFX, XO, A, M, MD and MDS instruction formats.
//!
//! A symbolic displacement or immediate is not resolved here. The assembler core
//! encodes a zero placeholder and patches the word or leaves the field to a relocation.

use crate::encoder::traits::{InstructionEncoder, ParsedInstruction};
use crate::error::RasError;

pub const NOP_WORD: u32 = 0x6000_0000;

pub struct PowerPcEncoder {
    position: usize,
}

impl Default for PowerPcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl PowerPcEncoder {
    pub fn new() -> Self {
        Self { position: 0 }
    }
}

/// Bit width of the displacement field a branch mnemonic carries, or `None` if it is
/// not a branch the encoder knows.
pub fn branch_field_bits(name: &str) -> Option<u32> {
    match name {
        "b" | "bl" => Some(24),
        _ => cond_branch(name).map(|_| 14),
    }
}

fn err(msg: impl Into<String>) -> RasError {
    RasError::EncodingError(msg.into())
}

fn need(ops: &[String], count: usize, name: &str) -> Result<(), RasError> {
    if ops.len() == count {
        Ok(())
    } else {
        Err(err(format!(
            "{name} takes {count} operands, got {}",
            ops.len()
        )))
    }
}

fn reg(s: &str, prefix: &str) -> Result<u32, RasError> {
    let t = s.trim();
    let digits = t.strip_prefix(prefix).unwrap_or(t);
    digits
        .parse::<u32>()
        .ok()
        .filter(|&n| n <= 31)
        .ok_or_else(|| err(format!("bad register: {s}")))
}

fn gpr(s: &str) -> Result<u32, RasError> {
    reg(s, "r")
}

fn fpr(s: &str) -> Result<u32, RasError> {
    reg(s, "f")
}

fn crf(s: &str) -> Result<u32, RasError> {
    let t = s.trim();
    let digits = t.strip_prefix("cr").unwrap_or(t);
    digits
        .parse::<u32>()
        .ok()
        .filter(|&n| n <= 7)
        .ok_or_else(|| err(format!("bad condition register field: {s}")))
}

fn imm(s: &str) -> Result<i64, RasError> {
    let t = s.trim();
    let (negative, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let magnitude = match body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        Some(hex) => i64::from_str_radix(hex, 16),
        None => body.parse::<i64>(),
    }
    .map_err(|_| err(format!("bad immediate: {s}")))?;
    Ok(if negative { -magnitude } else { magnitude })
}

fn ufield(value: i64, bits: u32, what: &str) -> Result<u32, RasError> {
    if (0..1i64 << bits).contains(&value) {
        Ok(value as u32)
    } else {
        Err(err(format!("{what} out of range: {value}")))
    }
}

fn sfield(value: i64, bits: u32, what: &str) -> Result<u32, RasError> {
    let limit = 1i64 << (bits - 1);
    if (-limit..limit).contains(&value) {
        Ok(value as u32 & ((1u32 << bits) - 1))
    } else {
        Err(err(format!("{what} out of range: {value}")))
    }
}

fn imm16(value: i64) -> Result<u32, RasError> {
    if (-0x8000..=0xffff).contains(&value) {
        Ok(value as u32 & 0xffff)
    } else {
        Err(err(format!("16-bit immediate out of range: {value}")))
    }
}

fn mem(s: &str) -> Result<(i64, u32), RasError> {
    let t = s.trim();
    let open = t
        .rfind('(')
        .filter(|_| t.ends_with(')'))
        .ok_or_else(|| err(format!("expected D(RA) operand, got {s}")))?;
    let disp = if open == 0 { 0 } else { imm(&t[..open])? };
    let ra = gpr(&t[open + 1..t.len() - 1])?;
    Ok((disp, ra))
}

fn d_form(op: u32, a: u32, b: u32, si: i64) -> Result<u32, RasError> {
    Ok(op << 26 | a << 21 | b << 16 | imm16(si)?)
}

fn ds_form(op: u32, a: u32, b: u32, disp: i64, xo: u32) -> Result<u32, RasError> {
    if disp % 4 != 0 {
        return Err(err(format!(
            "DS displacement must be a multiple of 4: {disp}"
        )));
    }
    let value = sfield(disp, 16, "DS displacement")?;
    Ok(op << 26 | a << 21 | b << 16 | (value & 0xfffc) | xo)
}

fn x_form(op: u32, a: u32, b: u32, c: u32, xo: u32, rc: u32) -> u32 {
    op << 26 | a << 21 | b << 16 | c << 11 | xo << 1 | rc
}

fn xo_form(op: u32, a: u32, b: u32, c: u32, oe: u32, xo: u32, rc: u32) -> u32 {
    op << 26 | a << 21 | b << 16 | c << 11 | oe << 10 | xo << 1 | rc
}

fn a_form(op: u32, frt: u32, fra: u32, frb: u32, frc: u32, xo: u32, rc: u32) -> u32 {
    op << 26 | frt << 21 | fra << 16 | frb << 11 | frc << 6 | xo << 1 | rc
}

fn m_form(op: u32, rs: u32, ra: u32, b: u32, mb: u32, me: u32, rc: u32) -> u32 {
    op << 26 | rs << 21 | ra << 16 | b << 11 | mb << 6 | me << 1 | rc
}

/// MD and MDS store the six-bit mb/me field with its high bit moved to the bottom.
fn six_bit_field(value: u32) -> u32 {
    ((value & 0x1f) << 1) | (value >> 5)
}

fn md_form(op: u32, rs: u32, ra: u32, sh: u32, mb: u32, xo: u32, rc: u32) -> u32 {
    op << 26
        | rs << 21
        | ra << 16
        | (sh & 0x1f) << 11
        | six_bit_field(mb) << 5
        | xo << 2
        | (sh >> 5) << 1
        | rc
}

fn mds_form(op: u32, rs: u32, ra: u32, rb: u32, mb: u32, xo: u32, rc: u32) -> u32 {
    op << 26 | rs << 21 | ra << 16 | rb << 11 | six_bit_field(mb) << 5 | xo << 1 | rc
}

fn xl_form(op: u32, bo: u32, bi: u32, xo: u32, lk: u32) -> u32 {
    op << 26 | bo << 21 | bi << 16 | xo << 1 | lk
}

fn xfx_form(op: u32, rt: u32, spr: u32, xo: u32) -> u32 {
    op << 26 | rt << 21 | ((spr & 0x1f) << 5 | spr >> 5) << 11 | xo << 1
}

fn cmp_x(xo: u32, l: u32, bf: u32, ra: u32, rb: u32) -> u32 {
    31 << 26 | bf << 23 | l << 21 | ra << 16 | rb << 11 | xo << 1
}

fn cmp_d(op: u32, l: u32, bf: u32, ra: u32, si: i64) -> Result<u32, RasError> {
    Ok(op << 26 | bf << 23 | l << 21 | ra << 16 | imm16(si)?)
}

/// Split an optional leading condition field: two operands mean CR0.
fn optional_crf(ops: &[String], name: &str) -> Result<(u32, usize), RasError> {
    match ops.len() {
        2 => Ok((0, 0)),
        3 => Ok((crf(&ops[0])?, 1)),
        _ => Err(err(format!(
            "{name} takes 2 or 3 operands, got {}",
            ops.len()
        ))),
    }
}

/// Condition bit within a CR field, and the BO value for taken-on-true or false.
fn cond_branch(name: &str) -> Option<(u32, u32)> {
    Some(match name {
        "blt" => (12, 0),
        "bge" => (4, 0),
        "bgt" => (12, 1),
        "ble" => (4, 1),
        "beq" => (12, 2),
        "bne" => (4, 2),
        "bun" => (12, 3),
        "bnu" => (4, 3),
        _ => return None,
    })
}

fn branch_disp(op: &str, value: i64, bits: u32) -> Result<u32, RasError> {
    if value % 4 != 0 {
        return Err(err(format!(
            "{op} displacement must be a multiple of 4: {value}"
        )));
    }
    sfield(value >> 2, bits, "branch displacement")
}

fn encode(name: &str, rc: u32, ops: &[String]) -> Result<u32, RasError> {
    let r = |i: usize| gpr(&ops[i]);
    let f = |i: usize| fpr(&ops[i]);
    let no_rc = |what: &str| {
        if rc == 0 {
            Ok(())
        } else {
            Err(err(format!("{what} does not take a '.' suffix")))
        }
    };
    match name {
        "nop" => {
            need(ops, 0, name)?;
            Ok(NOP_WORD)
        }
        "li" | "lis" => {
            need(ops, 2, name)?;
            let op = if name == "li" { 14 } else { 15 };
            d_form(op, r(0)?, 0, imm(&ops[1])?)
        }
        "addi" | "addis" => {
            need(ops, 3, name)?;
            let op = if name == "addi" { 14 } else { 15 };
            d_form(op, r(0)?, r(1)?, imm(&ops[2])?)
        }
        "ori" | "oris" | "xori" | "xoris" => {
            need(ops, 3, name)?;
            let op = match name {
                "ori" => 24,
                "oris" => 25,
                "xori" => 26,
                _ => 27,
            };
            d_form(op, r(1)?, r(0)?, imm(&ops[2])?)
        }
        "andi" | "andis" => {
            need(ops, 3, name)?;
            if rc == 0 {
                return Err(err(format!("{name} must be written {name}.")));
            }
            let op = if name == "andi" { 28 } else { 29 };
            d_form(op, r(1)?, r(0)?, imm(&ops[2])?)
        }
        "lwz" | "lwzu" | "lbz" | "lbzu" | "lhz" | "lhzu" | "lha" | "lhau" | "stw" | "stwu"
        | "stb" | "stbu" | "sth" | "sthu" | "lfs" | "lfsu" | "lfd" | "lfdu" | "stfs" | "stfsu"
        | "stfd" | "stfdu" => {
            need(ops, 2, name)?;
            no_rc(name)?;
            let op = match name {
                "lwz" => 32,
                "lwzu" => 33,
                "lbz" => 34,
                "lbzu" => 35,
                "stw" => 36,
                "stwu" => 37,
                "stb" => 38,
                "stbu" => 39,
                "lhz" => 40,
                "lhzu" => 41,
                "lha" => 42,
                "lhau" => 43,
                "sth" => 44,
                "sthu" => 45,
                "lfs" => 48,
                "lfsu" => 49,
                "lfd" => 50,
                "lfdu" => 51,
                "stfs" => 52,
                "stfsu" => 53,
                "stfd" => 54,
                _ => 55,
            };
            let (disp, ra) = mem(&ops[1])?;
            let t = if op >= 48 { f(0)? } else { r(0)? };
            d_form(op, t, ra, disp)
        }
        "ld" | "ldu" | "lwa" | "std" | "stdu" => {
            need(ops, 2, name)?;
            no_rc(name)?;
            let (op, xo) = match name {
                "ld" => (58, 0),
                "ldu" => (58, 1),
                "lwa" => (58, 2),
                "std" => (62, 0),
                _ => (62, 1),
            };
            let (disp, ra) = mem(&ops[1])?;
            ds_form(op, r(0)?, ra, disp, xo)
        }
        "lbzx" | "lhzx" | "lwzx" | "ldx" | "ldux" | "lbarx" | "lharx" | "lwarx" | "ldarx"
        | "stbx" | "sthx" | "stwx" | "stdx" | "stdux" | "stbcx" | "sthcx" | "stwcx" | "stdcx" => {
            need(ops, 3, name)?;
            let (xo, must_rc) = match name {
                "lbzx" => (87, false),
                "lhzx" => (279, false),
                "lwzx" => (23, false),
                "ldx" => (21, false),
                "ldux" => (53, false),
                "lbarx" => (52, false),
                "lharx" => (116, false),
                "lwarx" => (20, false),
                "ldarx" => (84, false),
                "stbx" => (215, false),
                "sthx" => (407, false),
                "stwx" => (151, false),
                "stdx" => (149, false),
                "stdux" => (181, false),
                "stbcx" => (694, true),
                "sthcx" => (726, true),
                "stwcx" => (150, true),
                _ => (214, true),
            };
            if must_rc && rc == 0 {
                return Err(err(format!("{name} must be written {name}.")));
            }
            if !must_rc {
                no_rc(name)?;
            }
            Ok(x_form(31, r(0)?, r(1)?, r(2)?, xo, rc))
        }
        "cmpd" | "cmpld" | "cmpw" | "cmplw" => {
            let (bf, skip) = optional_crf(ops, name)?;
            let (xo, l) = match name {
                "cmpd" => (0, 1),
                "cmpld" => (32, 1),
                "cmpw" => (0, 0),
                _ => (32, 0),
            };
            no_rc(name)?;
            Ok(cmp_x(xo, l, bf, gpr(&ops[skip])?, gpr(&ops[skip + 1])?))
        }
        "cmpdi" | "cmpwi" | "cmpldi" | "cmplwi" => {
            let (bf, skip) = optional_crf(ops, name)?;
            no_rc(name)?;
            let (op, l) = match name {
                "cmpdi" => (11, 1),
                "cmpwi" => (11, 0),
                "cmpldi" => (10, 1),
                _ => (10, 0),
            };
            cmp_d(op, l, bf, gpr(&ops[skip])?, imm(&ops[skip + 1])?)
        }
        "add" | "subf" | "mulld" | "mullw" | "divd" | "divdu" | "divw" | "divwu" => {
            need(ops, 3, name)?;
            let xo = match name {
                "add" => 266,
                "subf" => 40,
                "mulld" => 233,
                "mullw" => 235,
                "divd" => 489,
                "divdu" => 457,
                "divw" => 491,
                _ => 459,
            };
            Ok(xo_form(31, r(0)?, r(1)?, r(2)?, 0, xo, rc))
        }
        "sub" => {
            need(ops, 3, name)?;
            Ok(xo_form(31, r(0)?, r(2)?, r(1)?, 0, 40, rc))
        }
        "neg" => {
            need(ops, 2, name)?;
            Ok(xo_form(31, r(0)?, r(1)?, 0, 0, 104, rc))
        }
        "and" | "andc" | "or" | "orc" | "xor" | "nor" | "nand" | "eqv" | "sld" | "slw" | "srd"
        | "srw" | "srad" | "sraw" => {
            need(ops, 3, name)?;
            let xo = match name {
                "and" => 28,
                "andc" => 60,
                "or" => 444,
                "orc" => 412,
                "xor" => 316,
                "nor" => 124,
                "nand" => 476,
                "eqv" => 284,
                "slw" => 24,
                "sld" => 27,
                "srw" => 536,
                "srd" => 539,
                "sraw" => 792,
                _ => 794,
            };
            Ok(x_form(31, r(1)?, r(0)?, r(2)?, xo, rc))
        }
        "mr" => {
            need(ops, 2, name)?;
            Ok(x_form(31, r(1)?, r(0)?, r(1)?, 444, rc))
        }
        "extsb" | "extsh" | "extsw" | "cntlzd" | "cntlzw" | "popcntd" | "popcntw" => {
            need(ops, 2, name)?;
            let xo = match name {
                "extsb" => 954,
                "extsh" => 922,
                "extsw" => 986,
                "cntlzd" => 58,
                "cntlzw" => 26,
                "popcntd" => 506,
                _ => 378,
            };
            Ok(x_form(31, r(1)?, r(0)?, 0, xo, rc))
        }
        "isel" => {
            need(ops, 4, name)?;
            no_rc(name)?;
            let bc = imm(&ops[3])?;
            if !(0..32).contains(&bc) {
                return Err(err(format!("isel condition bit out of range: {bc}")));
            }
            Ok(31 << 26 | r(0)? << 21 | r(1)? << 16 | r(2)? << 11 | (bc as u32) << 6 | 15 << 1)
        }
        "rlwinm" | "rlwimi" | "rlwnm" => {
            need(ops, 5, name)?;
            let op = match name {
                "rlwimi" => 20,
                "rlwinm" => 21,
                _ => 23,
            };
            let b = if name == "rlwnm" {
                r(2)?
            } else {
                ufield(imm(&ops[2])?, 5, "shift")?
            };
            let mb = ufield(imm(&ops[3])?, 5, "mb")?;
            let me = ufield(imm(&ops[4])?, 5, "me")?;
            Ok(m_form(op, r(1)?, r(0)?, b, mb, me, rc))
        }
        "rldicl" | "rldicr" | "rldic" | "rldimi" => {
            need(ops, 4, name)?;
            let xo = match name {
                "rldicl" => 0,
                "rldicr" => 1,
                "rldic" => 2,
                _ => 3,
            };
            let sh = ufield(imm(&ops[2])?, 6, "shift")?;
            let m = ufield(imm(&ops[3])?, 6, "mask bound")?;
            Ok(md_form(30, r(1)?, r(0)?, sh, m, xo, rc))
        }
        "rldcl" | "rldcr" => {
            need(ops, 4, name)?;
            let xo = if name == "rldcl" { 8 } else { 9 };
            let m = ufield(imm(&ops[3])?, 6, "mask bound")?;
            Ok(mds_form(30, r(1)?, r(0)?, r(2)?, m, xo, rc))
        }
        "clrldi" => {
            need(ops, 3, name)?;
            let n = ufield(imm(&ops[2])?, 6, "clear count")?;
            Ok(md_form(30, r(1)?, r(0)?, 0, n, 0, rc))
        }
        "srdi" => {
            need(ops, 3, name)?;
            let n = ufield(imm(&ops[2])?, 6, "shift")?;
            Ok(md_form(30, r(1)?, r(0)?, (64 - n) % 64, n, 0, rc))
        }
        "sldi" => {
            need(ops, 3, name)?;
            let n = ufield(imm(&ops[2])?, 6, "shift")?;
            Ok(md_form(30, r(1)?, r(0)?, n, 63 - n, 1, rc))
        }
        "rotldi" => {
            need(ops, 3, name)?;
            let n = ufield(imm(&ops[2])?, 6, "rotate")?;
            Ok(md_form(30, r(1)?, r(0)?, n, 0, 0, rc))
        }
        "rotld" => {
            need(ops, 3, name)?;
            Ok(mds_form(30, r(1)?, r(0)?, r(2)?, 0, 8, rc))
        }
        "b" | "bl" => {
            need(ops, 1, name)?;
            no_rc(name)?;
            let lk = u32::from(name == "bl");
            let li = branch_disp(name, imm(&ops[0])?, 24)?;
            Ok(18 << 26 | li << 2 | lk)
        }
        "blr" | "bctr" | "bctrl" => {
            need(ops, 0, name)?;
            let (xo, lk) = match name {
                "blr" => (16, 0),
                "bctr" => (528, 0),
                _ => (528, 1),
            };
            Ok(xl_form(19, 20, 0, xo, lk))
        }
        "isync" => {
            need(ops, 0, name)?;
            Ok(xl_form(19, 0, 0, 150, 0))
        }
        "sync" | "lwsync" => {
            need(ops, 0, name)?;
            let l = u32::from(name == "lwsync");
            Ok(31 << 26 | l << 21 | 598 << 1)
        }
        "mflr" | "mfctr" => {
            need(ops, 1, name)?;
            let spr = if name == "mflr" { 8 } else { 9 };
            Ok(xfx_form(31, r(0)?, spr, 339))
        }
        "mtlr" | "mtctr" => {
            need(ops, 1, name)?;
            let spr = if name == "mtlr" { 8 } else { 9 };
            Ok(xfx_form(31, r(0)?, spr, 467))
        }
        "trap" => {
            need(ops, 0, name)?;
            Ok(x_form(31, 31, 0, 0, 4, 0))
        }
        "tw" | "td" => {
            need(ops, 3, name)?;
            let xo = if name == "tw" { 4 } else { 68 };
            Ok(x_form(
                31,
                ufield(imm(&ops[0])?, 5, "TO")?,
                r(1)?,
                r(2)?,
                xo,
                rc,
            ))
        }
        "twi" | "tdi" => {
            need(ops, 3, name)?;
            let op = if name == "twi" { 3 } else { 2 };
            let to = ufield(imm(&ops[0])?, 5, "TO")?;
            d_form(op, to, r(1)?, imm(&ops[2])?)
        }
        "fadd" | "fsub" | "fdiv" | "fadds" | "fsubs" | "fdivs" => {
            need(ops, 3, name)?;
            let (op, xo) = match name {
                "fadd" => (63, 21),
                "fsub" => (63, 20),
                "fdiv" => (63, 18),
                "fadds" => (59, 21),
                "fsubs" => (59, 20),
                _ => (59, 18),
            };
            Ok(a_form(op, f(0)?, f(1)?, f(2)?, 0, xo, rc))
        }
        "fmul" | "fmuls" => {
            need(ops, 3, name)?;
            let op = if name == "fmul" { 63 } else { 59 };
            Ok(a_form(op, f(0)?, f(1)?, 0, f(2)?, 25, rc))
        }
        "fsqrt" | "fsqrts" => {
            need(ops, 2, name)?;
            let op = if name == "fsqrt" { 63 } else { 59 };
            Ok(a_form(op, f(0)?, 0, f(1)?, 0, 22, rc))
        }
        "fneg" | "fmr" | "frsp" | "fctid" | "fctidz" | "fctidu" | "fctiduz" | "fcfid"
        | "fcfidu" | "fcfids" | "fcfidus" => {
            need(ops, 2, name)?;
            let (op, xo) = match name {
                "fneg" => (63, 40),
                "fmr" => (63, 72),
                "frsp" => (63, 12),
                "fctid" => (63, 814),
                "fctidz" => (63, 815),
                "fctidu" => (63, 942),
                "fctiduz" => (63, 943),
                "fcfid" => (63, 846),
                "fcfidu" => (63, 974),
                "fcfids" => (59, 846),
                _ => (59, 974),
            };
            Ok(x_form(op, f(0)?, 0, f(1)?, xo, rc))
        }
        "fcmpu" | "fcmpo" => {
            let (bf, skip) = optional_crf(ops, name)?;
            no_rc(name)?;
            let xo = if name == "fcmpu" { 0 } else { 32 };
            Ok(63 << 26 | bf << 23 | f(skip)? << 16 | f(skip + 1)? << 11 | xo << 1)
        }
        _ => match cond_branch(name) {
            Some((bo, bit)) => {
                no_rc(name)?;
                let (cr, target) = match ops.len() {
                    1 => (0, 0),
                    2 => (crf(&ops[0])?, 1),
                    _ => return Err(err(format!("{name} takes 1 or 2 operands"))),
                };
                let bd = branch_disp(name, imm(&ops[target])?, 14)?;
                Ok(16 << 26 | bo << 21 | (4 * cr + bit) << 16 | bd << 2)
            }
            None => Err(err(format!("unsupported PowerPC instruction: {name}"))),
        },
    }
}

impl InstructionEncoder for PowerPcEncoder {
    fn encode_instruction(&mut self, inst: &ParsedInstruction) -> Result<Vec<u8>, RasError> {
        let lowered = inst.opcode.to_lowercase();
        let (name, rc) = match lowered.strip_suffix('.') {
            Some(base) => (base, true),
            None => (lowered.as_str(), false),
        };
        let word = encode(name, u32::from(rc), &inst.operands)?;
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

    fn word(text: &str) -> u32 {
        let mut parts = text.splitn(2, ' ');
        let opcode = parts.next().unwrap().to_string();
        let operands = parts
            .next()
            .map(|rest| rest.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default();
        let bytes = PowerPcEncoder::new()
            .encode_instruction(&ParsedInstruction { opcode, operands })
            .unwrap();
        u32::from_le_bytes(bytes.try_into().unwrap())
    }

    #[test]
    fn xo_and_x_forms_place_fields_by_shift() {
        assert_eq!(
            word("add 3, 3, 4"),
            31 << 26 | 3 << 21 | 3 << 16 | 4 << 11 | 266 << 1
        );
        assert_eq!(word("subf 0, 4, 3"), 31 << 26 | 4 << 16 | 3 << 11 | 40 << 1);
        assert_eq!(word("sub 3, 3, 4"), word("subf 3, 4, 3"));
        assert_eq!(
            word("mr 4, 1"),
            31 << 26 | 1 << 21 | 4 << 16 | 1 << 11 | 444 << 1
        );
        assert_eq!(word("stdcx. 0, 0, 5"), 31 << 26 | 5 << 11 | 214 << 1 | 1);
        assert_eq!(word("trap"), 0x7fe0_0008);
        assert_eq!(word("lwsync"), 0x7c20_04ac);
        assert_eq!(word("sync"), 0x7c00_04ac);
        assert_eq!(word("isync"), 0x4c00_012c);
    }

    #[test]
    fn d_and_ds_forms() {
        assert_eq!(word("addi 3, 0, 1"), 14 << 26 | 3 << 21 | 1);
        assert_eq!(word("li 3, -1"), 14 << 26 | 3 << 21 | 0xffff);
        assert_eq!(word("ori 0, 0, 0"), NOP_WORD);
        assert_eq!(word("nop"), NOP_WORD);
        assert_eq!(word("ld 0, 16(1)"), 0xe801_0010);
        assert_eq!(word("std 0, 16(1)"), 0xf801_0010);
        assert_eq!(word("stdu 1, -144(1)"), 0xf821_ff71);
        assert_eq!(word("lfd 1, 408(1)"), 50 << 26 | 1 << 21 | 1 << 16 | 408);
    }

    #[test]
    fn md_and_m_forms_split_the_six_bit_mask() {
        assert_eq!(
            word("rldicl 4, 4, 0, 58"),
            30 << 26 | 4 << 21 | 4 << 16 | ((58 & 31) << 1 | 58 >> 5) << 5
        );
        assert_eq!(word("sldi 3, 3, 8"), word("rldicr 3, 3, 8, 55"));
        assert_eq!(word("srdi 4, 3, 8"), word("rldicl 4, 3, 56, 8"));
        assert_eq!(word("rldicr 11, 11, 32, 31"), word("sldi 11, 11, 32"));
        assert_eq!(
            word("rlwnm 3, 3, 4, 0, 31"),
            23 << 26 | 3 << 21 | 3 << 16 | 4 << 11 | 31 << 1
        );
    }

    #[test]
    fn spr_moves_store_the_split_field() {
        assert_eq!(word("mflr 0"), 0x7c08_02a6);
        assert_eq!(word("mtlr 0"), 0x7c08_03a6);
        assert_eq!(word("mtctr 12"), 0x7d89_03a6);
        assert_eq!(word("blr"), 0x4e80_0020);
        assert_eq!(word("bctrl"), 0x4e80_0421);
    }

    #[test]
    fn branches_encode_displacements_and_conditions() {
        assert_eq!(word("b 8"), 0x4800_0008);
        assert_eq!(word("bl 0"), 0x4800_0001);
        assert_eq!(word("beq 8"), 0x4182_0008);
        assert_eq!(word("bne 8"), 0x4082_0008);
        assert_eq!(word("bun 4"), 16 << 26 | 12 << 21 | 3 << 16 | 4);
        assert_eq!(word("bge cr1, 4"), 16 << 26 | 4 << 21 | 4 << 16 | 4);
    }

    #[test]
    fn compares_and_floating_point() {
        assert_eq!(word("cmpd 3, 4"), 31 << 26 | 1 << 21 | 3 << 16 | 4 << 11);
        assert_eq!(word("cmpdi 5, 0"), 11 << 26 | 1 << 21 | 5 << 16);
        assert_eq!(
            word("fadd 1, 2, 3"),
            63 << 26 | 1 << 21 | 2 << 16 | 3 << 11 | 21 << 1
        );
        assert_eq!(
            word("fmul 1, 2, 3"),
            63 << 26 | 1 << 21 | 2 << 16 | 3 << 6 | 25 << 1
        );
        assert_eq!(word("fcmpu 0, 1, 1"), 63 << 26 | 1 << 16 | 1 << 11);
        assert_eq!(word("fcfids 1, 1"), 59 << 26 | 1 << 21 | 1 << 11 | 846 << 1);
    }

    #[test]
    fn rejects_bad_operands() {
        let encode = |opcode: &str, operands: &[&str]| {
            PowerPcEncoder::new().encode_instruction(&ParsedInstruction {
                opcode: opcode.into(),
                operands: operands.iter().map(|s| s.to_string()).collect(),
            })
        };
        assert!(encode("addi", &["3", "0", "70000"]).is_err());
        assert!(encode("ld", &["3", "2(1)"]).is_err());
        assert!(encode("stdcx", &["0", "0", "5"]).is_err());
        assert!(encode("bl", &["6"]).is_err());
        assert!(encode("fnord", &[]).is_err());
    }
}
