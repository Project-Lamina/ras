use super::{gpr, is_stack_pointer};
use crate::error::RasError;

fn error(message: impl Into<String>) -> RasError {
    RasError::EncodingError(message.into())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    B,
    H,
    S,
    D,
}

impl Lane {
    fn count(self) -> u32 {
        match self {
            Lane::B => 16,
            Lane::H => 8,
            Lane::S => 4,
            Lane::D => 2,
        }
    }

    /// The imm5 field for element `index`: the lowest set bit names the size and
    /// the bits above it hold the index.
    fn imm5(self, index: u32) -> u32 {
        match self {
            Lane::B => (index << 1) | 0b1,
            Lane::H => (index << 2) | 0b10,
            Lane::S => (index << 3) | 0b100,
            Lane::D => (index << 4) | 0b1000,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arrangement {
    B8,
    B16,
    H4,
    H8,
    S2,
    S4,
    D2,
}

impl Arrangement {
    fn parse(text: &str) -> Result<Self, RasError> {
        Ok(match text {
            "8b" => Arrangement::B8,
            "16b" => Arrangement::B16,
            "4h" => Arrangement::H4,
            "8h" => Arrangement::H8,
            "2s" => Arrangement::S2,
            "4s" => Arrangement::S4,
            "2d" => Arrangement::D2,
            _ => return Err(error(format!("Unsupported vector arrangement: {text}"))),
        })
    }

    fn lane(self) -> Lane {
        match self {
            Arrangement::B8 | Arrangement::B16 => Lane::B,
            Arrangement::H4 | Arrangement::H8 => Lane::H,
            Arrangement::S2 | Arrangement::S4 => Lane::S,
            Arrangement::D2 => Lane::D,
        }
    }

    /// The Q bit (bit 30), set for the 128-bit arrangements.
    fn q(self) -> u32 {
        match self {
            Arrangement::B8 | Arrangement::H4 | Arrangement::S2 => 0,
            _ => 0x4000_0000,
        }
    }

    /// The size field (bits 23:22) of the integer vector encodings.
    fn size(self) -> u32 {
        match self {
            Arrangement::B8 | Arrangement::B16 => 0,
            Arrangement::H4 | Arrangement::H8 => 1 << 22,
            Arrangement::S2 | Arrangement::S4 => 2 << 22,
            Arrangement::D2 => 3 << 22,
        }
    }
}

pub(super) fn encode(op: &str, p: &[String]) -> Result<Option<u32>, RasError> {
    let vector_first = p.first().is_some_and(|s| s.starts_with('v'));
    let word = match op {
        "dup" if vector_first => dup_general(p)?,
        "ins" => ins_general(p)?,
        "umov" => umov(p)?,
        "add" | "sub" | "mul" | "and" | "eor" | "fmul" if vector_first => vector_arith(op, p)?,
        _ => return Ok(None),
    };
    Ok(Some(word))
}

fn exact(p: &[String], count: usize) -> Result<&[String], RasError> {
    if p.len() == count {
        Ok(p)
    } else {
        Err(error(format!(
            "Expected {count} operands, found {}",
            p.len()
        )))
    }
}

fn vector_register(text: &str) -> Result<(u32, Arrangement), RasError> {
    let (number, arrangement) = text
        .strip_prefix('v')
        .and_then(|body| body.split_once('.'))
        .ok_or_else(|| error(format!("Expected a v<n>.<T> register: {text}")))?;
    let number = number
        .parse::<u32>()
        .ok()
        .filter(|n| *n < 32)
        .ok_or_else(|| error(format!("Invalid vector register: {text}")))?;
    Ok((number, Arrangement::parse(arrangement)?))
}

fn element(text: &str) -> Result<(u32, Lane, u32), RasError> {
    let invalid = || {
        error(format!(
            "Expected a v<n>.<b|h|s|d>[<index>] element: {text}"
        ))
    };
    let (number, rest) = text
        .strip_prefix('v')
        .and_then(|body| body.split_once('.'))
        .ok_or_else(invalid)?;
    let number = number
        .parse::<u32>()
        .ok()
        .filter(|n| *n < 32)
        .ok_or_else(invalid)?;
    let lane = match rest.as_bytes().first() {
        Some(b'b') => Lane::B,
        Some(b'h') => Lane::H,
        Some(b's') => Lane::S,
        Some(b'd') => Lane::D,
        _ => return Err(invalid()),
    };
    let index = rest
        .get(1..)
        .and_then(|s| s.strip_prefix('['))
        .and_then(|s| s.strip_suffix(']'))
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|i| *i < lane.count())
        .ok_or_else(invalid)?;
    Ok((number, lane, index))
}

/// A general register used as a source or destination of a lane move, with
/// `x` for doublewords and `w` for every other lane size.
fn lane_gpr(reg: &str, lane: Lane) -> Result<u32, RasError> {
    if is_stack_pointer(reg) {
        return Err(error(format!("Stack pointer not allowed here: {reg}")));
    }
    let (number, wide) = gpr(reg)?;
    if wide != (lane == Lane::D) {
        let want = if lane == Lane::D { "x" } else { "w" };
        return Err(error(format!(
            "Expected a {want} register for this lane: {reg}"
        )));
    }
    Ok(number)
}

/// DUP (general), 0 Q 0 01110 000 imm5 0 0001 1 Rn Rd.
fn dup_general(p: &[String]) -> Result<u32, RasError> {
    let p = exact(p, 2)?;
    let (rd, arrangement) = vector_register(&p[0])?;
    let lane = arrangement.lane();
    let (rn, wide) = gpr(&p[1])?;
    if is_stack_pointer(&p[1]) {
        return Err(error(format!("Stack pointer not allowed here: {}", p[1])));
    }
    if wide != (lane == Lane::D) {
        let want = if lane == Lane::D { "x" } else { "w" };
        return Err(error(format!(
            "dup {} needs a {want} source register",
            p[0]
        )));
    }
    Ok(0x0E00_0C00 | arrangement.q() | (Lane::imm5(lane, 0) << 16) | (rn << 5) | rd)
}

/// INS (general), 0 1 0 01110 000 imm5 0 0011 1 Rn Rd.
fn ins_general(p: &[String]) -> Result<u32, RasError> {
    let p = exact(p, 2)?;
    let (rd, lane, index) = element(&p[0])?;
    let rn = lane_gpr(&p[1], lane)?;
    Ok(0x4E00_1C00 | (lane.imm5(index) << 16) | (rn << 5) | rd)
}

/// UMOV, 0 Q 0 01110 000 imm5 0 0111 1 Rn Rd, with Q set for doubleword lanes.
fn umov(p: &[String]) -> Result<u32, RasError> {
    let p = exact(p, 2)?;
    let (rn, lane, index) = element(&p[1])?;
    let rd = lane_gpr(&p[0], lane)?;
    let q = if lane == Lane::D { 0x4000_0000 } else { 0 };
    Ok(0x0E00_3C00 | q | (lane.imm5(index) << 16) | (rn << 5) | rd)
}

/// Three-register vector forms over one arrangement.
fn vector_arith(op: &str, p: &[String]) -> Result<u32, RasError> {
    let p = exact(p, 3)?;
    let (rd, arrangement) = vector_register(&p[0])?;
    let (rn, rn_arrangement) = vector_register(&p[1])?;
    let (rm, rm_arrangement) = vector_register(&p[2])?;
    if rn_arrangement != arrangement || rm_arrangement != arrangement {
        return Err(error(format!("Mismatched vector arrangements for {op}")));
    }
    let base = match op {
        "add" => 0x0E20_8400 | arrangement.size(),
        "sub" => 0x2E20_8400 | arrangement.size(),
        "mul" => {
            if arrangement == Arrangement::D2 {
                return Err(error("mul has no 2d arrangement"));
            }
            0x0E20_9C00 | arrangement.size()
        }
        "and" | "eor" => {
            if !matches!(arrangement, Arrangement::B8 | Arrangement::B16) {
                return Err(error(format!("{op} takes only 8b or 16b arrangements")));
            }
            if op == "and" {
                0x0E20_1C00
            } else {
                0x2E20_1C00
            }
        }
        "fmul" => match arrangement {
            Arrangement::S2 | Arrangement::S4 => 0x2E20_DC00,
            Arrangement::D2 => 0x2E20_DC00 | (1 << 22),
            _ => return Err(error("fmul takes only 2s, 4s or 2d arrangements")),
        },
        _ => return Err(error(format!("Unsupported instruction: {op}"))),
    };
    Ok(base | arrangement.q() | (rm << 16) | (rn << 5) | rd)
}
