use super::{gpr, is_stack_pointer};
use crate::encoder::traits::ParsedInstruction;
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

/// Rewrites Apple's dotted NEON spelling to the standard one, so both encode the same.
///
/// clang on Darwin puts the layout on the mnemonic: `fmla.4s v0, v1, v2` and
/// `ld1.4s { v0 }, [x0]`. Every plain vector register then takes the layout
/// (`v0.4s`), and an element `v1[2]` takes its lane letter (`v1.s[2]`). The
/// result is `None` for mnemonics without a layout suffix, so conditions such as
/// `b.eq` pass through unchanged.
pub(crate) fn normalize_dotted(
    inst: &ParsedInstruction,
) -> Result<Option<ParsedInstruction>, RasError> {
    let Some((opcode, suffix)) = inst.opcode.split_once('.') else {
        return Ok(None);
    };
    let (lane, arrangement) = match suffix {
        "8b" | "16b" => ("b", Some(suffix)),
        "4h" | "8h" => ("h", Some(suffix)),
        "2s" | "4s" => ("s", Some(suffix)),
        "2d" => ("d", Some(suffix)),
        "b" | "h" | "s" | "d" => (suffix, None),
        _ => return Ok(None),
    };
    if opcode.is_empty() || opcode.contains('.') {
        return Ok(None);
    }
    let mut tagged = false;
    let mut operands = Vec::with_capacity(inst.operands.len());
    for operand in &inst.operands {
        if let Some(inner) = operand.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
            let Some(arrangement) = arrangement else {
                return Err(error(format!(
                    "{opcode}.{suffix} takes a lane list without a layout"
                )));
            };
            let mut items = Vec::new();
            for item in inner.split(',') {
                let (text, done) = tag_vector(item.trim(), arrangement, lane)?;
                if !done {
                    return Err(error(format!("Expected a vector register in list: {item}")));
                }
                items.push(text);
            }
            tagged = true;
            operands.push(format!("{{ {} }}", items.join(", ")));
        } else {
            let (text, done) = match arrangement {
                Some(arrangement) => tag_vector(operand, arrangement, lane)?,
                None => tag_element(operand, lane)?,
            };
            tagged |= done;
            operands.push(text);
        }
    }
    if !tagged {
        return Err(error(format!(
            "{opcode}.{suffix} needs a vector register operand"
        )));
    }
    Ok(Some(ParsedInstruction {
        opcode: opcode.to_string(),
        operands,
    }))
}

/// Adds the layout to a plain `v<n>` operand, or the lane letter to `v<n>[i]`.
/// Returns the operand and whether it named a vector register.
fn tag_vector(text: &str, arrangement: &str, lane: &str) -> Result<(String, bool), RasError> {
    let Some((number, rest)) = split_vector(text) else {
        return Ok((text.to_string(), false));
    };
    if rest.starts_with('.') {
        return Err(error(format!(
            "{text} already names its layout, which the dotted form forbids"
        )));
    }
    Ok(if rest.is_empty() {
        (format!("v{number}.{arrangement}"), true)
    } else if rest.starts_with('[') {
        (format!("v{number}.{lane}{rest}"), true)
    } else {
        (text.to_string(), false)
    })
}

/// Adds the lane letter to an element operand `v<n>[i]` when the mnemonic has
/// only a lane suffix, as in `mov.s v0[1], w1`.
fn tag_element(text: &str, lane: &str) -> Result<(String, bool), RasError> {
    match split_vector(text) {
        Some((number, rest)) if rest.starts_with('[') => {
            Ok((format!("v{number}.{lane}{rest}"), true))
        }
        Some((_, rest)) if rest.starts_with('.') => Err(error(format!(
            "{text} already names its layout, which the dotted form forbids"
        ))),
        _ => Ok((text.to_string(), false)),
    }
}

/// Splits `v<n>` followed by an optional `.layout` or `[index]` tail.
fn split_vector(text: &str) -> Option<(&str, &str)> {
    let body = text.strip_prefix('v')?;
    let digits = body
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(body.len());
    if digits == 0 {
        return None;
    }
    let (number, rest) = body.split_at(digits);
    if rest.is_empty() || rest.starts_with(['.', '[']) {
        Some((number, rest))
    } else {
        None
    }
}

pub(super) fn encode(op: &str, p: &[String]) -> Result<Option<u32>, RasError> {
    let vector_first = p.first().is_some_and(|s| s.starts_with('v'));
    let word = match op {
        "dup" if vector_first => dup_general(p)?,
        "ins" => ins_general(p)?,
        "umov" => umov(p)?,
        // `mov` is the lane alias of INS and UMOV; vector-to-vector moves are not lane moves.
        "mov" if vector_first && p.get(1).is_some_and(|s| !s.starts_with('v')) => ins_general(p)?,
        "mov" if !vector_first && p.get(1).is_some_and(|s| s.starts_with('v')) => umov(p)?,
        "mov" if vector_first && p.len() == 2 && p[1].starts_with('v') && !p[1].contains('[') => {
            // A vector copy is ORR with the source in both sources.
            let source = p.get(1).cloned().unwrap_or_default();
            vector_arith("orr", &[p[0].clone(), source.clone(), source])?
        }
        "add" | "sub" | "mul" | "and" | "orr" | "eor" | "fadd" | "fsub" | "fmul" | "fdiv"
        | "fmla" | "fmls"
            if vector_first =>
        {
            vector_arith(op, p)?
        }
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

/// The sz bit (bit 22) of a floating-point vector form: 2s and 4s are single
/// precision, 2d is double precision, and half precision is not encoded.
fn float_size(op: &str, arrangement: Arrangement) -> Result<u32, RasError> {
    match arrangement {
        Arrangement::S2 | Arrangement::S4 => Ok(0),
        Arrangement::D2 => Ok(1 << 22),
        _ => Err(error(format!("{op} takes only 2s, 4s or 2d arrangements"))),
    }
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
        "and" | "orr" | "eor" => {
            if !matches!(arrangement, Arrangement::B8 | Arrangement::B16) {
                return Err(error(format!("{op} takes only 8b or 16b arrangements")));
            }
            match op {
                "and" => 0x0E20_1C00,
                "orr" => 0x0EA0_1C00,
                _ => 0x2E20_1C00,
            }
        }
        "fadd" => 0x0E20_D400 | float_size(op, arrangement)?,
        "fsub" => 0x0EA0_D400 | float_size(op, arrangement)?,
        "fmul" => 0x2E20_DC00 | float_size(op, arrangement)?,
        "fdiv" => 0x2E20_FC00 | float_size(op, arrangement)?,
        "fmla" => 0x0E20_CC00 | float_size(op, arrangement)?,
        "fmls" => 0x0EA0_CC00 | float_size(op, arrangement)?,
        _ => return Err(error(format!("Unsupported instruction: {op}"))),
    };
    Ok(base | arrangement.q() | (rm << 16) | (rn << 5) | rd)
}
