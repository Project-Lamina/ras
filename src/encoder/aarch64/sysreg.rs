use super::{gpr, immediate, is_stack_pointer};
use crate::error::RasError;

fn error(message: impl Into<String>) -> RasError {
    RasError::EncodingError(message.into())
}

fn two<'a>(p: &'a [String], op: &str) -> Result<(&'a str, &'a str), RasError> {
    match p {
        [a, b] => Ok((a.trim(), b.trim())),
        _ => Err(error(format!("{op} takes two operands"))),
    }
}

fn transfer_register(reg: &str, op: &str) -> Result<u32, RasError> {
    let (number, wide) = gpr(reg)?;
    if !wide || is_stack_pointer(reg) {
        return Err(error(format!(
            "{op} takes a 64-bit general purpose register: {reg}"
        )));
    }
    Ok(number)
}

/// MRS Xt, <sysreg>: 1101 0101 0011 op0[0] op1 CRn CRm op2 Rt.
pub(super) fn encode_mrs(p: &[String]) -> Result<u32, RasError> {
    let (rt, sysreg) = two(p, "mrs")?;
    let field = system_register_field(sysreg)?;
    Ok(0xD530_0000 | field | transfer_register(rt, "mrs")?)
}

pub(super) fn encode_msr(p: &[String]) -> Result<u32, RasError> {
    let (first, second) = two(p, "msr")?;
    if second.starts_with('#') {
        let (op1, op2, max) = match first.to_ascii_lowercase().as_str() {
            "daifset" => (3, 6, 15),
            "daifclr" => (3, 7, 15),
            "spsel" => (0, 5, 1),
            _ => {
                return Err(error(format!(
                    "Unknown PSTATE field for msr immediate: {first}"
                )));
            }
        };
        let imm = immediate(second)?;
        if !(0..=max).contains(&imm) {
            return Err(error(format!("msr {first} immediate out of range: {imm}")));
        }
        return Ok(0xD500_401F | (op1 << 16) | ((imm as u32) << 8) | (op2 << 5));
    }
    let field = system_register_field(first)?;
    Ok(0xD510_0000 | field | transfer_register(second, "msr")?)
}

/// Named system registers as (op0, op1, CRn, CRm, op2), from the ARM ARM (DDI 0487)
/// system register encodings.
const NAMED: &[(&str, [u32; 5])] = &[
    ("nzcv", [3, 3, 4, 2, 0]),
    ("daif", [3, 3, 4, 2, 1]),
    ("fpcr", [3, 3, 4, 4, 0]),
    ("fpsr", [3, 3, 4, 4, 1]),
    ("currentel", [3, 0, 4, 2, 2]),
    ("sp_el0", [3, 0, 4, 1, 0]),
    ("spsr_el1", [3, 0, 4, 0, 0]),
    ("elr_el1", [3, 0, 4, 0, 1]),
    ("sctlr_el1", [3, 0, 1, 0, 0]),
    ("esr_el1", [3, 0, 5, 2, 0]),
    ("vbar_el1", [3, 0, 12, 0, 0]),
    ("tpidr_el1", [3, 0, 13, 0, 4]),
    ("tpidr_el0", [3, 3, 13, 0, 2]),
    ("tpidrro_el0", [3, 3, 13, 0, 3]),
    ("cntfrq_el0", [3, 3, 14, 0, 0]),
    ("cntpct_el0", [3, 3, 14, 0, 1]),
    ("cntvct_el0", [3, 3, 14, 0, 2]),
    ("ctr_el0", [3, 3, 0, 0, 1]),
    ("dczid_el0", [3, 3, 0, 0, 7]),
    ("midr_el1", [3, 0, 0, 0, 0]),
    ("mpidr_el1", [3, 0, 0, 0, 5]),
];

fn decimal(text: &str) -> Option<u32> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// The `s<op0>_<op1>_c<CRn>_c<CRm>_<op2>` spelling that GAS accepts for any system register.
fn generic_fields(name: &str) -> Option<[u32; 5]> {
    let parts: Vec<&str> = name.strip_prefix('s')?.split('_').collect();
    let [op0, op1, crn, crm, op2] = parts.as_slice() else {
        return None;
    };
    let coord = |text: &str| text.strip_prefix('c').and_then(decimal);
    Some([
        decimal(op0)?,
        decimal(op1)?,
        coord(crn)?,
        coord(crm)?,
        decimal(op2)?,
    ])
}

/// The 15-bit system register field at bits 19:5 of MRS and MSR (register). op0 is 2 or 3, so
/// only its low bit is stored; bit 20 is fixed in the base word.
pub(super) fn system_register_field(name: &str) -> Result<u32, RasError> {
    let lower = name.to_ascii_lowercase();
    let [op0, op1, crn, crm, op2] = match NAMED.iter().find(|(known, _)| *known == lower) {
        Some((_, fields)) => *fields,
        None => generic_fields(&lower)
            .ok_or_else(|| error(format!("Unknown system register: {name}")))?,
    };
    if !(2..=3).contains(&op0) || op1 > 7 || crn > 15 || crm > 15 || op2 > 7 {
        return Err(error(format!(
            "System register fields out of range: {name}"
        )));
    }
    Ok(((op0 & 1) << 19) | (op1 << 16) | (crn << 12) | (crm << 8) | (op2 << 5))
}
