use crate::error::RasError;
use lamina_platform::TargetArchitecture;
use std::{collections::HashMap, fmt::Write};

struct Frame {
    start: String,
    last: String,
    instructions: String,
    personality: Option<(u8, String)>,
    lsda: Option<(u8, String)>,
    cfa_offset: i128,
    cfa_register: i128,
    states: Vec<(i128, i128)>,
    simple: bool,
    signal: bool,
    return_column: i128,
}

#[derive(PartialEq, Eq, Hash)]
struct CieKey {
    simple: bool,
    signal: bool,
    return_column: i128,
    personality: Option<(u8, String)>,
    lsda_encoding: Option<u8>,
}

pub(crate) fn advance(value: i128) -> Result<Vec<u8>, RasError> {
    let value = u32::try_from(value)
        .map_err(|_| RasError::EncodingError("CFI location advance out of range".into()))?;
    Ok(match value {
        0 => Vec::new(),
        1..=63 => vec![0x40 | value as u8],
        64..=255 => vec![2, value as u8],
        256..=65535 => {
            let mut bytes = vec![3];
            bytes.extend(value.to_le_bytes()[..2].iter());
            bytes
        }
        _ => {
            let mut bytes = vec![4];
            bytes.extend(value.to_le_bytes());
            bytes
        }
    })
}

fn cfa(opcode: u8, register: Option<i128>, offset: i128) -> Result<Vec<u8>, RasError> {
    let mut bytes = vec![opcode];
    if let Some(register) = register {
        if register < 0 {
            return Err(RasError::ParseError("Invalid CFA register".into()));
        }
        bytes.extend(crate::parser::leb128(register, false));
    }
    let unsigned = crate::parser::leb128(offset.max(0), false);
    let signed = crate::parser::leb128(offset / -8, true);
    if offset % 8 == 0 && (offset < 0 || signed.len() < unsigned.len()) {
        bytes[0] = if opcode == 12 { 0x12 } else { 0x13 };
        bytes.extend(signed);
    } else if offset >= 0 {
        bytes.extend(unsigned);
    } else {
        return Err(RasError::ParseError("Invalid CFA offset".into()));
    }
    Ok(bytes)
}

fn number(value: &str) -> Result<i128, RasError> {
    crate::parser::parse_integer(value)
}

fn register(value: &str, arch: TargetArchitecture) -> Result<i128, RasError> {
    if let Ok(value) = number(value) {
        return Ok(value);
    }
    let value = value.trim_start_matches('%');
    let names = [
        "rax", "rdx", "rcx", "rbx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15", "rip",
    ];
    let reg = if arch == TargetArchitecture::X86_64 {
        names
            .iter()
            .position(|name| *name == value)
            .map(|n| n as i128)
    } else if value == "sp" {
        Some(31)
    } else {
        value.strip_prefix('x').and_then(|n| n.parse().ok())
    };
    reg.ok_or_else(|| RasError::ParseError(format!("Unsupported CFI register: {value}")))
}

fn pointer(encoding: u8, symbol: &str) -> Result<String, RasError> {
    match encoding {
        0 => Ok(format!(".quad {symbol}\n")),
        0x1b | 0x9b => Ok(format!(".long {symbol}-.\n")),
        _ => Err(RasError::ParseError(format!(
            "Unsupported CFI pointer encoding: {encoding:#x}"
        ))),
    }
}

pub(crate) fn lower(
    source: &str,
    arch: TargetArchitecture,
) -> Result<(String, Option<String>), RasError> {
    if !source.contains(".cfi_") {
        return Ok((source.into(), None));
    }
    if !matches!(
        arch,
        TargetArchitecture::X86_64 | TargetArchitecture::Aarch64
    ) {
        return Err(RasError::ParseError("CFI unsupported for target".into()));
    }
    let mut prefix = ".Lras_cfi".to_string();
    while source.contains(&prefix) {
        prefix.push('_');
    }
    let mut output = String::new();
    let mut frames = String::new();
    let mut cies: HashMap<CieKey, String> = HashMap::new();
    let mut frame: Option<Frame> = None;
    let mut serial = 0;
    for line in source.lines() {
        let line = line.trim();
        if !line.starts_with(".cfi_") {
            writeln!(output, "{line}").unwrap();
            continue;
        }
        let (directive, args) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let args: Vec<_> = args.split(',').map(str::trim).collect();
        let invalid = || RasError::ParseError(format!("Invalid CFI directive: {line}"));
        let arg = |index: usize| {
            args.get(index)
                .copied()
                .filter(|s| !s.is_empty())
                .ok_or_else(invalid)
        };
        if directive == ".cfi_sections" {
            if args != [".eh_frame"] {
                return Err(invalid());
            }
            continue;
        }
        if directive == ".cfi_startproc" {
            if frame.is_some() || !matches!(args.as_slice(), [""] | ["simple"]) {
                return Err(invalid());
            }
            let start = format!("{prefix}_{serial}");
            serial += 1;
            writeln!(output, "{start}:").unwrap();
            frame = Some(Frame {
                start: start.clone(),
                last: start,
                instructions: String::new(),
                personality: None,
                lsda: None,
                cfa_offset: if arch == TargetArchitecture::X86_64 {
                    8
                } else {
                    0
                },
                cfa_register: if arch == TargetArchitecture::X86_64 {
                    7
                } else {
                    31
                },
                states: Vec::new(),
                simple: args == ["simple"],
                signal: false,
                return_column: if arch == TargetArchitecture::X86_64 {
                    16
                } else {
                    30
                },
            });
            continue;
        }
        let f = frame.as_mut().ok_or_else(invalid)?;
        if directive == ".cfi_endproc" {
            if !f.states.is_empty() {
                return Err(invalid());
            }
            let base = &f.start;
            writeln!(output, "{base}_end:").unwrap();
            let augmentation = format!(
                "z{}{}R{}",
                if f.personality.is_some() { "P" } else { "" },
                if f.lsda.is_some() { "L" } else { "" },
                if f.signal { "S" } else { "" }
            );
            let key = CieKey {
                simple: f.simple,
                signal: f.signal,
                return_column: f.return_column,
                personality: f.personality.clone(),
                lsda_encoding: f.lsda.as_ref().map(|(encoding, _)| *encoding),
            };
            let cie = if let Some(cie) = cies.get(&key) {
                cie.clone()
            } else {
                writeln!(frames, ".balign 4\n{base}_cie:\n.long {base}_cie_end-{base}_cie_body\n{base}_cie_body:\n.long 0\n.byte 1\n.asciz \"{augmentation}\"\n.uleb128 1\n.sleb128 -8\n.uleb128 {}\n.uleb128 {base}_aug_end-{base}_aug\n{base}_aug:", f.return_column).unwrap();
                if let Some((encoding, symbol)) = &f.personality {
                    writeln!(frames, ".byte {encoding}").unwrap();
                    frames.push_str(&pointer(*encoding, symbol)?);
                }
                if let Some((encoding, _)) = &f.lsda {
                    writeln!(frames, ".byte {encoding}").unwrap();
                }
                writeln!(frames, ".byte 0x1b\n{base}_aug_end:").unwrap();
                if !f.simple {
                    if arch == TargetArchitecture::X86_64 {
                        frames.push_str(".byte 12,7,8,0x90,1\n");
                    } else {
                        frames.push_str(".byte 12,31,0\n");
                    }
                }
                writeln!(frames, ".balign 4\n{base}_cie_end:").unwrap();
                let cie = format!("{base}_cie");
                cies.insert(key, cie.clone());
                cie
            };
            writeln!(frames, ".long {base}_fde_end-{base}_fde_body\n{base}_fde_body:\n.long .-{cie}\n.long {base}-.\n.long {base}_end-{base}\n.uleb128 {base}_fde_aug_end-{base}_fde_aug\n{base}_fde_aug:").unwrap();
            if let Some((encoding, symbol)) = &f.lsda {
                frames.push_str(&pointer(*encoding, symbol)?);
            }
            writeln!(
                frames,
                "{base}_fde_aug_end:\n{}.balign 4\n{base}_fde_end:",
                f.instructions
            )
            .unwrap();
            frame = None;
            continue;
        }
        match directive {
            ".cfi_personality" | ".cfi_lsda" => {
                let encoding = u8::try_from(number(arg(0)?)?).map_err(|_| invalid())?;
                pointer(encoding, arg(1)?)?;
                let value = Some((encoding, arg(1)?.to_string()));
                if directive == ".cfi_personality" {
                    f.personality = value;
                } else {
                    f.lsda = value;
                }
                continue;
            }
            ".cfi_signal_frame" => {
                f.signal = true;
                continue;
            }
            ".cfi_return_column" => {
                f.return_column = register(arg(0)?, arch)?;
                continue;
            }
            _ => {}
        }
        let label = format!("{prefix}_{serial}");
        serial += 1;
        writeln!(output, "{label}:").unwrap();
        writeln!(f.instructions, ".ras_cfi_advance {label}-{}", f.last).unwrap();
        f.last = label;
        let mut bytes = Vec::new();
        let mut unsigned = |opcode: u8, values: &[i128]| -> Result<(), RasError> {
            bytes.push(opcode);
            for value in values {
                if *value < 0 {
                    return Err(invalid());
                }
                bytes.extend(crate::parser::leb128(*value, false));
            }
            Ok(())
        };
        match directive {
            ".cfi_def_cfa" => {
                f.cfa_register = register(arg(0)?, arch)?;
                f.cfa_offset = number(arg(1)?)?;
                bytes = cfa(12, Some(f.cfa_register), f.cfa_offset)?;
            }
            ".cfi_def_cfa_register" => {
                f.cfa_register = register(arg(0)?, arch)?;
                unsigned(13, &[f.cfa_register])?;
            }
            ".cfi_def_cfa_offset" | ".cfi_adjust_cfa_offset" => {
                let value = number(arg(0)?)?;
                f.cfa_offset = if directive == ".cfi_adjust_cfa_offset" {
                    f.cfa_offset + value
                } else {
                    value
                };
                bytes = cfa(14, None, f.cfa_offset)?;
            }
            ".cfi_offset" | ".cfi_rel_offset" | ".cfi_val_offset" => {
                let reg = register(arg(0)?, arch)?;
                let offset = number(arg(1)?)?
                    - if directive == ".cfi_rel_offset" {
                        f.cfa_offset
                    } else {
                        0
                    };
                if offset % -8 != 0 {
                    return Err(invalid());
                }
                let factor = offset / -8;
                if reg < 0 {
                    return Err(invalid());
                }
                if directive == ".cfi_val_offset" {
                    bytes.push(if factor < 0 { 0x15 } else { 0x14 });
                    bytes.extend(crate::parser::leb128(reg, false));
                } else if factor >= 0 && reg < 64 {
                    bytes.push(0x80 | reg as u8);
                } else {
                    bytes.push(if factor < 0 { 0x11 } else { 5 });
                    bytes.extend(crate::parser::leb128(reg, false));
                }
                bytes.extend(crate::parser::leb128(factor, factor < 0));
            }
            ".cfi_restore" => {
                let reg = register(arg(0)?, arch)?;
                if (0..64).contains(&reg) {
                    bytes.push(0xc0 | reg as u8);
                } else {
                    unsigned(6, &[reg])?;
                }
            }
            ".cfi_undefined" => unsigned(7, &[register(arg(0)?, arch)?])?,
            ".cfi_same_value" => unsigned(8, &[register(arg(0)?, arch)?])?,
            ".cfi_register" => unsigned(9, &[register(arg(0)?, arch)?, register(arg(1)?, arch)?])?,
            ".cfi_remember_state" => {
                f.states.push((f.cfa_register, f.cfa_offset));
                bytes.push(10);
            }
            ".cfi_restore_state" => {
                (f.cfa_register, f.cfa_offset) = f.states.pop().ok_or_else(invalid)?;
                bytes.push(11);
            }
            ".cfi_escape" => {
                for value in args {
                    bytes.push(u8::try_from(number(value)?).map_err(|_| invalid())?);
                }
            }
            ".cfi_negate_ra_state" if arch == TargetArchitecture::Aarch64 => bytes.push(0x2d),
            _ => {
                return Err(RasError::ParseError(format!(
                    "Unsupported directive: {directive}"
                )));
            }
        }
        for byte in bytes {
            writeln!(f.instructions, ".byte {byte}").unwrap();
        }
    }
    if frame.is_some() {
        return Err(RasError::ParseError("Unterminated CFI procedure".into()));
    }
    if !frames.is_empty() {
        writeln!(output, ".section .eh_frame,\"a\",@progbits\n.balign 8\n{frames}").unwrap();
    }
    Ok((output, Some(prefix)))
}
