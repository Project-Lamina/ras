//! Core assembler implementation
//!
//! This module contains the main RasAssembler struct and basic operations
//! that are architecture-independent. Uses two-pass assembly for jmp/call
//! with label resolution.

use crate::{
    encoder::{
        aarch64::AArch64Encoder, arx64::Arx64Encoder, riscv::RiscVEncoder,
        traits::InstructionEncoder, x86_64::X86_64Encoder,
    },
    error::RasError,
    object::{
        ExternalReloc, ObjectSymbol, ObjectWriteOptions, ObjectWriteRequest, ObjectWriter,
        object_writer_for_os,
    },
    parser::{AssemblyParser, DataExpression, Line, parse_data_expression},
    target::unsupported_target_hint,
};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

const R_X86_64_REX_GOTPCRELX: u32 = 42;
use std::collections::{HashMap, HashSet};

/// Return type of the two-pass encoding pass.
type EncodeResult = Result<(Vec<u8>, Vec<ObjectSymbol>, Vec<ExternalReloc>), RasError>;

struct ShortJump {
    index: usize,
    offset: usize,
    target: String,
}

struct Layout {
    code: Vec<u8>,
    symbol_offsets: HashMap<String, usize>,
    patch_points: Vec<PatchPoint>,
    mapping_symbols: Vec<ObjectSymbol>,
    short_jumps: Vec<ShortJump>,
}

#[cfg(windows)]
mod windows_loader {
    use std::ffi::{c_char, c_void};

    unsafe extern "system" {
        pub fn GetModuleHandleA(module_name: *const c_char) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, proc_name: *const c_char) -> *mut c_void;
    }
}

/// Assembler: converts assembly text to object files
pub struct RasAssembler {
    pub(crate) target_arch: TargetArchitecture,
    pub(crate) target_os: TargetOperatingSystem,
    encoder: Box<dyn InstructionEncoder>,
    object_writer: Box<dyn ObjectWriter>,
    object_write_options: ObjectWriteOptions,
    pub(crate) function_pointers: HashMap<String, u64>, // Function name -> address
    #[cfg(feature = "encoder")]
    pub(crate) current_module: Option<*const lamina_mir::Module>, // Current module being compiled (for internal call detection)
}

enum PatchPoint {
    Data {
        offset: usize,
        width: usize,
        expression: String,
        arch: TargetArchitecture,
    },
    AArch64 {
        offset: usize,
        target: String,
        word: u32,
        bits: u32,
        kind: u32,
        relocate: bool,
    },
    X86Rel32 {
        offset: usize,
        target: String,
        relocate: bool,
    },
    X86RipRel32 {
        offset: usize,
        target: String,
        trailing: usize,
    },
    Arx64Jal {
        offset: usize,
        target: String,
        rd: u8,
    },
    Arx64Branch {
        offset: usize,
        target: String,
        rs1: u8,
        rs2: u8,
        funct3: u8,
    },
}

impl RasAssembler {
    /// Create a new assembler for the given target architecture and OS.
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Result<Self, RasError> {
        Self::with_object_write_options(target_arch, target_os, ObjectWriteOptions::default())
    }

    /// Create a new assembler with custom object-file write options.
    pub fn with_object_write_options(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
        object_write_options: ObjectWriteOptions,
    ) -> Result<Self, RasError> {
        // Create encoder based on target architecture
        let encoder: Box<dyn InstructionEncoder> = match target_arch {
            TargetArchitecture::X86_64 => Box::new(X86_64Encoder::new()),
            TargetArchitecture::Aarch64 => Box::new(AArch64Encoder::new()),
            TargetArchitecture::Arx64 => Box::new(Arx64Encoder::new()),
            TargetArchitecture::Riscv32 => Box::new(RiscVEncoder::new(false)),
            TargetArchitecture::Riscv64 => Box::new(RiscVEncoder::new(true)),
            _ => {
                return Err(RasError::UnsupportedTarget(unsupported_target_hint(
                    target_arch,
                    target_os,
                )));
            }
        };

        let object_writer = match object_writer_for_os(target_os) {
            Ok(w) => w,
            Err(_) => {
                return Err(RasError::UnsupportedTarget(format!(
                    "Unsupported OS for cross-compilation: {:?}. {}",
                    target_os,
                    unsupported_target_hint(target_arch, target_os)
                )));
            }
        };

        Ok(Self {
            target_arch,
            target_os,
            encoder,
            object_writer,
            object_write_options,
            function_pointers: HashMap::new(),
            #[cfg(feature = "encoder")]
            current_module: None,
        })
    }

    /// Replace the object-file write options used by subsequent assembly calls.
    pub fn set_object_write_options(&mut self, opts: ObjectWriteOptions) {
        self.object_write_options = opts;
    }

    /// Assemble `asm_text` and write a relocatable object file to `output_path`.
    ///
    /// Uses two-pass assembly for `jmp`/`call` instructions with forward label
    /// resolution.
    pub fn assemble_text_to_object(
        &mut self,
        asm_text: &str,
        output_path: &std::path::Path,
    ) -> Result<(), RasError> {
        let asm_text = crate::cfi::lower(asm_text, self.target_arch)?;
        let parsed = AssemblyParser::new()
            .parse(&asm_text)
            .map_err(|e| RasError::ParseError(e.to_string()))?;

        let mut sections = parsed.sections.clone();
        if !sections.iter().any(|s| s.name == ".text") {
            sections.insert(
                0,
                crate::parser::Section {
                    group: None,
                    name: ".text".into(),
                    flags: crate::parser::SectionFlags {
                        alloc: true,
                        exec: true,
                        write: false,
                    },
                    data: Vec::new(),
                    alignment: 1,
                },
            );
        }
        let section_lines: Vec<Vec<Line>> = sections
            .iter()
            .map(|section| {
                let mut current = ".text";
                parsed
                    .lines
                    .iter()
                    .filter(|line| {
                        if let Line::Section(name) = line {
                            current = name;
                            return false;
                        }
                        current == section.name
                    })
                    .cloned()
                    .collect()
            })
            .collect();
        let mut positions = HashMap::new();
        let mut layouts = Vec::new();
        for iteration in 0..64 {
            layouts.clear();
            let mut next = HashMap::new();
            for (section, lines) in sections.iter().zip(&section_lines) {
                let layout = self.layout_two_pass(lines, section.flags.exec, &positions)?;
                for (name, offset) in &layout.symbol_offsets {
                    next.insert(name.clone(), (section.name.clone(), *offset));
                }
                layouts.push(layout);
            }
            if next == positions {
                break;
            }
            if iteration == 63 {
                return Err(RasError::EncodingError(
                    "Section layout did not converge".into(),
                ));
            }
            positions = next;
        }
        let mut symbols = Vec::new();
        let mut relocs = Vec::new();
        for ((section, lines), layout) in sections.iter_mut().zip(&section_lines).zip(layouts) {
            for line in lines {
                if let Line::Leb128 { expression, signed } = line {
                    let value = absolute_expression(expression, &positions)?;
                    if !signed && value < 0 {
                        return Err(RasError::EncodingError("Negative unsigned LEB128".into()));
                    }
                }
            }
            let (data, mut syms, mut relocations) =
                Self::finish_layout(layout, lines, &positions, &section.name)?;
            section.alignment =
                if self.target_arch == TargetArchitecture::Aarch64 && section.flags.exec {
                    4
                } else {
                    1
                };
            for line in lines {
                if let Line::Align {
                    value, directive, ..
                } = line
                {
                    section.alignment = section.alignment.max(self.alignment(*value, directive)?);
                }
            }
            section.data = data;
            for reloc in &mut relocations {
                reloc.section = section.name.clone();
            }
            for sym in &mut syms {
                sym.section = section.name.clone();
            }
            symbols.append(&mut syms);
            relocs.append(&mut relocations);
        }
        let mut aliases = parsed.aliases.clone();
        while !aliases.is_empty() {
            let count = aliases.len();
            let mut pending = Vec::new();
            for (name, expression) in aliases {
                let resolved = match parse_data_expression(&expression)? {
                    DataExpression::Constant(value) => Some(("*ABS*".to_string(), value as u64)),
                    DataExpression::Symbol(target, addend) => symbols
                        .iter()
                        .find(|s| s.name == target)
                        .map(|s| (s.section.clone(), s.value.wrapping_add(addend as u64))),
                    _ => {
                        return Err(RasError::EncodingError(
                            "Unsupported .set expression".into(),
                        ));
                    }
                };
                if let Some((section, value)) = resolved {
                    symbols.push(ObjectSymbol {
                        name: name.clone(),
                        global: false,
                        section,
                        value,
                        attributes: Default::default(),
                    });
                } else {
                    pending.push((name, expression));
                }
            }
            if pending.len() == count {
                return Err(RasError::EncodingError("Unresolved .set expression".into()));
            }
            aliases = pending;
        }
        let mut ordered_attributes: Vec<_> = parsed.attributes.iter().collect();
        ordered_attributes.sort_by_key(|(name, _)| {
            parsed
                .common_order
                .iter()
                .position(|n| n == *name)
                .unwrap_or(usize::MAX)
        });
        for (name, attributes) in ordered_attributes {
            if let Some((size, alignment)) = attributes.common {
                let (section, value) = if attributes.local {
                    if !sections.iter().any(|s| s.name == ".bss") {
                        sections.push(crate::parser::Section {
                            group: None,
                            name: ".bss".into(),
                            flags: crate::parser::SectionFlags {
                                alloc: true,
                                exec: false,
                                write: true,
                            },
                            data: Vec::new(),
                            alignment: 1,
                        });
                    }
                    let section = sections.iter_mut().find(|s| s.name == ".bss").unwrap();
                    section.alignment = section.alignment.max(alignment as usize);
                    let value = (section.data.len() as u64).div_ceil(alignment) * alignment;
                    section.data.resize((value + size) as usize, 0);
                    (".bss", value)
                } else {
                    ("*COM*", alignment)
                };
                symbols.push(ObjectSymbol {
                    name: name.clone(),
                    global: !attributes.local,
                    section: section.into(),
                    value,
                    attributes: attributes.clone(),
                });
                continue;
            }
            if let Some(sym) = symbols.iter_mut().find(|s| &s.name == name) {
                sym.attributes = attributes.clone();
                sym.global = !attributes.local && (sym.global || attributes.weak);
            } else {
                symbols.push(ObjectSymbol {
                    name: name.clone(),
                    global: !attributes.local,
                    section: String::new(),
                    value: 0,
                    attributes: attributes.clone(),
                });
            }
        }
        for _ in 0..parsed.aliases.len() {
            for (name, expression) in &parsed.aliases {
                if let DataExpression::Symbol(target, _) = parse_data_expression(expression)? {
                    let inherited = symbols
                        .iter()
                        .find(|s| s.name == target)
                        .map(|s| s.attributes.clone());
                    if let Some(inherited) = inherited {
                        let symbol = symbols.iter_mut().find(|s| &s.name == name).unwrap();
                        if symbol.attributes.symbol_type == 0 {
                            symbol.attributes.symbol_type = inherited.symbol_type;
                        }
                        if symbol.attributes.size.is_none() {
                            symbol.attributes.size = inherited.size;
                        }
                    }
                }
            }
        }
        for symbol in &mut symbols {
            if let Some(expression) = &symbol.attributes.size {
                let size = absolute_expression(expression, &positions)?;
                u64::try_from(size)
                    .map_err(|_| RasError::EncodingError("Invalid symbol size".into()))?;
                symbol.attributes.size = Some(size.to_string());
            }
        }
        let code = &sections.iter().find(|s| s.name == ".text").unwrap().data;

        self.object_writer
            .write_object_file(
                output_path,
                &ObjectWriteRequest {
                    code,
                    sections: &sections,
                    symbols: &symbols,
                    relocations: &relocs,
                    target_arch: self.target_arch,
                    target_os: self.target_os,
                    opts: &self.object_write_options,
                },
            )
            .map_err(|e| RasError::ObjectError(e.to_string()))?;

        Ok(())
    }

    fn alignment(&self, value: usize, directive: &str) -> Result<usize, RasError> {
        let power = directive == ".p2align"
            || directive == ".align" && self.target_arch == TargetArchitecture::Aarch64;
        let alignment = if power {
            1usize.checked_shl(u32::try_from(value).unwrap_or(u32::MAX))
        } else {
            Some(value.max(1))
        };
        alignment
            .filter(|n| n.is_power_of_two())
            .ok_or_else(|| RasError::EncodingError("Invalid section alignment".into()))
    }

    fn layout_two_pass(
        &mut self,
        lines: &[Line],
        executable: bool,
        positions: &HashMap<String, (String, usize)>,
    ) -> Result<Layout, RasError> {
        let mut long_jumps = HashSet::new();
        let layout = loop {
            let layout = self.layout_lines(lines, executable, &long_jumps, positions)?;
            let grown: Vec<usize> = layout
                .short_jumps
                .iter()
                .filter(|site| {
                    !layout
                        .symbol_offsets
                        .get(&site.target)
                        .is_some_and(|&target| {
                            i8::try_from(target as i64 - (site.offset as i64 + 2)).is_ok()
                        })
                })
                .map(|site| site.index)
                .collect();
            if grown.is_empty() {
                break layout;
            }
            long_jumps.extend(grown);
        };
        Ok(layout)
    }

    fn layout_lines(
        &mut self,
        lines: &[Line],
        executable: bool,
        long_jumps: &HashSet<usize>,
        positions: &HashMap<String, (String, usize)>,
    ) -> Result<Layout, RasError> {
        let global_labels: HashSet<&str> = lines
            .iter()
            .filter_map(|line| match line {
                Line::Label(s) if s.global => Some(s.name.as_str()),
                _ => None,
            })
            .collect();
        let mut symbol_offsets: HashMap<String, usize> = HashMap::new();
        let mut patch_points: Vec<PatchPoint> = Vec::new();
        let mut code = Vec::new();
        let mut mapping_symbols = Vec::new();
        let mut mapping_mode = "";
        let mut current_offset = 0usize;
        let mut short_jumps = Vec::new();

        for (index, line) in lines.iter().enumerate() {
            if self.target_arch == TargetArchitecture::Aarch64 {
                let mode = match line {
                    Line::Instruction(_) => "$x",
                    Line::Data(bytes) if !bytes.is_empty() => "$d",
                    Line::DataExpression { .. } => "$d",
                    _ => mapping_mode,
                };
                if mode != mapping_mode {
                    mapping_symbols.push(ObjectSymbol {
                        attributes: Default::default(),
                        name: format!("{mode}.{}", mapping_symbols.len()),
                        global: false,
                        section: String::new(),
                        value: current_offset as u64,
                    });
                    mapping_mode = mode;
                }
            }
            match line {
                Line::Section(_) => {}
                Line::Align {
                    value,
                    directive,
                    fill,
                    max_skip,
                } => {
                    let alignment = self.alignment(*value, directive)?;
                    let padding = (alignment - current_offset % alignment) % alignment;
                    if max_skip.is_some_and(|maximum| padding > maximum) {
                        continue;
                    }
                    if let Some(fill) = fill {
                        code.resize(code.len() + padding, *fill);
                    } else if executable && self.target_arch == TargetArchitecture::Aarch64 {
                        let prefix = padding % 4;
                        code.resize(code.len() + prefix, 0);
                        for _ in 0..padding / 4 {
                            code.extend_from_slice(&0xd503_201fu32.to_le_bytes());
                        }
                    } else if executable && self.target_arch == TargetArchitecture::X86_64 {
                        x86_nop_padding(&mut code, padding);
                    } else {
                        code.resize(code.len() + padding, 0);
                    }
                    current_offset += padding;
                }
                Line::Label(sym) => {
                    symbol_offsets.insert(sym.name.clone(), current_offset);
                }
                Line::Leb128 { expression, signed } => {
                    let value = absolute_expression(expression, positions).unwrap_or(0);
                    if !signed && value < 0 {
                        return Err(RasError::EncodingError("Negative unsigned LEB128".into()));
                    }
                    let bytes = crate::parser::leb128(value, *signed);
                    current_offset += bytes.len();
                    code.extend(bytes);
                }
                Line::Data(bytes) => {
                    code.extend_from_slice(bytes);
                    current_offset += bytes.len();
                }
                Line::DataExpression { width, expression } => {
                    patch_points.push(PatchPoint::Data {
                        offset: current_offset,
                        width: *width,
                        expression: expression.clone(),
                        arch: self.target_arch,
                    });
                    code.resize(code.len() + width, 0);
                    current_offset += width;
                }
                Line::Instruction(inst) => {
                    let opcode = inst.opcode.to_lowercase();
                    let is_jmp_call = opcode == "jmp" || opcode == "jmpq" || opcode == "call";

                    if self.target_arch == TargetArchitecture::Arx64
                        && let Some(patch) =
                            arx64_label_patch(&opcode, &inst.operands, current_offset)?
                    {
                        code.extend_from_slice(&[0u8; 4]);
                        patch_points.push(patch);
                        current_offset += 4;
                        continue;
                    }

                    if self.target_arch == TargetArchitecture::Aarch64 {
                        let operands = &inst.operands;
                        let patch = if matches!(opcode.as_str(), "b" | "bl")
                            || opcode.starts_with("b.")
                        {
                            let target = operands
                                .first()
                                .ok_or_else(|| {
                                    RasError::EncodingError("Missing branch target".into())
                                })?
                                .clone();
                            let conditional = opcode.starts_with("b.");
                            let word = if conditional {
                                0x5400_0000 | crate::encoder::aarch64::condition(&opcode[2..])?
                            } else if opcode == "bl" {
                                0x9400_0000
                            } else {
                                0x1400_0000
                            };
                            Some(PatchPoint::AArch64 {
                                offset: current_offset,
                                target: target.clone(),
                                word,
                                bits: if conditional { 19 } else { 26 },
                                kind: if conditional {
                                    280
                                } else if opcode == "bl" {
                                    283
                                } else {
                                    282
                                },
                                relocate: opcode == "bl" && !target.starts_with(".L"),
                            })
                        } else if opcode == "adrp" && operands.len() == 2 {
                            Some(PatchPoint::AArch64 {
                                offset: current_offset,
                                target: operands[1].clone(),
                                word: 0x9000_0000
                                    | u32::from(crate::encoder::aarch64::register(&operands[0])?),
                                bits: 0,
                                kind: 275,
                                relocate: true,
                            })
                        } else if opcode == "add"
                            && operands.len() == 3
                            && operands[2].starts_with(":lo12:")
                        {
                            Some(PatchPoint::AArch64 {
                                offset: current_offset,
                                target: operands[2][6..].into(),
                                word: 0x9100_0000
                                    | (u32::from(crate::encoder::aarch64::register(&operands[1])?)
                                        << 5)
                                    | u32::from(crate::encoder::aarch64::register(&operands[0])?),
                                bits: 0,
                                kind: 277,
                                relocate: true,
                            })
                        } else {
                            None
                        };
                        if let Some(patch) = patch {
                            if let PatchPoint::AArch64 { word, .. } = &patch {
                                code.extend_from_slice(&word.to_le_bytes());
                            }
                            patch_points.push(patch);
                            current_offset += 4;
                            continue;
                        }
                    }

                    // x86_64 conditional jumps: 2-byte opcode (0x0F 8x) + rel32
                    if self.target_arch == TargetArchitecture::X86_64
                        && inst.operands.len() == 1
                        && let Some(cc_byte) = x86_cond_jmp_byte(&opcode)
                    {
                        let target = inst.operands[0].trim();
                        if !long_jumps.contains(&index) {
                            code.extend_from_slice(&[cc_byte - 0x10, 0]);
                            short_jumps.push(ShortJump {
                                index,
                                offset: current_offset,
                                target: target.to_string(),
                            });
                            current_offset += 2;
                            continue;
                        }
                        code.extend_from_slice(&[0x0F, cc_byte, 0, 0, 0, 0]);
                        patch_points.push(PatchPoint::X86Rel32 {
                            offset: current_offset + 2,
                            target: target.to_string(),
                            relocate: false,
                        });
                        current_offset += 6;
                        continue;
                    }

                    if is_jmp_call
                        && inst.operands.len() == 1
                        && !inst.operands[0].trim().starts_with('*')
                    {
                        let target = inst.operands[0].trim();
                        if self.target_arch == TargetArchitecture::X86_64 {
                            let is_call = opcode == "call";
                            if !is_call && !long_jumps.contains(&index) {
                                code.extend_from_slice(&[0xeb, 0]);
                                short_jumps.push(ShortJump {
                                    index,
                                    offset: current_offset,
                                    target: target.to_string(),
                                });
                                current_offset += 2;
                                continue;
                            }
                            let opcode_byte: u8 = if is_call { 0xe8 } else { 0xe9 };
                            code.push(opcode_byte);
                            code.extend_from_slice(&[0u8; 4]);
                            patch_points.push(PatchPoint::X86Rel32 {
                                offset: current_offset + 1,
                                target: target.to_string(),
                                relocate: is_call && global_labels.contains(target),
                            });
                            current_offset += 5;
                        } else {
                            let bytes = self
                                .encoder
                                .encode_instruction(inst)
                                .map_err(|e| RasError::EncodingError(e.to_string()))?;
                            code.extend_from_slice(&bytes);
                            current_offset += bytes.len();
                        }
                    } else {
                        let bytes = self
                            .encoder
                            .encode_instruction(inst)
                            .map_err(|e| RasError::EncodingError(e.to_string()))?;
                        if let Some(fixup) = self.encoder.take_rip_fixup()
                            && let Some(label) = inst
                                .operands
                                .iter()
                                .find_map(|op| extract_rip_label(op.trim().trim_start_matches('*')))
                        {
                            patch_points.push(PatchPoint::X86RipRel32 {
                                offset: current_offset + fixup.disp_offset,
                                target: label.to_string(),
                                trailing: fixup.trailing,
                            });
                        }
                        code.extend_from_slice(&bytes);
                        current_offset += bytes.len();
                    }
                }
            }
        }

        Ok(Layout {
            code,
            symbol_offsets,
            patch_points,
            mapping_symbols,
            short_jumps,
        })
    }

    fn finish_layout(
        layout: Layout,
        lines: &[Line],
        positions: &HashMap<String, (String, usize)>,
        section: &str,
    ) -> EncodeResult {
        let Layout {
            mut code,
            symbol_offsets,
            patch_points,
            mapping_symbols,
            short_jumps,
        } = layout;
        for site in &short_jumps {
            let target = symbol_offsets.get(&site.target).ok_or_else(|| {
                RasError::EncodingError(format!("Undefined label: {}", site.target))
            })?;
            let rel = i8::try_from(*target as i64 - (site.offset as i64 + 2)).map_err(|_| {
                RasError::EncodingError(format!("Short jump out of range: {}", site.target))
            })?;
            code[site.offset + 1] = rel as u8;
        }

        let mut external_relocs: Vec<ExternalReloc> = Vec::new();

        for patch in &patch_points {
            if let PatchPoint::Data {
                offset,
                width,
                expression,
                arch,
            } = patch
            {
                match parse_data_expression(expression)? {
                    DataExpression::Constant(value) => {
                        code[*offset..*offset + width]
                            .copy_from_slice(&value.to_le_bytes()[..*width]);
                    }
                    DataExpression::Difference(left, right) => {
                        let lookup = |name: &str| {
                            if name == "." {
                                Some((section.to_string(), *offset))
                            } else {
                                positions.get(name).cloned()
                            }
                        };
                        let left_value = lookup(&left);
                        let right_value = lookup(&right);
                        if let (Some((ls, left)), Some((rs, right))) = (&left_value, &right_value)
                            && ls == rs
                        {
                            let value = *left as i128 - *right as i128;
                            code[*offset..*offset + width]
                                .copy_from_slice(&value.to_le_bytes()[..*width]);
                        } else if let Some((rs, right)) =
                            right_value.filter(|(rs, _)| rs == section)
                        {
                            let _ = rs;
                            let kind = match (arch, width) {
                                (TargetArchitecture::X86_64, 4) => 2,
                                (TargetArchitecture::X86_64, 8) => 24,
                                (TargetArchitecture::Aarch64, 4) => 261,
                                (TargetArchitecture::Aarch64, 8) => 260,
                                _ => {
                                    return Err(RasError::EncodingError(
                                        "Unsupported relative data relocation".into(),
                                    ));
                                }
                            };
                            external_relocs.push(ExternalReloc {
                                offset: *offset,
                                symbol: left,
                                section: String::new(),
                                kind,
                                addend: *offset as i64 - right as i64,
                            });
                        } else {
                            return Err(RasError::EncodingError(format!(
                                "Undefined or cross-section label in difference: {right}"
                            )));
                        }
                    }
                    DataExpression::Symbol(symbol, addend) => {
                        let kind = match (arch, width) {
                            (TargetArchitecture::X86_64, 1) => 14,
                            (TargetArchitecture::X86_64, 2) => 12,
                            (TargetArchitecture::X86_64, 4) => 10,
                            (TargetArchitecture::X86_64, 8) => 1,
                            (TargetArchitecture::Aarch64, 2) => 259,
                            (TargetArchitecture::Aarch64, 4) => 258,
                            (TargetArchitecture::Aarch64, 8) => 257,
                            _ => {
                                return Err(RasError::EncodingError(
                                    "Unsupported data relocation width or architecture".into(),
                                ));
                            }
                        };
                        external_relocs.push(ExternalReloc {
                            offset: *offset,
                            symbol,
                            section: String::new(),
                            kind,
                            addend: i64::try_from(addend).map_err(|_| {
                                RasError::EncodingError(
                                    "Data relocation addend out of range".into(),
                                )
                            })?,
                        });
                    }
                }
                continue;
            }
            let target = match patch {
                PatchPoint::Data { .. } => unreachable!(),
                PatchPoint::AArch64 { target, .. }
                | PatchPoint::X86Rel32 { target, .. }
                | PatchPoint::X86RipRel32 { target, .. }
                | PatchPoint::Arx64Jal { target, .. }
                | PatchPoint::Arx64Branch { target, .. } => target,
            };
            if let PatchPoint::AArch64 {
                offset,
                target,
                word,
                bits,
                kind,
                relocate,
            } = patch
            {
                let target = reloc_symbol(target);
                if !relocate && let Some(&target_offset) = symbol_offsets.get(target) {
                    let delta = target_offset as i64 - *offset as i64;
                    if delta % 4 != 0
                        || !(-(1i64 << (bits + 1))..(1i64 << (bits + 1))).contains(&delta)
                    {
                        return Err(RasError::EncodingError(
                            "AArch64 branch out of range".into(),
                        ));
                    }
                    let imm = (delta / 4) as u32 & ((1 << bits) - 1);
                    let instruction = word | if *bits == 19 { imm << 5 } else { imm };
                    code[*offset..*offset + 4].copy_from_slice(&instruction.to_le_bytes());
                } else {
                    external_relocs.push(ExternalReloc {
                        offset: *offset,
                        symbol: target.to_string(),
                        section: String::new(),
                        kind: *kind,
                        addend: 0,
                    });
                }
                continue;
            }
            let got = target.ends_with("@GOTPCREL");
            let target = reloc_symbol(target);
            if let Some(&target_offset) = symbol_offsets
                .get(target)
                .filter(|_| !got && !matches!(patch, PatchPoint::X86Rel32 { relocate: true, .. }))
            {
                match patch {
                    PatchPoint::Data { .. } | PatchPoint::AArch64 { .. } => unreachable!(),
                    PatchPoint::X86Rel32 { offset, .. } => {
                        let rel32 = (target_offset as i64) - (*offset as i64 + 4);
                        let rel32_bytes = (rel32 as i32).to_le_bytes();
                        code[*offset..*offset + 4].copy_from_slice(&rel32_bytes);
                    }
                    PatchPoint::X86RipRel32 {
                        offset, trailing, ..
                    } => {
                        let rel32 =
                            (target_offset as i64) - (*offset as i64 + 4 + *trailing as i64);
                        let rel32_bytes = (rel32 as i32).to_le_bytes();
                        code[*offset..*offset + 4].copy_from_slice(&rel32_bytes);
                    }
                    PatchPoint::Arx64Jal { offset, rd, .. } => {
                        let rel = (target_offset as i64) - (*offset as i64);
                        let word = arx64_j_type(rel as i32, *rd);
                        code[*offset..*offset + 4].copy_from_slice(&word.to_le_bytes());
                    }
                    PatchPoint::Arx64Branch {
                        offset,
                        rs1,
                        rs2,
                        funct3,
                        ..
                    } => {
                        let rel = (target_offset as i64) - (*offset as i64);
                        let word = arx64_b_type(rel as i32, *rs2, *rs1, *funct3);
                        code[*offset..*offset + 4].copy_from_slice(&word.to_le_bytes());
                    }
                }
            } else {
                // Unresolved symbol → external relocation for the linker.
                match patch {
                    PatchPoint::X86Rel32 { offset, .. } => {
                        external_relocs.push(ExternalReloc {
                            offset: *offset,
                            symbol: target.to_string(),
                            section: String::new(),
                            kind: 4,
                            addend: -4,
                        });
                    }
                    PatchPoint::X86RipRel32 {
                        offset, trailing, ..
                    } => {
                        external_relocs.push(ExternalReloc {
                            offset: *offset,
                            symbol: target.to_string(),
                            section: String::new(),
                            kind: if got { R_X86_64_REX_GOTPCRELX } else { 2 },
                            addend: -(4 + *trailing as i64),
                        });
                    }
                    _ => {
                        return Err(RasError::EncodingError(format!(
                            "Undefined label: {}",
                            target
                        )));
                    }
                }
            }
        }

        let mut symbols: Vec<_> = lines
            .iter()
            .filter_map(|l| match l {
                Line::Label(s) => Some(ObjectSymbol {
                    attributes: Default::default(),
                    name: s.name.clone(),
                    global: s.global,
                    section: s.section.clone(),
                    value: symbol_offsets.get(&s.name).copied().unwrap_or(0) as u64,
                }),
                _ => None,
            })
            .collect();

        symbols.extend(mapping_symbols);
        Ok((code, symbols, external_relocs))
    }

    /// Register a function pointer for runtime calls.
    ///
    /// Resolves the named symbol using `dlsym` (Unix) or `GetProcAddress` (Windows)
    /// and stores its address for use in generated code.
    pub fn register_function(&mut self, name: &str) -> Result<(), RasError> {
        #[cfg(unix)]
        {
            use std::ffi::CString;

            let symbol = CString::new(name)
                .map_err(|e| RasError::EncodingError(format!("Invalid function name: {}", e)))?;

            // Try to resolve using RTLD_DEFAULT first (searches already loaded libraries)
            // This is safer and doesn't require opening/closing handles
            let ptr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr()) };

            if ptr.is_null() {
                // Fallback: try opening libc explicitly
                // Clear any previous error
                unsafe {
                    libc::dlerror();
                }

                let handle = unsafe { libc::dlopen(std::ptr::null(), libc::RTLD_LAZY) };
                if handle.is_null() {
                    let err_msg = unsafe {
                        let err_ptr = libc::dlerror();
                        if err_ptr.is_null() {
                            "unknown error (dlerror returned null)"
                        } else {
                            std::ffi::CStr::from_ptr(err_ptr)
                                .to_str()
                                .unwrap_or("unknown error")
                        }
                    };
                    return Err(RasError::EncodingError(format!(
                        "Failed to open libc: {}",
                        err_msg
                    )));
                }

                // Clear error before dlsym
                unsafe {
                    libc::dlerror();
                }

                let ptr2 = unsafe { libc::dlsym(handle, symbol.as_ptr()) };
                if ptr2.is_null() {
                    let err_msg = unsafe {
                        let err_ptr = libc::dlerror();
                        if err_ptr.is_null() {
                            "symbol not found"
                        } else {
                            std::ffi::CStr::from_ptr(err_ptr)
                                .to_str()
                                .unwrap_or("unknown error")
                        }
                    };
                    unsafe { libc::dlclose(handle) };
                    return Err(RasError::EncodingError(format!(
                        "Failed to resolve symbol {}: {}",
                        name, err_msg
                    )));
                }

                self.function_pointers.insert(name.to_string(), ptr2 as u64);
                unsafe { libc::dlclose(handle) };
            } else {
                self.function_pointers.insert(name.to_string(), ptr as u64);
            }

            Ok(())
        }

        #[cfg(windows)]
        {
            use std::ffi::CString;
            use windows_loader::{GetModuleHandleA, GetProcAddress};

            let module = unsafe { GetModuleHandleA(c"msvcrt.dll".as_ptr() as *const i8) };
            if module.is_null() {
                return Err(RasError::EncodingError(
                    "Failed to get msvcrt.dll handle".to_string(),
                ));
            }

            let symbol = CString::new(name)
                .map_err(|e| RasError::EncodingError(format!("Invalid function name: {}", e)))?;

            let ptr = unsafe { GetProcAddress(module, symbol.as_ptr()) };
            if ptr.is_null() {
                return Err(RasError::EncodingError(format!(
                    "Failed to resolve symbol {}",
                    name
                )));
            }

            self.function_pointers.insert(name.to_string(), ptr as u64);
            Ok(())
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::EncodingError(
                "Runtime function resolution not supported on this platform".to_string(),
            ))
        }
    }

    /// Compile all functions in a MIR module to machine code and return the raw bytes.
    ///
    /// Equivalent to calling [`compile_mir_to_binary_function`] with `function_name = None`.
    /// Requires the `encoder` feature.
    ///
    /// [`compile_mir_to_binary_function`]: Self::compile_mir_to_binary_function
    #[cfg(feature = "encoder")]
    pub fn compile_mir_to_binary(
        &mut self,
        module: &lamina_mir::Module,
    ) -> Result<Vec<u8>, RasError> {
        let (code, _) = self.compile_mir_to_binary_function(module, None)?;
        Ok(code)
    }

    /// Compile a specific function from a MIR module to binary.
    ///
    /// If `function_name` is `None`, all functions in the module are compiled.
    /// Returns `(binary_code, function_offsets)` where `function_offsets` maps
    /// each function name to its byte offset within `binary_code`.
    #[cfg(feature = "encoder")]
    pub fn compile_mir_to_binary_function(
        &mut self,
        module: &lamina_mir::Module,
        function_name: Option<&str>,
    ) -> Result<(Vec<u8>, std::collections::HashMap<String, usize>), RasError> {
        // Store module reference for checking internal vs external calls
        self.current_module = Some(module);
        // Reuse register allocation and ABI from mir_codegen
        match self.target_arch {
            TargetArchitecture::X86_64 => {
                crate::assembler::x86_64::compile_mir_x86_64_function(self, module, function_name)
            }
            TargetArchitecture::Aarch64 => {
                crate::assembler::aarch64::compile_mir_aarch64_function(self, module, function_name)
            }
            TargetArchitecture::Riscv64 => crate::assembler::riscv::compile_mir_riscv_function(
                self,
                module,
                function_name,
                true,
            ),
            TargetArchitecture::Riscv32 => crate::assembler::riscv::compile_mir_riscv_function(
                self,
                module,
                function_name,
                false,
            ),
            _ => Err(RasError::UnsupportedTarget(format!(
                "MIR compilation not supported for architecture: {:?}",
                self.target_arch
            ))),
        }
    }
}

/// `call puts@PLT` names the symbol `puts`; the suffix selects the relocation,
/// it is not part of the name.
fn reloc_symbol(target: &str) -> &str {
    target.split_once('@').map_or(target, |(name, _)| name)
}

fn arx64_label_patch(
    opcode: &str,
    operands: &[String],
    offset: usize,
) -> Result<Option<PatchPoint>, RasError> {
    match opcode {
        "j" if operands.len() == 1 && !is_numeric(&operands[0]) => Ok(Some(PatchPoint::Arx64Jal {
            offset,
            target: operands[0].trim().to_string(),
            rd: 0,
        })),
        "call" if operands.len() == 1 && !is_numeric(&operands[0]) => {
            Ok(Some(PatchPoint::Arx64Jal {
                offset,
                target: operands[0].trim().to_string(),
                rd: 1,
            }))
        }
        "jal" if operands.len() == 2 && !is_numeric(&operands[1]) => {
            Ok(Some(PatchPoint::Arx64Jal {
                offset,
                target: operands[1].trim().to_string(),
                rd: parse_arx64_reg(&operands[0])?,
            }))
        }
        "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu"
            if operands.len() == 3 && !is_numeric(&operands[2]) =>
        {
            Ok(Some(PatchPoint::Arx64Branch {
                offset,
                target: operands[2].trim().to_string(),
                rs1: parse_arx64_reg(&operands[0])?,
                rs2: parse_arx64_reg(&operands[1])?,
                funct3: match opcode {
                    "beq" => 0x0,
                    "bne" => 0x1,
                    "blt" => 0x4,
                    "bge" => 0x5,
                    "bltu" => 0x6,
                    "bgeu" => 0x7,
                    _ => unreachable!(),
                },
            }))
        }
        _ => Ok(None),
    }
}

fn is_numeric(value: &str) -> bool {
    let value = value.trim();
    value.parse::<i64>().is_ok()
        || value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"))
            .is_some_and(|hex| i64::from_str_radix(hex, 16).is_ok())
}

fn parse_arx64_reg(value: &str) -> Result<u8, RasError> {
    let value = value.trim().trim_start_matches('%');
    let raw = match value {
        "zero" => 0,
        "ra" | "lr" => 1,
        "sp" => 2,
        _ => value
            .strip_prefix('r')
            .or_else(|| value.strip_prefix('x'))
            .ok_or_else(|| RasError::EncodingError(format!("Unknown ARX64 register: {}", value)))?
            .parse::<u8>()
            .map_err(|_| RasError::EncodingError(format!("Unknown ARX64 register: {}", value)))?,
    };
    if raw < 32 {
        Ok(raw)
    } else {
        Err(RasError::EncodingError(format!(
            "ARX64 register out of range: {}",
            value
        )))
    }
}

fn arx64_j_type(offset: i32, rd: u8) -> u32 {
    let o = offset as u32;
    (((o >> 20) & 0x1) << 31)
        | (((o >> 1) & 0x03ff) << 21)
        | (((o >> 11) & 0x1) << 20)
        | (((o >> 12) & 0xff) << 12)
        | ((rd as u32) << 7)
        | 0x6f
}

fn arx64_b_type(offset: i32, rs2: u8, rs1: u8, funct3: u8) -> u32 {
    let o = offset as u32;
    (((o >> 12) & 0x1) << 31)
        | (((o >> 5) & 0x3f) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | ((funct3 as u32) << 12)
        | (((o >> 1) & 0x0f) << 8)
        | (((o >> 11) & 0x1) << 7)
        | 0x63
}

/// Maps x86 conditional-jump mnemonic to the second opcode byte of the `0F 8x rel32` encoding.
/// Returns `None` for non-conditional-jump mnemonics.
fn x86_nop_padding(code: &mut Vec<u8>, mut padding: usize) {
    const NOPS: [&[u8]; 12] = [
        &[],
        &[0x90],
        &[0x66, 0x90],
        &[0x0f, 0x1f, 0],
        &[0x0f, 0x1f, 0x40, 0],
        &[0x0f, 0x1f, 0x44, 0, 0],
        &[0x66, 0x0f, 0x1f, 0x44, 0, 0],
        &[0x0f, 0x1f, 0x80, 0, 0, 0, 0],
        &[0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0],
        &[0x66, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0],
        &[0x66, 0x2e, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0],
        &[0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0],
    ];
    while padding != 0 {
        let size = padding.min(11);
        code.extend_from_slice(NOPS[size]);
        padding -= size;
    }
}

fn x86_cond_jmp_byte(opcode: &str) -> Option<u8> {
    match opcode {
        "jo" => Some(0x80),
        "jno" => Some(0x81),
        "jb" | "jnae" | "jc" => Some(0x82),
        "jnb" | "jae" | "jnc" => Some(0x83),
        "je" | "jz" => Some(0x84),
        "jne" | "jnz" => Some(0x85),
        "jbe" | "jna" => Some(0x86),
        "ja" | "jnbe" => Some(0x87),
        "js" => Some(0x88),
        "jns" => Some(0x89),
        "jp" | "jpe" => Some(0x8A),
        "jnp" | "jpo" => Some(0x8B),
        "jl" | "jnge" => Some(0x8C),
        "jge" | "jnl" => Some(0x8D),
        "jle" | "jng" => Some(0x8E),
        "jg" | "jnle" => Some(0x8F),
        _ => None,
    }
}

/// Returns the label name if `op` has the form `label(%rip)` (RIP-relative addressing).
fn extract_rip_label(op: &str) -> Option<&str> {
    let paren = op.find('(')?;
    let close = op.find(')')?;
    if close <= paren {
        return None;
    }
    let base = op[paren + 1..close].trim().trim_start_matches('%');
    if !base.eq_ignore_ascii_case("rip") {
        return None;
    }
    let label = op[..paren].trim();
    // Must be a non-empty label name (not a numeric displacement)
    if label.is_empty() || label.starts_with(|c: char| c.is_ascii_digit() || c == '-') {
        return None;
    }
    Some(label)
}

fn absolute_expression(
    expression: &str,
    positions: &HashMap<String, (String, usize)>,
) -> Result<i128, RasError> {
    match parse_data_expression(expression)? {
        DataExpression::Constant(value) => Ok(value),
        DataExpression::Difference(left, right) => {
            if let (Some((ls, l)), Some((rs, r))) = (positions.get(&left), positions.get(&right))
                && ls == rs
            {
                return Ok(*l as i128 - *r as i128);
            }
            Err(RasError::EncodingError(format!(
                "Unresolved absolute expression: {expression}"
            )))
        }
        _ => Err(RasError::EncodingError(format!(
            "Unsupported absolute expression: {expression}"
        ))),
    }
}
