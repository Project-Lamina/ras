//! Core assembler implementation
//!
//! This module contains the main RasAssembler struct and basic operations
//! that are architecture-independent. Uses two-pass assembly for jmp/call
//! with label resolution.

use crate::{
    encoder::{
        aarch64::AArch64Encoder, arx64::Arx64Encoder, riscv::RiscVEncoder,
        traits::{InstructionEncoder, ParsedInstruction, RipFixup},
        x86_64::X86_64Encoder,
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

struct LayoutCache {
    sizes: Vec<usize>,
    short_jumps: Vec<ShortJump>,
    code: Vec<u8>,
    line_offsets: Vec<usize>,
    rip_fixups: Vec<(usize, RipFixup)>,
}

impl LayoutCache {
    fn rip_fixup(&self, index: usize) -> Option<RipFixup> {
        self.rip_fixups
            .binary_search_by_key(&index, |&(line, _)| line)
            .ok()
            .map(|position| self.rip_fixups[position].1)
    }
}

struct RelaxationEvent {
    index: usize,
    gap: usize,
    size: usize,
    kind: RelaxationKind,
}

enum RelaxationKind {
    Label,
    Align {
        alignment: usize,
        max_skip: Option<usize>,
    },
    Branch {
        target: Option<usize>,
        same_region: bool,
        near_size: usize,
        growth: usize,
    },
}

impl RelaxationKind {
    fn padding(&self, offset: usize) -> usize {
        let Self::Align {
            alignment,
            max_skip,
        } = self
        else {
            return 0;
        };
        let padding = (alignment - offset % alignment) % alignment;
        if max_skip.is_some_and(|maximum| padding > maximum) {
            0
        } else {
            padding
        }
    }
}

struct Layout {
    line_offsets: Vec<usize>,
    code: Vec<u8>,
    symbol_offsets: HashMap<String, usize>,
    patch_points: Vec<PatchPoint>,
    mapping_symbols: Vec<ObjectSymbol>,
    short_jumps: Vec<ShortJump>,
    rip_fixups: Vec<(usize, RipFixup)>,
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
    pub(crate) function_pointers: HashMap<String, u64>,
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
        addend: i64,
        relocate: bool,
    },
    Riscv {
        offset: usize,
        target: String,
        word: u32,
        jump: bool,
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
        let (asm_text, internal_prefix) = crate::cfi::lower(asm_text, self.target_arch)?;
        let parsed = AssemblyParser::new()
            .with_darwin_directives(self.target_os == TargetOperatingSystem::MacOS)
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
        let section_indices: HashMap<_, _> = sections
            .iter()
            .enumerate()
            .map(|(index, section)| (section.name.as_str(), index))
            .collect();
        let mut section_lines: Vec<Vec<Line>> = sections.iter().map(|_| Vec::new()).collect();
        let mut current = section_indices.get(".text").copied();
        for line in parsed.lines {
            if let Line::Section(name) = &line {
                current = section_indices.get(name.as_str()).copied();
            } else if let Some(index) = current {
                section_lines[index].push(line);
            }
        }
        let dependencies: Vec<Vec<_>> = section_lines
            .iter()
            .map(|lines| {
                lines
                    .iter()
                    .filter_map(|line| match line {
                        Line::CfiAdvance(expression) | Line::Leb128 { expression, .. } => {
                            Some(expression.as_str())
                        }
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        let mut positions = HashMap::new();
        let mut layouts: Vec<Option<Layout>> = sections.iter().map(|_| None).collect();
        let mut layout_inputs: Vec<Vec<i128>> = sections.iter().map(|_| Vec::new()).collect();
        let mut layout_caches: Vec<_> = sections.iter().map(|_| None).collect();
        for iteration in 0..64 {
            let mut next = HashMap::new();
            for (index, section) in sections.iter().enumerate() {
                let inputs: Vec<_> = dependencies[index]
                    .iter()
                    .map(|expression| absolute_expression(expression, &positions).unwrap_or(0))
                    .collect();
                if layouts[index].is_none() || inputs != layout_inputs[index] {
                    layouts[index] = Some(self.layout_two_pass(
                        &section_lines[index],
                        section.flags.exec,
                        &positions,
                        &mut layout_caches[index],
                        true,
                    )?);
                    layout_inputs[index] = inputs;
                }
                for (name, offset) in &layouts[index].as_ref().unwrap().symbol_offsets {
                    next.insert(name.clone(), (section.name.clone(), *offset));
                }
            }
            let converged = dependencies
                .iter()
                .zip(&layout_inputs)
                .all(|(expressions, inputs)| {
                    expressions.iter().zip(inputs).all(|(expression, &input)| {
                        absolute_expression(expression, &next).unwrap_or(0) == input
                    })
                });
            positions = next;
            if converged {
                log::debug!("section layout: iterations={}", iteration + 1);
                break;
            }
            if iteration == 63 {
                return Err(RasError::EncodingError(
                    "Section layout did not converge".into(),
                ));
            }
        }
        let global_symbols: HashSet<_> = parsed
            .symbols
            .iter()
            .filter(|s| s.global)
            .map(|s| s.name.as_str())
            .chain(
                parsed
                    .attributes
                    .iter()
                    .filter(|(_, a)| !a.local && (a.global || a.weak))
                    .map(|(name, _)| name.as_str()),
            )
            .collect();
        let mut symbols = Vec::new();
        let mut relocs = Vec::new();
        for ((section, lines), layout) in sections.iter_mut().zip(&section_lines).zip(layouts) {
            for line in lines {
                if let Line::CfiAdvance(expression) = line {
                    crate::cfi::advance(absolute_expression(expression, &positions)?)?;
                }
                if let Line::Leb128 { expression, signed } = line {
                    let value = absolute_expression(expression, &positions)?;
                    if !signed && value < 0 {
                        return Err(RasError::EncodingError("Negative unsigned LEB128".into()));
                    }
                }
            }
            let (data, mut syms, mut relocations) = Self::finish_layout(
                layout.unwrap(),
                lines,
                &positions,
                &section.name,
                &global_symbols,
                self.target_os == TargetOperatingSystem::MacOS,
            )?;
            section.alignment = if (self.target_arch == TargetArchitecture::Aarch64
                && self.target_os != TargetOperatingSystem::MacOS
                || matches!(
                    self.target_arch,
                    TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64
                ))
                && section.flags.exec
            {
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
        let mut symbol_index: HashMap<String, usize> = HashMap::with_capacity(symbols.len());
        for (i, symbol) in symbols.iter().enumerate() {
            symbol_index.entry(symbol.name.clone()).or_insert(i);
        }
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
                symbol_index.entry(name.clone()).or_insert(symbols.len());
                symbols.push(ObjectSymbol {
                    name: name.clone(),
                    global: !attributes.local,
                    section: section.into(),
                    value,
                    attributes: attributes.clone(),
                });
                continue;
            }
            if let Some(sym) = symbol_index.get(name).map(|&i| &mut symbols[i]) {
                sym.attributes = attributes.clone();
                sym.global =
                    !attributes.local && (sym.global || attributes.global || attributes.weak);
            } else {
                symbol_index.entry(name.clone()).or_insert(symbols.len());
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
            if internal_prefix
                .as_ref()
                .is_some_and(|prefix| symbol.name.starts_with(prefix))
            {
                symbol.attributes.internal = true;
            }
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
            || directive == ".align"
                && matches!(
                    self.target_arch,
                    TargetArchitecture::Aarch64
                        | TargetArchitecture::Riscv32
                        | TargetArchitecture::Riscv64
                );
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
        cache: &mut Option<LayoutCache>,
        reuse_encodings: bool,
    ) -> Result<Layout, RasError> {
        let mut labels = HashMap::new();
        let mut regions = Vec::with_capacity(lines.len());
        let mut region = 0;
        for (index, line) in lines.iter().enumerate() {
            regions.push(region);
            if let Line::Label(symbol) = line {
                labels.insert(symbol.name.as_str(), index);
            }
            if matches!(line, Line::Align { .. }) {
                region += 1;
            }
        }
        let mut long_jumps = HashSet::new();
        for (index, line) in lines.iter().enumerate() {
            if let Line::Instruction(inst) = line
                && (inst.opcode == "jmp" || x86_cond_jmp_byte(&inst.opcode).is_some())
                && inst.operands.len() == 1
                && !inst.operands[0].starts_with('*')
                && !labels.contains_key(inst.operands[0].as_str())
            {
                long_jumps.insert(index);
            }
        }
        if cache.is_none() {
            let layout = self.layout_lines(lines, executable, &long_jumps, positions, None)?;
            let sizes = layout
                .line_offsets
                .iter()
                .zip(
                    layout
                        .line_offsets
                        .iter()
                        .skip(1)
                        .chain(std::iter::once(&layout.code.len())),
                )
                .map(|(start, end)| end - start)
                .collect();
            *cache = Some(LayoutCache {
                sizes,
                short_jumps: layout.short_jumps,
                code: layout.code,
                line_offsets: layout.line_offsets,
                rip_fixups: layout.rip_fixups,
            });
        }
        let cache = cache.as_mut().unwrap();
        for site in &cache.short_jumps {
            cache.sizes[site.index] = 2;
        }
        let targets: HashSet<_> = cache
            .short_jumps
            .iter()
            .map(|site| site.target.as_str())
            .collect();
        let mut events = Vec::new();
        let mut label_events = HashMap::with_capacity(targets.len());
        let mut sites = cache.short_jumps.iter().peekable();
        let mut gap = 0;
        for (index, line) in lines.iter().enumerate() {
            let kind = match line {
                Line::Label(symbol) if targets.contains(symbol.name.as_str()) => {
                    label_events.insert(symbol.name.as_str(), events.len());
                    Some(RelaxationKind::Label)
                }
                Line::Align {
                    value,
                    directive,
                    max_skip,
                    ..
                } => Some(RelaxationKind::Align {
                    alignment: self.alignment(*value, directive)?,
                    max_skip: *max_skip,
                }),
                Line::CfiAdvance(expression) => {
                    let value = absolute_expression(expression, positions).unwrap_or(0);
                    cache.sizes[index] = crate::cfi::advance(value)?.len();
                    None
                }
                Line::Leb128 { expression, signed } => {
                    let value = absolute_expression(expression, positions).unwrap_or(0);
                    if !signed && value < 0 {
                        return Err(RasError::EncodingError("Negative unsigned LEB128".into()));
                    }
                    cache.sizes[index] = crate::parser::leb128(value, *signed).len();
                    None
                }
                Line::Instruction(inst) if sites.peek().is_some_and(|site| site.index == index) => {
                    let site = sites.next().unwrap();
                    Some(RelaxationKind::Branch {
                        target: None,
                        same_region: labels
                            .get(site.target.as_str())
                            .is_some_and(|&label| regions[label] == regions[index]),
                        near_size: if x86_cond_jmp_byte(&inst.opcode).is_some() {
                            6
                        } else {
                            5
                        },
                        growth: if inst.opcode == "jmp" { 3 } else { 4 },
                    })
                }
                _ => None,
            };
            if let Some(kind) = kind {
                events.push(RelaxationEvent {
                    index,
                    gap,
                    size: cache.sizes[index],
                    kind,
                });
                gap = 0;
            } else {
                gap += cache.sizes[index];
            }
        }
        let mut sites = cache.short_jumps.iter();
        for event in &mut events {
            if let RelaxationKind::Branch { target, .. } = &mut event.kind {
                *target = label_events
                    .get(sites.next().unwrap().target.as_str())
                    .copied();
            }
        }
        let mut offsets = vec![0; events.len()];
        let mut adjusted = vec![0; events.len()];
        let mut rounds = 0;
        loop {
            rounds += 1;
            let mut offset = 0;
            for (event, address) in events.iter_mut().zip(&mut offsets) {
                offset += event.gap;
                *address = offset;
                if matches!(event.kind, RelaxationKind::Align { .. }) {
                    event.size = event.kind.padding(offset);
                }
                offset += event.size;
            }
            let mut stretch = 0i64;
            let mut grown = false;
            for (index, event) in events.iter_mut().enumerate() {
                let old = offsets[index] as i64;
                let address = old + stretch;
                adjusted[index] = address;
                match event.kind {
                    RelaxationKind::Align { .. } => {
                        stretch += event.kind.padding(address as usize) as i64 - event.size as i64;
                    }
                    RelaxationKind::Branch {
                        target,
                        same_region,
                        near_size,
                        growth,
                    } if event.size == 2 => {
                        let target = match target {
                            Some(target) if target < index => adjusted[target],
                            Some(target) if stretch < 0 || same_region => {
                                offsets[target] as i64 + stretch
                            }
                            Some(target) => (offsets[target] as i64).max(address),
                            None => i64::MAX,
                        };
                        if i8::try_from(target - (address + 2)).is_err() {
                            long_jumps.insert(event.index);
                            event.size = near_size;
                            stretch += growth as i64;
                            grown = true;
                        }
                    }
                    _ => {}
                }
            }
            if !grown {
                break;
            }
        }
        log::debug!(
            "branch relaxation: lines={}, events={}, rounds={rounds}",
            lines.len(),
            events.len()
        );
        self.layout_lines(
            lines,
            executable,
            &long_jumps,
            positions,
            reuse_encodings.then_some(&*cache),
        )
    }

    fn layout_lines(
        &mut self,
        lines: &[Line],
        executable: bool,
        long_jumps: &HashSet<usize>,
        positions: &HashMap<String, (String, usize)>,
        reuse: Option<&LayoutCache>,
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
        let mut rip_fixups = Vec::new();
        let mut line_offsets = Vec::with_capacity(lines.len());

        for (index, line) in lines.iter().enumerate() {
            line_offsets.push(current_offset);
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
                    } else if executable
                        && matches!(
                            self.target_arch,
                            TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64
                        )
                    {
                        let prefix = padding % 4;
                        code.resize(code.len() + prefix, 0);
                        for _ in 0..padding / 4 {
                            code.extend_from_slice(&0x0000_0013u32.to_le_bytes());
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
                Line::CfiAdvance(expression) => {
                    let value = absolute_expression(expression, positions).unwrap_or(0);
                    let bytes = crate::cfi::advance(value)?;
                    current_offset += bytes.len();
                    code.extend(bytes);
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

                    if matches!(
                        self.target_arch,
                        TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64
                    ) && let Some((target, jump)) = riscv_label_jump(&opcode, &inst.operands)
                    {
                        let mut placeholder = inst.clone();
                        if let Some(last) = placeholder.operands.last_mut() {
                            *last = "0".into();
                        }
                        let bytes = self
                            .encoder
                            .encode_instruction(&placeholder)
                            .map_err(|e| RasError::EncodingError(e.to_string()))?;
                        let word =
                            u32::from_le_bytes(bytes.as_slice().try_into().map_err(|_| {
                                RasError::EncodingError("RISC-V encoding is not 4 bytes".into())
                            })?);
                        code.extend_from_slice(&bytes);
                        patch_points.push(PatchPoint::Riscv {
                            offset: current_offset,
                            target,
                            word,
                            jump,
                        });
                        current_offset += 4;
                        continue;
                    }

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
                                relocate: opcode == "bl"
                                    && !target.starts_with(".L")
                                    && self.target_os != TargetOperatingSystem::MacOS,
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

                    if self.target_arch == TargetArchitecture::X86_64
                        && is_jmp_call
                        && inst.operands.len() == 1
                        && !inst.operands[0].trim().starts_with('*')
                    {
                        let target = inst.operands[0].trim();
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
                    } else if let Some(cache) = reuse {
                        let size = cache.sizes[index];
                        let start = cache.line_offsets[index];
                        code.extend_from_slice(&cache.code[start..start + size]);
                        if let Some(patch) = cache.rip_fixup(index).and_then(|fixup| {
                            rip_relative_patch(inst, fixup, current_offset, &global_labels)
                        }) {
                            patch_points.push(patch);
                        }
                        current_offset += size;
                    } else {
                        let bytes = self
                            .encoder
                            .encode_instruction(inst)
                            .map_err(|e| RasError::EncodingError(e.to_string()))?;
                        if let Some(fixup) = self.encoder.take_rip_fixup() {
                            rip_fixups.push((index, fixup));
                            if let Some(patch) =
                                rip_relative_patch(inst, fixup, current_offset, &global_labels)
                            {
                                patch_points.push(patch);
                            }
                        }
                        code.extend_from_slice(&bytes);
                        current_offset += bytes.len();
                    }
                }
            }
        }

        Ok(Layout {
            line_offsets,
            code,
            symbol_offsets,
            patch_points,
            mapping_symbols,
            short_jumps,
            rip_fixups,
        })
    }

    /// Mach-O private labels (`L` prefix, not global) never reach the symbol table.
fn macho_private_label(name: &str, global_symbols: &HashSet<&str>) -> bool {
    name.starts_with('L') && !global_symbols.contains(name)
}

fn macho_atom(
    symbol_offsets: &HashMap<String, usize>,
    global_symbols: &HashSet<&str>,
    offset: usize,
) -> Option<usize> {
    symbol_offsets
        .iter()
        .filter(|(name, _)| !Self::macho_private_label(name, global_symbols))
        .map(|(_, &at)| at)
        .filter(|&at| at <= offset)
        .max()
}

fn finish_layout(
        layout: Layout,
        lines: &[Line],
        positions: &HashMap<String, (String, usize)>,
        section: &str,
        global_symbols: &HashSet<&str>,
        macho: bool,
    ) -> EncodeResult {
        let Layout {
            mut code,
            symbol_offsets,
            patch_points,
            mapping_symbols,
            short_jumps,
            ..
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
                                trailing: 0,
                                branch: false,
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
                            trailing: 0,
                            branch: false,
                        });
                    }
                }
                continue;
            }
            if let PatchPoint::Riscv {
                offset,
                target,
                word,
                jump,
            } = patch
            {
                match symbol_offsets.get(target) {
                    Some(&target_offset) if !global_symbols.contains(target.as_str()) => {
                        let delta = target_offset as i64 - *offset as i64;
                        let imm = if *jump {
                            RiscVEncoder::j_type(RiscVEncoder::jump_offset(delta)?, 0, 0)
                        } else {
                            RiscVEncoder::b_type(RiscVEncoder::branch_offset(delta)?, 0, 0, 0, 0)
                        };
                        let instruction = word | imm;
                        code[*offset..*offset + 4].copy_from_slice(&instruction.to_le_bytes());
                    }
                    _ => external_relocs.push(ExternalReloc {
                        offset: *offset,
                        symbol: target.clone(),
                        section: String::new(),
                        kind: if *jump { 17 } else { 16 },
                        addend: 0,
                        trailing: 0,
                        branch: false,
                    }),
                }
                continue;
            }
            let target = match patch {
                PatchPoint::Data { .. } => unreachable!(),
                PatchPoint::Riscv { target, .. }
                | PatchPoint::AArch64 { target, .. }
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
                let preemptible_call = *kind == 283 && global_symbols.contains(target);
                if (!relocate || (*kind == 283 && !preemptible_call))
                    && let Some(&target_offset) = symbol_offsets.get(target)
                {
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
                        trailing: 0,
                        branch: false,
                    });
                }
                continue;
            }
            let got = target.ends_with("@GOTPCREL");
            let target = reloc_symbol(target);
            let (x86_rel, patch_offset) = match patch {
                PatchPoint::X86Rel32 { offset, .. } | PatchPoint::X86RipRel32 { offset, .. } => {
                    (true, *offset)
                }
                _ => (false, 0),
            };
            let target_offset = symbol_offsets.get(target).copied().filter(|&target_offset| {
                let same_atom = || {
                    Self::macho_atom(&symbol_offsets, global_symbols, target_offset)
                        == Self::macho_atom(&symbol_offsets, global_symbols, patch_offset)
                };
                let macho_resolves = !macho
                    || !x86_rel
                    || (Self::macho_private_label(target, global_symbols) && same_atom());
                !got
                    && macho_resolves
                    && !matches!(
                        patch,
                        PatchPoint::X86Rel32 { relocate: true, .. }
                            | PatchPoint::X86RipRel32 { relocate: true, .. }
                    )
            });
            if let Some(target_offset) = target_offset {
                match patch {
                    PatchPoint::Data { .. }
                    | PatchPoint::AArch64 { .. }
                    | PatchPoint::Riscv { .. } => unreachable!(),
                    PatchPoint::X86Rel32 { offset, .. } => {
                        let rel32 = (target_offset as i64) - (*offset as i64 + 4);
                        let rel32_bytes = (rel32 as i32).to_le_bytes();
                        code[*offset..*offset + 4].copy_from_slice(&rel32_bytes);
                    }
                    PatchPoint::X86RipRel32 {
                        offset,
                        trailing,
                        addend,
                        ..
                    } => {
                        let rel32 = (target_offset as i64) + *addend
                            - (*offset as i64 + 4 + *trailing as i64);
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
                            kind: if positions.contains_key(target)
                                && !global_symbols.contains(target)
                            {
                                2
                            } else {
                                4
                            },
                            addend: -4,
                            trailing: 0,
                            branch: true,
                        });
                    }
                    PatchPoint::X86RipRel32 {
                        offset,
                        trailing,
                        addend,
                        ..
                    } => {
                        code[*offset..*offset + 4].fill(0);
                        external_relocs.push(ExternalReloc {
                            offset: *offset,
                            symbol: target.to_string(),
                            section: String::new(),
                            kind: if got { R_X86_64_REX_GOTPCRELX } else { 2 },
                            addend: *addend - (4 + *trailing as i64),
                            trailing: *trailing,
                            branch: false,
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

        if !macho {
            symbols.extend(mapping_symbols);
        }
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
        self.current_module = Some(module);
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

fn riscv_label_jump(opcode: &str, operands: &[String]) -> Option<(String, bool)> {
    let jump = match opcode {
        "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" | "beqz" | "bnez" | "bltz" | "bgez"
        | "blez" | "bgtz" | "bgt" | "ble" | "bgtu" | "bleu" => false,
        "jal" | "j" => true,
        _ => return None,
    };
    let target = operands.last()?.trim();
    target
        .starts_with(|c: char| c.is_ascii_alphabetic() || matches!(c, '_' | '.' | '$'))
        .then(|| (target.to_string(), jump))
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
fn rip_relative_patch(
    inst: &ParsedInstruction,
    fixup: RipFixup,
    offset: usize,
    global_labels: &HashSet<&str>,
) -> Option<PatchPoint> {
    let (label, addend) = inst
        .operands
        .iter()
        .find_map(|op| extract_rip_label(op.trim().trim_start_matches('*')))?;
    Some(PatchPoint::X86RipRel32 {
        offset: offset + fixup.disp_offset,
        target: label.to_string(),
        trailing: fixup.trailing,
        addend,
        relocate: global_labels.contains(label),
    })
}

fn extract_rip_label(op: &str) -> Option<(&str, i64)> {
    let op = op
        .strip_prefix("jmp ")
        .or_else(|| op.strip_prefix("call "))
        .map_or(op, |operand| operand.trim().trim_start_matches('*'));
    let paren = op.find('(')?;
    let close = op.find(')')?;
    if close <= paren {
        return None;
    }
    let base = op[paren + 1..close].trim().trim_start_matches('%');
    if !base.eq_ignore_ascii_case("rip") {
        return None;
    }
    crate::encoder::x86_64::rip_expression(op[..paren].trim())
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

#[cfg(test)]
mod layout_cache_tests {
    use super::*;

    fn relaxation_source() -> String {
        let mut text = String::from("f:\n    jne .Lend\n");
        for _ in 0..5 {
            text.push_str("    jmp .Lfar\n");
        }
        for i in 0..46 {
            text.push_str("    xorl %eax, %eax\n");
            if i % 20 == 0 {
                text.push_str("    movq gsym(%rip), %rdx\n");
            }
        }
        text.push_str(".Lend:\n    movq gsym(%rip), %rax\n");
        for _ in 0..200 {
            text.push_str("    xorl %ecx, %ecx\n");
        }
        text.push_str(".Lfar:\n    ret\n");
        text
    }

    fn rip_patches(layout: &Layout) -> Vec<(usize, String)> {
        layout
            .patch_points
            .iter()
            .filter_map(|patch| match patch {
                PatchPoint::X86RipRel32 { offset, target, .. } => Some((*offset, target.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn cached_layout_matches_full_reencode_after_relaxation() {
        let parsed = AssemblyParser::new()
            .parse(&relaxation_source())
            .expect("parse");
        let positions = HashMap::new();
        let mut asm = RasAssembler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .expect("assembler");

        let mut cache = None;
        let cached = asm
            .layout_two_pass(&parsed.lines, true, &positions, &mut cache, true)
            .expect("cached layout");
        let mut full_cache = None;
        let full = asm
            .layout_two_pass(&parsed.lines, true, &positions, &mut full_cache, false)
            .expect("full layout");

        assert_eq!(cached.code[0], 0x0f, "top branch should be promoted");
        assert_eq!(cached.code, full.code);
        assert_eq!(cached.symbol_offsets, full.symbol_offsets);
        assert_eq!(rip_patches(&cached), rip_patches(&full));
        assert!(!rip_patches(&full).is_empty());
    }
}
