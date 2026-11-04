//! Assembly text parser
//!
//! Parses gas-compatible assembly: .text, .data, .global/.globl, directives,
//! labels, and instructions. Outputs `lines` preserving label/instruction order
//! for two-pass assembly (jmp/call with label resolution).

use crate::{encoder::traits::ParsedInstruction, error::RasError};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Section {
    pub group: Option<String>,
    pub name: String,
    pub flags: SectionFlags,
    pub data: Vec<u8>,
    pub alignment: usize,
    pub macho: Option<MachOSectionType>,
}

#[derive(Debug, Clone)]
pub struct SectionFlags {
    pub alloc: bool,
    pub exec: bool,
    pub write: bool,
    pub tls: bool,
}

/// Mach-O section type and attributes, as written in the `.section` directive.
/// `kind` is the low byte of `S_*` section flags and `attributes` the high bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachOSectionType {
    pub kind: u32,
    pub attributes: u32,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub global: bool,
    pub section: String,
}

#[derive(Debug, Clone)]
pub struct DebugFile {
    pub directory: Option<String>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub enum Line {
    Section(String),
    Loc {
        file: u32,
        line: u32,
        column: u32,
        /// `is_stmt` and `isa` are line-program registers that persist across rows;
        /// the three flags and `discriminator` are reset by every row the program
        /// emits. DWARF 4 section 6.2.2.
        is_stmt: bool,
        prologue_end: bool,
        epilogue_begin: bool,
        basic_block: bool,
        discriminator: u64,
        isa: u64,
    },
    Label(Symbol),
    Instruction(ParsedInstruction),
    Data(Vec<u8>),
    CfiAdvance(String),
    Leb128 {
        expression: String,
        signed: bool,
    },
    DataExpression {
        width: usize,
        expression: String,
    },
    Align {
        value: usize,
        directive: String,
        fill: Option<u8>,
        max_skip: Option<usize>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct SymbolAttributes {
    pub internal: bool,
    pub retain: bool,
    pub weak: bool,
    pub global: bool,
    pub hidden: bool,
    pub private_extern: bool,
    pub internal_visibility: bool,
    pub protected: bool,
    pub local: bool,
    pub common: Option<(u64, u64)>,
    pub symbol_type: u8,
    pub size: Option<String>,
    pub local_entry: Option<String>,
}

/// An ARM build attribute value: EABI tags take either a number or a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArmAttributeValue {
    Number(u32),
    Text(String),
}

pub struct ParsedAssembly {
    pub common_order: Vec<String>,
    pub aliases: Vec<(String, String)>,
    pub attributes: HashMap<String, SymbolAttributes>,
    pub sections: Vec<Section>,
    pub symbols: Vec<Symbol>,
    pub instructions: Vec<ParsedInstruction>,
    pub lines: Vec<Line>,
    pub files: BTreeMap<u32, DebugFile>,
    pub arm_attributes: Vec<(u32, ArmAttributeValue)>,
}

pub struct AssemblyParser {
    common_order: Vec<String>,
    generated_prefix: String,
    merge_sections: HashSet<String>,
    aliases: Vec<(String, String)>,
    current_section: String,
    sections: Vec<Section>,
    symbols: Vec<Symbol>,
    lines: Vec<Line>,
    files: BTreeMap<u32, DebugFile>,
    is_stmt: bool,
    isa: u64,
    attributes: HashMap<String, SymbolAttributes>,
    pending_globals: HashSet<String>,
    symbol_names: HashSet<String>,
    /// Name of the symbol a `.def`/`.endef` block is currently describing.
    coff_def: Option<String>,
    arm_attributes: Vec<(u32, ArmAttributeValue)>,
    /// Darwin reads the third `.comm` operand and `.align` as log2 exponents.
    darwin: bool,
    /// COFF reads `.section` flags with its own letters, not ELF's.
    coff: bool,
    line_comments: &'static [&'static str],
}

impl Default for AssemblyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AssemblyParser {
    pub fn new() -> Self {
        Self {
            common_order: Vec::new(),
            generated_prefix: ".Lras_size".into(),
            merge_sections: HashSet::new(),
            aliases: Vec::new(),
            current_section: ".text".to_string(),
            sections: Vec::new(),
            symbols: Vec::new(),
            lines: Vec::new(),
            files: BTreeMap::new(),
            is_stmt: true,
            isa: 0,
            attributes: HashMap::new(),
            pending_globals: HashSet::new(),
            symbol_names: HashSet::new(),
            coff_def: None,
            arm_attributes: Vec::new(),
            darwin: false,
            coff: false,
            line_comments: &["#", "//"],
        }
    }

    pub fn with_darwin_directives(mut self, darwin: bool) -> Self {
        self.darwin = darwin;
        self
    }

    pub fn with_coff_directives(mut self, coff: bool) -> Self {
        self.coff = coff;
        self
    }

    /// Markers that start a comment running to the end of a line, outside string literals.
    pub fn with_line_comments(mut self, markers: &'static [&'static str]) -> Self {
        self.line_comments = markers;
        self
    }

    pub fn parse(mut self, text: &str) -> Result<ParsedAssembly, RasError> {
        while text.contains(&self.generated_prefix) {
            self.generated_prefix.push('_');
        }
        for line in text.lines() {
            let line = strip_line_comment(line.trim(), self.line_comments).trim();
            if line.is_empty() {
                continue;
            }

            // A label token has no internal whitespace and ends with ':'.
            if let Some(colon_pos) = line.find(':') {
                let before = &line[..colon_pos];
                if !before.is_empty() && !before.contains(char::is_whitespace) {
                    let label = before.trim();
                    if !is_numeric_label(label) && !self.symbol_names.insert(label.to_string()) {
                        return Err(RasError::ParseError(format!(
                            "Symbol already defined: {label}"
                        )));
                    }
                    let global = !is_numeric_label(label) && self.pending_globals.remove(label);
                    let sym = Symbol {
                        name: label.to_string(),
                        global,
                        section: self.current_section.clone(),
                    };
                    self.symbols.push(sym.clone());
                    self.lines.push(Line::Label(sym));
                    let rest = line[colon_pos + 1..].trim();
                    if !rest.is_empty() {
                        if rest.starts_with('.') {
                            self.parse_directive(rest)?;
                        } else {
                            self.parse_instruction(rest)?;
                        }
                    }
                    continue;
                }
            }

            // clang on Darwin names its DWARF lengths with `Lset0 = a-b`.
            if self.darwin
                && let Some((name, expression)) = line.split_once('=')
                && !name.trim().contains(char::is_whitespace)
            {
                parse_data_expression(expression.trim())?;
                self.aliases
                    .push((symbol_name(name.trim())?.into(), expression.trim().into()));
                continue;
            }

            if line.starts_with('.') {
                self.parse_directive(line)?;
            } else {
                self.parse_instruction(line)?;
            }
        }

        self.resolve_numeric_labels()?;
        for symbol in &self.symbols {
            if self.merge_sections.contains(&symbol.section) {
                self.attributes
                    .entry(symbol.name.clone())
                    .or_default()
                    .retain = true;
            }
        }
        for line in &mut self.lines {
            if let Line::Label(symbol) = line
                && let Some(attributes) = self.attributes.get(&symbol.name)
            {
                symbol.global =
                    !attributes.local && (symbol.global || attributes.global || attributes.weak);
            }
        }
        self.symbols = self
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Label(symbol) => Some(symbol.clone()),
                _ => None,
            })
            .collect();

        let instructions = self
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Instruction(inst) => Some(inst.clone()),
                _ => None,
            })
            .collect();

        Ok(ParsedAssembly {
            common_order: self.common_order,
            aliases: self.aliases,
            attributes: self.attributes,
            sections: self.sections,
            symbols: self.symbols,
            instructions,
            lines: self.lines,
            files: self.files,
            arm_attributes: self.arm_attributes,
        })
    }

    fn resolve_numeric_labels(&mut self) -> Result<(), RasError> {
        let mut definitions = Vec::new();
        for (index, line) in self.lines.iter().enumerate() {
            if let Line::Label(symbol) = line
                && is_numeric_label(&symbol.name)
            {
                let mut name = format!(".Lras_numeric_{index}");
                while self.symbol_names.contains(&name) {
                    name.push('_');
                }
                definitions.push((index, symbol.name.clone(), name));
            }
        }
        let mut symbols_changed = false;
        for (index, line) in self.lines.iter_mut().enumerate() {
            match line {
                Line::Label(symbol) => {
                    if let Some((_, _, name)) = definitions.iter().find(|d| d.0 == index) {
                        self.attributes.entry(name.clone()).or_default().internal = true;
                        symbol.name = name.clone();
                        symbols_changed = true;
                    }
                }
                Line::Instruction(inst) => {
                    for operand in &mut inst.operands {
                        if let Cow::Owned(text) =
                            resolve_numeric_references(operand, index, &definitions)?
                        {
                            *operand = text;
                        }
                    }
                }
                Line::DataExpression { expression, .. }
                | Line::Leb128 { expression, .. }
                | Line::CfiAdvance(expression) => {
                    if let Cow::Owned(text) =
                        resolve_numeric_references(expression, index, &definitions)?
                    {
                        *expression = text;
                    }
                }
                _ => {}
            }
        }
        if symbols_changed {
            self.symbols = self
                .lines
                .iter()
                .filter_map(|line| match line {
                    Line::Label(symbol) => Some(symbol.clone()),
                    _ => None,
                })
                .collect();
        }
        Ok(())
    }

    fn parse_directive(&mut self, line: &str) -> Result<(), RasError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        match parts[0] {
            ".text" | ".data" | ".bss" | ".section" => {
                let fields = if parts[0] == ".section" {
                    let operand = line
                        .trim_start()
                        .strip_prefix(".section")
                        .unwrap_or_default()
                        .trim();
                    split_operands(operand)
                } else {
                    Vec::new()
                };
                let (name, macho) = if parts[0] == ".section" {
                    let segment = fields
                        .first()
                        .map(|field| field.trim_matches('"'))
                        .filter(|field| !field.is_empty())
                        .ok_or_else(|| RasError::ParseError(".section requires a name".into()))?;
                    match fields
                        .get(1)
                        .filter(|section| segment.starts_with("__") && !section.starts_with('"'))
                    {
                        Some(section) => {
                            let spec = macho_section_type(&fields[2..])?;
                            match macho_builtin_section(segment, section) {
                                Some((alias, builtin)) => {
                                    if !fields[2..].is_empty()
                                        && (spec.kind != builtin.kind
                                            || spec.attributes & !builtin.attributes != 0)
                                    {
                                        return Err(RasError::ParseError(format!(
                                            "Mach-O section {segment},{section} is built in with a different type or attributes"
                                        )));
                                    }
                                    (alias.to_string(), None)
                                }
                                None => (format!("{segment},{section}"), Some(spec)),
                            }
                        }
                        None => (segment.to_string(), None),
                    }
                } else {
                    (parts[0].to_string(), None)
                };
                let name = name.as_str();
                let explicit = fields
                    .get(1)
                    .filter(|field| field.starts_with('"'))
                    .map(|field| field.trim_matches('"'));
                if explicit.is_some_and(|flags| flags.contains('M')) {
                    self.merge_sections.insert(name.to_string());
                }
                self.current_section = name.to_string();
                self.lines.push(Line::Section(name.to_string()));
                if let Some(existing) = self.sections.iter().find(|s| s.name == name) {
                    if macho.is_some() && existing.macho != macho {
                        return Err(RasError::ParseError(format!(
                            "Mach-O section {name} redeclared with a different type or attributes"
                        )));
                    }
                } else {
                    let macho_exec = macho
                        .is_some_and(|spec| spec.attributes & MACHO_PURE_OR_SOME_INSTRUCTIONS != 0);
                    self.sections.push(Section {
                        group: if explicit.is_some_and(|f| f.contains('G')) {
                            if fields.len() != 5 || fields[4] != "comdat" || fields[3].is_empty() {
                                return Err(RasError::ParseError(
                                    "Unsupported section group".into(),
                                ));
                            }
                            Some(fields[3].as_str().into())
                        } else {
                            None
                        },
                        name: name.to_string(),
                        flags: match explicit {
                            Some(f) if self.coff => coff_section_flags(name, f),
                            _ => SectionFlags {
                                alloc: explicit.map_or(
                                    !name.starts_with(".debug") && name != ".note.GNU-stack",
                                    |f| f.contains('a'),
                                ),
                                exec: explicit
                                    .map_or(name == ".text" || macho_exec, |f| f.contains('x')),
                                write: explicit.map_or(
                                    matches!(name, ".data" | ".bss" | ".tdata" | ".tbss")
                                        || name.starts_with("__DATA,"),
                                    |f| f.contains('w'),
                                ),
                                tls: explicit.map_or(matches!(name, ".tdata" | ".tbss"), |f| {
                                    f.contains('T')
                                }),
                            },
                        },
                        data: Vec::new(),
                        alignment: 1,
                        macho,
                    });
                }
            }
            ".set" | ".equ" => {
                let (name, expression) =
                    line[parts[0].len()..].split_once(',').ok_or_else(|| {
                        RasError::ParseError(".set requires symbol and expression".into())
                    })?;
                parse_data_expression(expression.trim())?;
                self.aliases
                    .push((symbol_name(name.trim())?.into(), expression.trim().into()));
            }
            ".zerofill" => {
                // .zerofill segment,section[,symbol,size[,align]]
                let fields: Vec<_> = line[parts[0].len()..].split(',').map(str::trim).collect();
                let [segment, section, rest @ ..] = fields.as_slice() else {
                    return Err(RasError::ParseError("Invalid .zerofill operands".into()));
                };
                if let [symbol, size, align @ ..] = rest {
                    let resume = self.current_section.clone();
                    self.parse_directive(&format!(".section {segment},{section},zerofill"))?;
                    if let [align] = align {
                        self.parse_directive(&format!(".p2align {align}"))?;
                    }
                    self.parse_line_label(symbol)?;
                    self.parse_directive(&format!(".zero {size}"))?;
                    self.switch_back_to(resume);
                }
            }
            ".comm" => {
                let fields: Vec<_> = line[parts[0].len()..].split(',').map(str::trim).collect();
                if !(2..=3).contains(&fields.len()) || fields[0].is_empty() {
                    return Err(RasError::ParseError("Invalid .comm operands".into()));
                }
                symbol_name(fields[0])?;
                let size = u64::try_from(parse_integer(fields[1])?)
                    .map_err(|_| RasError::ParseError("Invalid common size".into()))?;
                let alignment = match fields.get(2) {
                    Some(value) => {
                        let operand = u64::try_from(parse_integer(value)?)
                            .map_err(|_| RasError::ParseError("Invalid common alignment".into()))?;
                        if self.darwin {
                            u32::try_from(operand)
                                .ok()
                                .and_then(|exponent| 1u64.checked_shl(exponent))
                                .ok_or_else(|| {
                                    RasError::ParseError("Invalid common alignment".into())
                                })?
                        } else {
                            operand
                        }
                    }
                    None if self.darwin => 1,
                    None => size.next_power_of_two().min(16),
                };
                if !alignment.is_power_of_two() {
                    return Err(RasError::ParseError("Invalid common alignment".into()));
                }
                let attributes = self.attributes.entry(fields[0].into()).or_default();
                if let Some((old_size, old_alignment)) = attributes.common {
                    attributes.common = Some((size.max(old_size), alignment.max(old_alignment)));
                } else {
                    self.common_order.push(fields[0].into());
                    attributes.common = Some((size, alignment));
                }
                attributes.symbol_type = 1;
            }
            ".local" => {
                for name in line[parts[0].len()..].split(',').map(str::trim) {
                    if name.is_empty() {
                        return Err(RasError::ParseError(".local requires symbol".into()));
                    }
                    symbol_name(name)?;
                    self.attributes.entry(name.into()).or_default().local = true;
                }
            }
            ".internal" | ".protected" => {
                for name in line[parts[0].len()..].split(',').map(str::trim) {
                    if name.is_empty() {
                        return Err(RasError::ParseError(format!(
                            "{} requires symbol",
                            parts[0]
                        )));
                    }
                    symbol_name(name)?;
                    let attributes = self.attributes.entry(name.into()).or_default();
                    if parts[0] == ".internal" {
                        attributes.internal_visibility = true;
                    } else {
                        attributes.protected = true;
                    }
                }
            }
            ".hidden" => {
                for name in line[parts[0].len()..].split(',').map(str::trim) {
                    if name.is_empty() {
                        return Err(RasError::ParseError(".hidden requires symbol".into()));
                    }
                    symbol_name(name)?;
                    self.attributes.entry(name.into()).or_default().hidden = true;
                }
            }
            ".private_extern" => {
                for name in line[parts[0].len()..].split(',').map(str::trim) {
                    if name.is_empty() {
                        return Err(RasError::ParseError(
                            ".private_extern requires symbol".into(),
                        ));
                    }
                    symbol_name(name)?;
                    self.attributes
                        .entry(name.into())
                        .or_default()
                        .private_extern = true;
                }
            }
            ".weak" | ".weak_definition" | ".weak_reference" => {
                for name in line[parts[0].len()..].split(',').map(str::trim) {
                    if name.is_empty() {
                        return Err(RasError::ParseError(format!(
                            "{} requires symbol",
                            parts[0]
                        )));
                    }
                    symbol_name(name)?;
                    self.attributes.entry(name.into()).or_default().weak = true;
                }
            }
            ".global" | ".globl" => {
                if parts.len() < 2 {
                    return Err(RasError::ParseError(
                        ".global requires a symbol name".to_string(),
                    ));
                }
                for name in line[parts[0].len()..]
                    .split([',', ' ', '\t'])
                    .filter(|s| !s.is_empty())
                {
                    self.pending_globals.insert(name.to_string());
                    self.attributes.entry(name.into()).or_default().global = true;
                }
            }
            ".asciz" | ".string" => {
                let rest = line[parts[0].len()..].trim();
                let bytes = parse_quoted_string_bytes(rest, true).ok_or_else(|| {
                    RasError::ParseError(format!("{} requires a quoted string", parts[0]))
                })?;
                self.lines.push(Line::Data(bytes));
            }
            ".ascii" => {
                let rest = line[parts[0].len()..].trim();
                let bytes = parse_quoted_string_bytes(rest, false).ok_or_else(|| {
                    RasError::ParseError(format!("{} requires a quoted string", parts[0]))
                })?;
                self.lines.push(Line::Data(bytes));
            }
            ".ras_cfi_advance" => {
                let expression = line[parts[0].len()..].trim();
                parse_data_expression(expression)?;
                self.lines.push(Line::CfiAdvance(expression.into()));
            }
            ".uleb128" | ".sleb128" => {
                for item in line[parts[0].len()..].split(',') {
                    parse_data_expression(item.trim())?;
                    self.lines.push(Line::Leb128 {
                        expression: item.trim().into(),
                        signed: parts[0] == ".sleb128",
                    });
                }
            }
            ".float" | ".single" | ".double" => {
                for item in line[parts[0].len()..].split(',') {
                    let value: f64 = item.trim().parse().map_err(|_| {
                        RasError::ParseError(format!("Invalid float: {}", item.trim()))
                    })?;
                    let bytes = if parts[0] == ".double" {
                        value.to_le_bytes().to_vec()
                    } else {
                        (value as f32).to_le_bytes().to_vec()
                    };
                    self.lines.push(Line::Data(bytes));
                }
            }
            ".byte" | ".short" | ".hword" | ".value" | ".int" | ".long" | ".quad" | ".xword"
            | ".word" | ".half" | ".dword" | ".2byte" | ".4byte" | ".8byte" => {
                let width = match parts[0] {
                    ".byte" => 1,
                    ".short" | ".hword" | ".value" | ".half" | ".2byte" => 2,
                    ".quad" | ".xword" | ".dword" | ".8byte" => 8,
                    _ => 4,
                };
                for item in line[parts[0].len()..].split(',') {
                    let expression = item.trim();
                    if let Ok(value) = parse_integer(expression) {
                        self.lines
                            .push(Line::Data(value.to_le_bytes()[..width].to_vec()));
                    } else {
                        parse_data_expression(expression)?;
                        self.lines.push(Line::DataExpression {
                            width,
                            expression: expression.into(),
                        });
                    }
                }
            }
            ".zero" | ".space" | ".skip" => {
                let items: Vec<_> = line[parts[0].len()..].split(',').map(str::trim).collect();
                let count = usize::try_from(parse_integer(items[0])?)
                    .map_err(|_| RasError::ParseError("Invalid space count".into()))?;
                let fill = items.get(1).map_or(Ok(0), |s| parse_integer(s))? as u8;
                self.lines.push(Line::Data(vec![fill; count]));
            }
            ".align" | ".balign" | ".p2align" => {
                let items: Vec<_> = line[parts[0].len()..].split(',').map(str::trim).collect();
                if items.len() > 3 {
                    return Err(RasError::ParseError(format!(
                        "{} takes at most three operands",
                        parts[0]
                    )));
                }
                let value = usize::try_from(parse_integer(items[0])?)
                    .map_err(|_| RasError::ParseError("Invalid alignment".into()))?;
                let fill = items
                    .get(1)
                    .filter(|s| !s.is_empty())
                    .map(|s| parse_integer(s).map(|n| n as u8))
                    .transpose()?;
                let max_skip = items
                    .get(2)
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        parse_integer(s).and_then(|n| {
                            usize::try_from(n).map_err(|_| {
                                RasError::ParseError("Invalid maximum alignment padding".into())
                            })
                        })
                    })
                    .transpose()?
                    .filter(|maximum| *maximum != 0);
                let directive = if self.darwin && parts[0] == ".align" {
                    ".p2align"
                } else {
                    parts[0]
                };
                self.lines.push(Line::Align {
                    value,
                    directive: directive.into(),
                    fill,
                    max_skip,
                });
            }
            ".def" => {
                let name = parts
                    .get(1)
                    .ok_or_else(|| RasError::ParseError("Invalid .def".into()))?;
                self.coff_def = Some(symbol_name(name.trim_end_matches(';').trim())?.into());
            }
            ".scl" => {}
            ".endef" => self.coff_def = None,
            ".type" if self.coff_def.is_some() => {
                let name = self.coff_def.clone().unwrap_or_default();
                let value: u16 = parts
                    .get(1)
                    .and_then(|v| v.trim_end_matches(';').trim().parse().ok())
                    .ok_or_else(|| RasError::ParseError("Invalid .type in .def".into()))?;
                if value >> 4 == 2 {
                    self.attributes.entry(name).or_default().symbol_type = 2;
                }
            }
            ".type" => {
                let (name, kind) = line[parts[0].len()..]
                    .split_once(',')
                    .ok_or_else(|| RasError::ParseError("Invalid .type".into()))?;
                let name = symbol_name(name.trim())?;
                let kind = match kind.trim().trim_start_matches(['@', '%']) {
                    "notype" => 0,
                    "object" => 1,
                    "function" => 2,
                    "tls_object" => 6,
                    _ => return Err(RasError::ParseError("Unsupported symbol type".into())),
                };
                self.attributes
                    .entry(name.trim().into())
                    .or_default()
                    .symbol_type = kind;
            }
            ".size" => {
                let (name, expression) = line[parts[0].len()..]
                    .split_once(',')
                    .ok_or_else(|| RasError::ParseError("Invalid .size".into()))?;
                let name = symbol_name(name.trim())?;
                let expression = match parse_data_expression(expression.trim())? {
                    DataExpression::Difference(left, right) if left == "." || right == "." => {
                        let location = format!("{}_{}", self.generated_prefix, self.lines.len());
                        self.attributes
                            .entry(location.clone())
                            .or_default()
                            .internal = true;
                        self.lines.push(Line::Label(Symbol {
                            name: location.clone(),
                            global: false,
                            section: self.current_section.clone(),
                        }));
                        format!(
                            "{}-{}",
                            if left == "." { &location } else { &left },
                            if right == "." { &location } else { &right }
                        )
                    }
                    _ => expression.trim().into(),
                };
                self.attributes.entry(name.trim().into()).or_default().size = Some(expression);
            }
            ".localentry" => {
                let (name, expression) = line[parts[0].len()..]
                    .split_once(',')
                    .ok_or_else(|| RasError::ParseError("Invalid .localentry".into()))?;
                let name = symbol_name(name.trim())?;
                self.attributes
                    .entry(name.trim().into())
                    .or_default()
                    .local_entry = Some(expression.trim().into());
            }
            ".abiversion" => {
                if line[parts[0].len()..].trim() != "2" {
                    return Err(RasError::ParseError(
                        "Only ELFv2 is supported: .abiversion must be 2".into(),
                    ));
                }
            }
            ".extern" => {
                // Symbols are undefined until something defines them, so gas ignores
                // this too; the name only has to be well formed.
                for name in line[parts[0].len()..]
                    .split([',', ' ', '\t'])
                    .filter(|s| !s.is_empty())
                {
                    symbol_name(name)?;
                }
            }
            ".file" => {
                let operands = split_fields(line[parts[0].len()..].trim());
                if let Some(Ok(number)) = operands.first().map(|field| field.parse::<u32>()) {
                    let quoted = |field: &str| {
                        parse_quoted_string_bytes(field, false)
                            .and_then(|bytes| String::from_utf8(bytes).ok())
                    };
                    let (strings, checksum) = match operands.iter().position(|f| *f == "md5") {
                        Some(at) => (&operands[1..at], &operands[at + 1..]),
                        None => (&operands[1..], &operands[operands.len()..]),
                    };
                    let file = match (strings, checksum) {
                        ([name], []) | ([name], [_]) => quoted(name).map(|name| DebugFile {
                            directory: None,
                            name,
                        }),
                        ([directory, name], []) | ([directory, name], [_]) => quoted(directory)
                            .zip(quoted(name))
                            .map(|(directory, name)| DebugFile {
                                directory: Some(directory),
                                name,
                            }),
                        _ => None,
                    };
                    let file = file.ok_or_else(|| {
                        RasError::ParseError(format!("Invalid .file operands: {line}"))
                    })?;
                    self.files.insert(number, file);
                }
            }
            ".loc" => {
                let operands: Vec<&str> = line[parts[0].len()..].split_whitespace().collect();
                let number = |text: Option<&&str>, what: &str| {
                    text.and_then(|text| text.parse::<u32>().ok())
                        .ok_or_else(|| RasError::ParseError(format!("Invalid .loc {what}")))
                };
                let file = number(operands.first(), "file number")?;
                let source_line = number(operands.get(1), "line")?;
                let (column, options_start) = match operands.get(2) {
                    Some(text) if text.parse::<u32>().is_ok() => (number(Some(text), "column")?, 3),
                    _ => (0, 2),
                };
                let mut options = operands[options_start..].iter();
                let (mut prologue_end, mut epilogue_begin, mut basic_block) = (false, false, false);
                let mut discriminator = 0u64;
                while let Some(&option) = options.next() {
                    let value = |text: Option<&&str>| {
                        text.and_then(|text| text.parse::<u64>().ok())
                            .ok_or_else(|| {
                                RasError::ParseError(format!(".loc {option} requires a number"))
                            })
                    };
                    match option {
                        "is_stmt" => {
                            self.is_stmt = match options.next() {
                                Some(&"0") => false,
                                Some(&"1") => true,
                                _ => {
                                    return Err(RasError::ParseError(
                                        ".loc is_stmt requires 0 or 1".into(),
                                    ));
                                }
                            };
                        }
                        "prologue_end" => prologue_end = true,
                        "epilogue_begin" => epilogue_begin = true,
                        "basic_block" => basic_block = true,
                        "discriminator" => discriminator = value(options.next())?,
                        "isa" => self.isa = value(options.next())?,
                        "view" => {
                            return Err(RasError::ParseError(
                                ".loc view numbers are a gas bookkeeping feature ras does not implement".into(),
                            ));
                        }
                        other => {
                            return Err(RasError::ParseError(format!(
                                "Unsupported .loc option: {other}"
                            )));
                        }
                    }
                }
                self.lines.push(Line::Loc {
                    file,
                    line: source_line,
                    column,
                    is_stmt: self.is_stmt,
                    prologue_end,
                    epilogue_begin,
                    basic_block,
                    discriminator,
                    isa: self.isa,
                });
            }
            ".ident" => {}
            // Hints only: the linker takes the deployment target from its own arguments,
            // and dropping a linker optimization hint leaves the code correct.
            ".build_version" | ".macosx_version_min" | ".loh" | ".subsections_via_symbols" => {}
            ".syntax" => {
                if parts.get(1).map(|s| s.trim()) != Some("unified") {
                    return Err(RasError::ParseError(
                        "ras reads unified ARM syntax only".into(),
                    ));
                }
            }
            ".arm" => {}
            ".thumb" | ".thumb_func" | ".code" | ".force_thumb" => {
                return Err(RasError::ParseError(format!(
                    "{} selects Thumb, which ras does not encode",
                    parts[0]
                )));
            }
            ".arch" | ".arch_extension" | ".fpu" => {
                let name = parts
                    .get(1)
                    .map(|s| s.trim())
                    .ok_or_else(|| RasError::ParseError(format!("{} needs a name", parts[0])))?;
                self.arm_attributes
                    .extend(arm_isa_attributes(parts[0], name)?);
            }
            ".eabi_attribute" => {
                let (tag, value) = line[parts[0].len()..]
                    .split_once(',')
                    .ok_or_else(|| RasError::ParseError("Invalid .eabi_attribute".into()))?;
                let tag = u32::try_from(parse_integer(tag.trim())?)
                    .map_err(|_| RasError::ParseError("Invalid EABI attribute tag".into()))?;
                let value = value.trim();
                let value = match parse_quoted_string_bytes(value, false) {
                    Some(bytes) => ArmAttributeValue::Text(
                        String::from_utf8(bytes)
                            .map_err(|_| RasError::ParseError("Invalid EABI string".into()))?,
                    ),
                    None => {
                        ArmAttributeValue::Number(u32::try_from(parse_integer(value)?).map_err(
                            |_| RasError::ParseError("Invalid EABI attribute value".into()),
                        )?)
                    }
                };
                self.arm_attributes.push((tag, value));
            }
            _ => {
                return Err(RasError::ParseError(format!(
                    "Unsupported directive: {}",
                    parts[0]
                )));
            }
        }

        Ok(())
    }

    fn switch_back_to(&mut self, section: String) {
        self.current_section = section.clone();
        self.lines.push(Line::Section(section));
    }

    fn parse_line_label(&mut self, label: &str) -> Result<(), RasError> {
        if !self.symbol_names.insert(label.to_string()) {
            return Err(RasError::ParseError(format!(
                "Symbol already defined: {label}"
            )));
        }
        let global = self.pending_globals.remove(label);
        let sym = Symbol {
            name: label.to_string(),
            global,
            section: self.current_section.clone(),
        };
        self.symbols.push(sym.clone());
        self.lines.push(Line::Label(sym));
        Ok(())
    }

    fn parse_instruction(&mut self, line: &str) -> Result<(), RasError> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(());
        }

        let (opcode, operand_text) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let operand_text = operand_text.trim();
        let operands = if operand_text.is_empty() {
            Vec::new()
        } else {
            split_operands(operand_text)
        };

        self.lines.push(Line::Instruction(ParsedInstruction {
            opcode: opcode.to_string(),
            operands,
        }));
        Ok(())
    }
}

/// EABI build attributes implied by `.arch`, `.arch_extension` and `.fpu`.
/// ras names what it does not know rather than leaving the object silently
/// describing the wrong ABI.
fn arm_isa_attributes(
    directive: &str,
    name: &str,
) -> Result<Vec<(u32, ArmAttributeValue)>, RasError> {
    use ArmAttributeValue::{Number, Text};
    let attributes = match (directive, name) {
        (".arch", "armv7-a") => vec![
            (5, Text("7-A".into())),
            (6, Number(10)),
            (7, Number(u32::from(b'A'))),
            (8, Number(1)),
            (9, Number(2)),
        ],
        (".fpu", "vfpv3-d16") => vec![(10, Number(4))],
        (".arch_extension", "idiv") => vec![(44, Number(2))],
        _ => {
            return Err(RasError::ParseError(format!(
                "ras has no build attributes for `{directive} {name}`"
            )));
        }
    };
    Ok(attributes)
}

fn is_numeric_label(name: &str) -> bool {
    name.len() == 1 && name.as_bytes()[0].is_ascii_digit()
}

fn resolve_numeric_references<'a>(
    text: &'a str,
    position: usize,
    definitions: &[(usize, String, String)],
) -> Result<Cow<'a, str>, RasError> {
    let mut result: Option<String> = None;
    let mut copied = 0;
    let mut start = 0;
    for (end, ch) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        if ch.is_alphanumeric() || matches!(ch, '_' | '.' | '$') {
            continue;
        }
        if start < end {
            let token = &text[start..end];
            if token.len() == 2
                && token.as_bytes()[0].is_ascii_digit()
                && matches!(token.as_bytes()[1], b'f' | b'b')
            {
                let digit = &token[..1];
                let definition = if token.ends_with('f') {
                    definitions.iter().find(|d| d.1 == digit && d.0 >= position)
                } else {
                    definitions
                        .iter()
                        .rev()
                        .find(|d| d.1 == digit && d.0 < position)
                }
                .ok_or_else(|| RasError::ParseError(format!("Undefined numeric label: {token}")))?;
                let out = result.get_or_insert_with(|| String::with_capacity(text.len()));
                out.push_str(&text[copied..start]);
                out.push_str(&definition.2);
                copied = end;
            }
        }
        start = end + ch.len_utf8();
    }
    Ok(match result {
        Some(mut out) => {
            out.push_str(&text[copied..]);
            Cow::Owned(out)
        }
        None => Cow::Borrowed(text),
    })
}

pub(crate) enum DataExpression {
    Constant(i128),
    Symbol(String, i128),
    Difference(String, String),
}

pub(crate) fn parse_data_expression(text: &str) -> Result<DataExpression, RasError> {
    if let Some(value) = parse_integer_opt(text) {
        return Ok(DataExpression::Constant(value));
    }
    if let Some(suffix) = crate::encoder::x86_64::unsupported_tls_suffix(text) {
        return Err(crate::encoder::x86_64::unsupported_tls_error(suffix));
    }
    let symbol = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '$'))
            && !s.as_bytes()[0].is_ascii_digit()
    };
    let operand = |s: &str| {
        symbol(s)
            || (s.len() == 2
                && s.as_bytes()[0].is_ascii_digit()
                && matches!(s.as_bytes()[1], b'f' | b'b'))
    };
    if let Some(name) = text.strip_suffix("@tpoff")
        && symbol(name)
    {
        return Ok(DataExpression::Symbol(text.into(), 0));
    }
    if let Some((index, op)) = text.char_indices().find(|(_, c)| matches!(c, '+' | '-')) {
        let left = text[..index].trim();
        let right = text[index + 1..].trim();
        if operand(left) {
            if let Some(value) = parse_integer_opt(right) {
                return Ok(DataExpression::Symbol(
                    left.into(),
                    if op == '-' { -value } else { value },
                ));
            }
            if op == '-' && operand(right) {
                return Ok(DataExpression::Difference(left.into(), right.into()));
            }
        }
    } else if operand(text) {
        return Ok(DataExpression::Symbol(text.into(), 0));
    }
    Err(RasError::ParseError(format!(
        "Invalid data expression: {text}"
    )))
}

pub(crate) fn parse_integer(text: &str) -> Result<i128, RasError> {
    parse_integer_opt(text).ok_or_else(|| RasError::ParseError(format!("Invalid integer: {text}")))
}

fn parse_integer_opt(text: &str) -> Option<i128> {
    let (sign, digits) = text.strip_prefix('-').map_or((1, text), |s| (-1, s));
    let result = if let Some(hex) = digits.strip_prefix("0x") {
        i128::from_str_radix(hex, 16)
    } else {
        digits.parse()
    };
    result.ok().map(|n| sign * n)
}

fn symbol_name(name: &str) -> Result<&str, RasError> {
    let mut chars = name.chars();
    let valid = matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '.')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$'));
    if valid {
        Ok(name)
    } else {
        Err(RasError::ParseError(format!("Invalid symbol name: {name}")))
    }
}

fn strip_line_comment<'a>(line: &'a str, markers: &[&str]) -> &'a str {
    let mut in_string = false;
    let mut escaped = false;
    for (index, c) in line.char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
        } else if markers
            .iter()
            .any(|marker| line[index..].starts_with(marker))
        {
            return &line[..index];
        }
    }
    line
}

fn parse_quoted_string_bytes(s: &str, null_terminate: bool) -> Option<Vec<u8>> {
    let body = s.trim().strip_prefix('"')?;
    let mut out = Vec::new();
    let mut chars = body.char_indices().peekable();
    loop {
        let (index, c) = chars.next()?;
        match c {
            '"' => {
                let rest = body[index + 1..].trim();
                if !(rest.is_empty() || rest.starts_with('#') || rest.starts_with("//")) {
                    return None;
                }
                break;
            }
            '\\' => match chars.next()?.1 {
                'b' => out.push(8),
                'f' => out.push(12),
                'v' => out.push(11),
                'n' => out.push(b'\n'),
                'r' => out.push(b'\r'),
                't' => out.push(b'\t'),
                digit @ '0'..='7' => {
                    let mut value = digit as u8 - b'0';
                    for _ in 0..2 {
                        if let Some(next @ '0'..='7') = chars.peek().map(|&(_, c)| c) {
                            chars.next();
                            value = value.wrapping_mul(8).wrapping_add(next as u8 - b'0');
                        } else {
                            break;
                        }
                    }
                    out.push(value);
                }
                'x' => {
                    let mut value = 0u8;
                    let mut digits = 0;
                    while let Some(digit) = chars.peek().and_then(|&(_, c)| c.to_digit(16)) {
                        chars.next();
                        value = value.wrapping_mul(16).wrapping_add(digit as u8);
                        digits += 1;
                    }
                    if digits == 0 {
                        return None;
                    }
                    out.push(value);
                }
                '"' => out.push(b'"'),
                '\\' => out.push(b'\\'),
                c => out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
            },
            c => out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
        }
    }
    if null_terminate {
        out.push(0);
    }
    Some(out)
}

fn split_fields(text: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut start = None;
    let mut in_string = false;
    let mut escaped = false;
    for (index, c) in text.char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
            start.get_or_insert(index);
        } else if c.is_whitespace() {
            if let Some(begin) = start.take() {
                fields.push(&text[begin..index]);
            }
        } else {
            start.get_or_insert(index);
        }
    }
    if let Some(begin) = start {
        fields.push(&text[begin..]);
    }
    fields
}

const MACHO_PURE_OR_SOME_INSTRUCTIONS: u32 = 0x8000_0000 | 0x0000_0400;

/// The gas name and Mach-O type of a `segment,section` pair that the assembler also knows by name.
fn macho_builtin_section(segment: &str, section: &str) -> Option<(&'static str, MachOSectionType)> {
    let regular = |attributes| MachOSectionType {
        kind: 0,
        attributes,
    };
    Some(match (segment, section) {
        ("__TEXT", "__text") => (".text", regular(MACHO_PURE_OR_SOME_INSTRUCTIONS)),
        ("__TEXT", "__const") => (".rodata", regular(0)),
        ("__DATA", "__data") => (".data", regular(0)),
        ("__DATA", "__bss") => (
            ".bss",
            MachOSectionType {
                kind: 0x1,
                attributes: 0,
            },
        ),
        _ => return None,
    })
}

/// The Mach-O type and attributes named by the optional fields after `segment,section`.
fn macho_section_type(fields: &[String]) -> Result<MachOSectionType, RasError> {
    if fields.len() > 2 {
        return Err(RasError::ParseError(
            "Mach-O .section takes at most a type and attributes".into(),
        ));
    }
    let kind = match fields.first().map_or("regular", String::as_str) {
        "regular" => 0x0,
        "zerofill" => 0x1,
        "thread_local_regular" => 0x11,
        "thread_local_zerofill" => 0x12,
        "thread_local_variables" => 0x13,
        "cstring_literals" => 0x2,
        "4byte_literals" => 0x3,
        "8byte_literals" => 0x4,
        "literal_pointers" => 0x5,
        "mod_init_funcs" => 0x9,
        "mod_term_funcs" => 0xa,
        "coalesced" => 0xb,
        "16byte_literals" => 0xe,
        other => {
            return Err(RasError::ParseError(format!(
                "unsupported Mach-O section type {other}"
            )));
        }
    };
    let mut attributes = 0;
    for attribute in fields
        .get(1)
        .map_or("", String::as_str)
        .split('+')
        .filter(|a| !a.is_empty())
    {
        attributes |= match attribute {
            "pure_instructions" => 0x8000_0000,
            "some_instructions" => 0x0000_0400,
            "debug" => 0x0200_0000,
            other => {
                return Err(RasError::ParseError(format!(
                    "unsupported Mach-O section attribute {other}"
                )));
            }
        };
    }
    Ok(MachOSectionType { kind, attributes })
}

/// COFF `.section` flags as gas and LLVM read them: `x` is code, `x` or `r` make the
/// section read-only unless `w` is also given, and `D` or a `.debug` name makes it
/// discardable. `a` has no meaning on COFF, and thread-local data is marked by the
/// `.tls$` name, not a flag.
fn coff_section_flags(name: &str, flags: &str) -> SectionFlags {
    let exec = flags.contains('x');
    SectionFlags {
        alloc: !flags.contains('D') && !name.starts_with(".debug"),
        exec,
        write: flags.contains('w') || !(exec || flags.contains('r')),
        tls: false,
    }
}

fn split_operands(text: &str) -> Vec<String> {
    let mut operands = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                operands.push(text[start..index].trim().to_string());
                start = index + 1;
            }
            _ => {}
        }
    }
    operands.push(text[start..].trim().to_string());
    operands
}

pub(crate) fn leb128(mut value: i128, signed: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value = if signed {
            value >> 7
        } else {
            ((value as u128) >> 7) as i128
        };
        let done = if signed {
            (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0)
        } else {
            value == 0
        };
        bytes.push(byte | if done { 0 } else { 0x80 });
        if done {
            return bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> ParsedAssembly {
        AssemblyParser::new().parse(src).expect("parse failed")
    }

    fn string_bytes(src: &str) -> Vec<u8> {
        let asm = parse(src);
        match &asm.lines[..] {
            [Line::Data(bytes)] => bytes.clone(),
            other => panic!("expected one data line, got {other:?}"),
        }
    }

    #[test]
    fn string_directives_decode_escapes_and_utf8() {
        assert_eq!(string_bytes(".ascii \"\\x41B\"\n"), b"\x1b");
        assert_eq!(string_bytes(".ascii \"\\101\\\"\"\n"), b"A\"");
        assert_eq!(string_bytes(".ascii \"\u{e9}\"\n"), "\u{e9}".as_bytes());
        assert_eq!(string_bytes(".asciz \"ok\" # note\n"), b"ok\0");
    }

    #[test]
    fn trailing_comments_end_at_the_marker_outside_strings() {
        let asm = AssemblyParser::new()
            .with_line_comments(&["#"])
            .parse("call bar # note\n.ascii \"a#b\" # tail\n")
            .expect("parse failed");
        assert!(matches!(&asm.lines[0], Line::Instruction(inst)
            if inst.opcode == "call" && inst.operands == ["bar"]));
        assert!(matches!(&asm.lines[1], Line::Data(bytes) if bytes == b"a#b"));
    }

    #[test]
    fn malformed_string_directives_are_rejected() {
        for src in [
            ".asciz hello\n",
            ".ascii \"open\n",
            ".ascii \"\\x\"\n",
            ".asciz \"a\" junk\n",
        ] {
            assert!(
                AssemblyParser::new().parse(src).is_err(),
                "{src:?} was accepted"
            );
        }
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let asm = parse("");
        assert!(asm.lines.is_empty());
        assert!(asm.symbols.is_empty());
        assert!(asm.instructions.is_empty());
    }

    fn align_directives(parsed: &ParsedAssembly) -> Vec<&str> {
        parsed
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Align { directive, .. } => Some(directive.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn darwin_reads_comm_and_align_operands_as_exponents() {
        let src = ".comm x,8,8\n.comm y,8,3\n.comm z,8\n.align 4\n.balign 4\n";
        let darwin = AssemblyParser::new()
            .with_darwin_directives(true)
            .parse(src)
            .expect("parse");
        assert_eq!(darwin.attributes["x"].common, Some((8, 256)));
        assert_eq!(darwin.attributes["y"].common, Some((8, 8)));
        assert_eq!(darwin.attributes["z"].common, Some((8, 1)));
        assert_eq!(align_directives(&darwin), [".p2align", ".balign"]);
    }

    #[test]
    fn elf_reads_comm_and_align_operands_as_byte_counts() {
        let src = ".comm x,8,8\n.comm z,8\n.align 4\n.balign 4\n";
        let elf = parse(src);
        assert_eq!(elf.attributes["x"].common, Some((8, 8)));
        assert_eq!(elf.attributes["z"].common, Some((8, 8)));
        assert_eq!(align_directives(&elf), [".align", ".balign"]);
        assert!(AssemblyParser::new().parse(".comm y,8,3\n").is_err());
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let asm = parse("# this is a comment\n// also a comment\n\n");
        assert!(asm.lines.is_empty());
    }

    #[test]
    fn label_is_parsed_and_recorded_in_symbols() {
        let asm = parse("main:");
        assert_eq!(asm.symbols.len(), 1);
        assert_eq!(asm.symbols[0].name, "main");
        assert!(!asm.symbols[0].global);
        assert!(matches!(&asm.lines[0], Line::Label(s) if s.name == "main"));
    }

    #[test]
    fn global_directive_marks_following_label_as_global() {
        let asm = parse(".globl main\nmain:");
        assert_eq!(asm.symbols.len(), 1);
        assert!(asm.symbols[0].global, "symbol should be global");
    }

    #[test]
    fn global_without_label_does_not_panic() {
        let asm = parse(".global _foo");
        assert!(asm.symbols.is_empty(), "no label defined yet");
    }

    #[test]
    fn global_missing_name_returns_error() {
        let result = AssemblyParser::new().parse(".global");
        assert!(result.is_err());
    }

    #[test]
    fn instruction_with_no_operands() {
        let asm = parse("ret");
        assert_eq!(asm.instructions.len(), 1);
        assert_eq!(asm.instructions[0].opcode, "ret");
        assert!(asm.instructions[0].operands.is_empty());
    }

    #[test]
    fn instruction_with_two_operands() {
        let asm = parse("mov rax, rbx");
        assert_eq!(asm.instructions[0].opcode, "mov");
        assert_eq!(asm.instructions[0].operands, &["rax", "rbx"]);
    }

    #[test]
    fn instruction_with_three_operands() {
        let asm = parse("add x0, x1, x2");
        assert_eq!(asm.instructions[0].operands, &["x0", "x1", "x2"]);
    }

    #[test]
    fn text_and_data_directives_push_sections() {
        let asm = parse(".text\n.data");
        assert_eq!(asm.sections.len(), 2);
        assert_eq!(asm.sections[0].name, ".text");
        assert!(asm.sections[0].flags.exec);
        assert!(!asm.sections[0].flags.write);
        assert_eq!(asm.sections[1].name, ".data");
        assert!(asm.sections[1].flags.write);
        assert!(!asm.sections[1].flags.exec);
    }

    #[test]
    fn macho_segment_section_pairs_map_to_gas_sections() {
        let asm =
            parse(".section __DATA,__data\nvar:\n.section __TEXT,__const\n.section __DATA,__bss");
        assert_eq!(asm.sections[0].name, ".data");
        assert!(asm.sections[0].flags.write);
        assert_eq!(asm.symbols[0].section, ".data");
        assert_eq!(asm.sections[1].name, ".rodata");
        assert_eq!(asm.sections[2].name, ".bss");
    }

    #[test]
    fn label_section_tracks_current_section() {
        let asm = parse(".data\nvar:");
        assert_eq!(asm.symbols[0].section, ".data");
    }

    #[test]
    fn symbol_metadata_directives_are_recorded() {
        let asm = parse(".size foo, 4\n.type foo, @function");
        assert!(asm.lines.is_empty());
        assert_eq!(asm.attributes["foo"].symbol_type, 2);
        assert_eq!(asm.attributes["foo"].size.as_deref(), Some("4"));
    }

    #[test]
    fn mixed_labels_and_instructions_preserve_order() {
        let asm = parse("foo:\n  mov rax, 0\n  ret");
        assert_eq!(asm.lines.len(), 3);
        assert!(matches!(&asm.lines[0], Line::Label(_)));
        assert!(matches!(&asm.lines[1], Line::Instruction(_)));
        assert!(matches!(&asm.lines[2], Line::Instruction(_)));
    }

    #[test]
    fn multiple_globals_on_one_directive() {
        // .global accepts a list; each name should be queued.
        let asm = parse(".global a b\na:\nb:");
        assert!(asm.symbols.iter().all(|s| s.global));
        assert_eq!(asm.symbols.len(), 2);
    }

    #[test]
    fn inline_label_with_asciz_emits_label_and_data() {
        let asm = parse(".L_fmt: .asciz \"%lld\\n\"");
        assert_eq!(asm.symbols.len(), 1);
        assert_eq!(asm.symbols[0].name, ".L_fmt");
        assert!(matches!(&asm.lines[0], Line::Label(s) if s.name == ".L_fmt"));
        let data_line = &asm.lines[1];
        match data_line {
            Line::Data(bytes) => {
                assert_eq!(bytes, b"%lld\n\0");
            }
            _ => panic!("expected Line::Data, got {:?}", data_line),
        }
    }

    #[test]
    fn asciz_standalone_emits_data() {
        let asm = parse(".asciz \"hello\"");
        assert_eq!(asm.lines.len(), 1);
        match &asm.lines[0] {
            Line::Data(bytes) => assert_eq!(bytes, b"hello\0"),
            _ => panic!("expected Line::Data"),
        }
    }

    #[test]
    fn file_and_loc_directives_record_rows_and_sticky_is_stmt() {
        let asm = parse(
            ".file 1 \"a.c\"\n.file 2 \"dir\" \"b.c\" md5 0x0123\n.loc 1 3 4\n.loc 2 5 0 is_stmt 0 discriminator 1\nnop\n.loc 2 6 1\nnop\n",
        );
        assert_eq!(asm.files[&1].name, "a.c");
        assert_eq!(asm.files[&2].directory.as_deref(), Some("dir"));
        assert_eq!(asm.files[&2].name, "b.c");
        let rows: Vec<_> = asm
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Loc {
                    file,
                    line,
                    column,
                    is_stmt,
                    ..
                } => Some((*file, *line, *column, *is_stmt)),
                _ => None,
            })
            .collect();
        assert_eq!(rows, [(1, 3, 4, true), (2, 5, 0, false), (2, 6, 1, false)]);
    }

    #[test]
    fn file_directive_without_number_is_ignored() {
        assert!(parse(".file \"x.c\"\n").files.is_empty());
    }

    #[test]
    fn loc_rejects_malformed_operands() {
        assert!(AssemblyParser::new().parse(".loc 1\n").is_err());
        assert!(
            AssemblyParser::new()
                .parse(".loc 1 2 3 is_stmt 2\n")
                .is_err()
        );
    }

    #[test]
    fn loc_flags_are_per_row_and_isa_is_sticky() {
        let asm = parse(
            ".file 1 \"a.c\"\n.loc 1 1 0 prologue_end isa 2 discriminator 7\nnop\n.loc 1 2 0 epilogue_begin basic_block\nnop\n",
        );
        let rows: Vec<_> = asm
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Loc {
                    prologue_end,
                    epilogue_begin,
                    basic_block,
                    discriminator,
                    isa,
                    ..
                } => Some((
                    *prologue_end,
                    *epilogue_begin,
                    *basic_block,
                    *discriminator,
                    *isa,
                )),
                _ => None,
            })
            .collect();
        assert_eq!(
            rows,
            [(true, false, false, 7, 2), (false, true, true, 0, 2),]
        );
    }

    #[test]
    fn loc_names_options_it_does_not_implement() {
        let view = match AssemblyParser::new().parse(".loc 1 1 0 view .LVU0\n") {
            Err(error) => error.to_string(),
            Ok(_) => panic!("expected .loc view to be refused"),
        };
        assert!(view.contains("view"), "{view}");
        let unknown = match AssemblyParser::new().parse(".loc 1 1 0 atom_group 3\n") {
            Err(error) => error.to_string(),
            Ok(_) => panic!("expected an unknown .loc option to be refused"),
        };
        assert!(unknown.contains("atom_group"), "{unknown}");
        assert!(
            AssemblyParser::new()
                .parse(".loc 1 1 0 discriminator x\n")
                .is_err()
        );
    }

    fn macho_parse(src: &str) -> Result<ParsedAssembly, RasError> {
        AssemblyParser::new().parse(src)
    }

    #[test]
    fn macho_custom_sections_take_segment_section_name_and_type() {
        let asm = parse(
            ".section __TEXT,__custom\n.section __DATA,__thread_vars,thread_local_variables\n.section __DATA,__thread_data,thread_local_regular\n.section __DATA,__thread_bss,thread_local_zerofill\n.section __DATA,__x,regular,pure_instructions",
        );
        let find = |name: &str| {
            asm.sections
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("section {name} missing"))
        };
        assert_eq!(find("__TEXT,__custom").macho.unwrap().kind, 0);
        assert_eq!(find("__DATA,__thread_vars").macho.unwrap().kind, 0x13);
        assert_eq!(find("__DATA,__thread_data").macho.unwrap().kind, 0x11);
        assert_eq!(find("__DATA,__thread_bss").macho.unwrap().kind, 0x12);
        let custom = find("__DATA,__x").macho.unwrap();
        assert_eq!(custom.attributes, 0x8000_0000);
        assert!(find("__DATA,__x").flags.write);
        assert!(find("__DATA,__x").flags.exec);
    }

    #[test]
    fn macho_builtin_pairs_accept_their_own_type_and_keep_gas_names() {
        let asm = parse(
            ".section __TEXT,__text,regular,pure_instructions\n.section __DATA,__bss,zerofill\n.section __DATA,__data,regular",
        );
        assert_eq!(asm.sections[0].name, ".text");
        assert_eq!(asm.sections[1].name, ".bss");
        assert_eq!(asm.sections[2].name, ".data");
        assert!(asm.sections.iter().all(|s| s.macho.is_none()));
    }

    #[test]
    fn macho_section_types_are_checked() {
        let err = |src: &str| match macho_parse(src) {
            Ok(_) => panic!("accepted {src}"),
            Err(e) => e.to_string(),
        };
        assert!(err(".section __DATA,__bss,regular").contains("built in"));
        assert!(
            err(".section __TEXT,__text,regular,some_instructions+live_support")
                .contains("attribute")
        );
        assert_eq!(
            macho_parse(".section __LD,__compact_unwind,regular,debug")
                .expect("compact unwind section")
                .sections
                .iter()
                .find(|s| s.name == "__LD,__compact_unwind")
                .and_then(|s| s.macho)
                .map(|m| m.attributes),
            Some(0x0200_0000),
            "S_ATTR_DEBUG"
        );
        assert!(err(".section __DATA,__x,bogus").contains("section type bogus"));
        assert!(err(".section __DATA,__x\n.section __DATA,__x,zerofill").contains("redeclared"));
        assert!(macho_parse(".section __DATA,__x\n.section __DATA,__x").is_ok());
    }
}
