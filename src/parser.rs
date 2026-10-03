//! Assembly text parser
//!
//! Parses gas-compatible assembly: .text, .data, .global/.globl, directives,
//! labels, and instructions. Outputs `lines` preserving label/instruction order
//! for two-pass assembly (jmp/call with label resolution).

use crate::{encoder::traits::ParsedInstruction, error::RasError};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Section {
    pub group: Option<String>,
    pub name: String,
    pub flags: SectionFlags,
    pub data: Vec<u8>,
    pub alignment: usize,
}

#[derive(Debug, Clone)]
pub struct SectionFlags {
    pub alloc: bool,
    pub exec: bool,
    pub write: bool,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub global: bool,
    pub section: String,
}

#[derive(Debug, Clone)]
pub enum Line {
    Section(String),
    Label(Symbol),
    Instruction(ParsedInstruction),
    Data(Vec<u8>),
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
    pub weak: bool,
    pub hidden: bool,
    pub local: bool,
    pub common: Option<(u64, u64)>,
}

pub struct ParsedAssembly {
    pub common_order: Vec<String>,
    pub aliases: Vec<(String, String)>,
    pub attributes: HashMap<String, SymbolAttributes>,
    pub sections: Vec<Section>,
    pub symbols: Vec<Symbol>,
    pub instructions: Vec<ParsedInstruction>,
    pub lines: Vec<Line>,
}

pub struct AssemblyParser {
    common_order: Vec<String>,
    aliases: Vec<(String, String)>,
    current_section: String,
    sections: Vec<Section>,
    symbols: Vec<Symbol>,
    instructions: Vec<ParsedInstruction>,
    lines: Vec<Line>,
    attributes: HashMap<String, SymbolAttributes>,
    pending_globals: HashSet<String>,
    symbol_names: HashSet<String>,
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
            aliases: Vec::new(),
            current_section: ".text".to_string(),
            sections: Vec::new(),
            symbols: Vec::new(),
            instructions: Vec::new(),
            lines: Vec::new(),
            attributes: HashMap::new(),
            pending_globals: HashSet::new(),
            symbol_names: HashSet::new(),
        }
    }

    pub fn parse(&mut self, text: &str) -> Result<ParsedAssembly, RasError> {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }

            // Handle "label:" and "label: directive/instruction" forms.
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

            if line.starts_with('.') {
                self.parse_directive(line)?;
            } else {
                self.parse_instruction(line)?;
            }
        }

        self.resolve_numeric_labels()?;

        Ok(ParsedAssembly {
            common_order: self.common_order.clone(),
            aliases: self.aliases.clone(),
            attributes: self.attributes.clone(),
            sections: self.sections.clone(),
            symbols: self.symbols.clone(),
            instructions: self.instructions.clone(),
            lines: self.lines.clone(),
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
        for (index, line) in self.lines.iter_mut().enumerate() {
            match line {
                Line::Label(symbol) => {
                    if let Some((_, _, name)) = definitions.iter().find(|d| d.0 == index) {
                        symbol.name = name.clone();
                    }
                }
                Line::Instruction(inst) => {
                    for operand in &mut inst.operands {
                        *operand = resolve_numeric_references(operand, index, &definitions)?;
                    }
                }
                Line::DataExpression { expression, .. } | Line::Leb128 { expression, .. } => {
                    *expression = resolve_numeric_references(expression, index, &definitions)?;
                }
                _ => {}
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
        self.instructions = self
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Instruction(inst) => Some(inst.clone()),
                _ => None,
            })
            .collect();
        Ok(())
    }

    fn parse_directive(&mut self, line: &str) -> Result<(), RasError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        match parts[0] {
            ".text" | ".data" | ".bss" | ".section" => {
                let name = if parts[0] == ".section" {
                    parts
                        .get(1)
                        .ok_or_else(|| RasError::ParseError(".section requires a name".into()))?
                        .split(',')
                        .next()
                        .unwrap()
                } else {
                    parts[0]
                };
                self.current_section = name.to_string();
                self.lines.push(Line::Section(name.to_string()));
                if !self.sections.iter().any(|s| s.name == name) {
                    let explicit = line.split('"').nth(1);
                    self.sections.push(Section {
                        group: if explicit.is_some_and(|f| f.contains('G')) {
                            let fields: Vec<_> = line.split(',').map(str::trim).collect();
                            if fields.len() != 5 || fields[4] != "comdat" || fields[3].is_empty() {
                                return Err(RasError::ParseError(
                                    "Unsupported section group".into(),
                                ));
                            }
                            Some(fields[3].into())
                        } else {
                            None
                        },
                        name: name.to_string(),
                        flags: SectionFlags {
                            alloc: explicit.map_or(
                                !name.starts_with(".debug") && name != ".note.GNU-stack",
                                |f| f.contains('a'),
                            ),
                            exec: explicit.map_or(name == ".text", |f| f.contains('x')),
                            write: explicit
                                .map_or(name == ".data" || name == ".bss", |f| f.contains('w')),
                        },
                        data: Vec::new(),
                        alignment: 1,
                    });
                }
            }
            ".set" => {
                let (name, expression) =
                    line[parts[0].len()..].split_once(',').ok_or_else(|| {
                        RasError::ParseError(".set requires symbol and expression".into())
                    })?;
                parse_data_expression(expression.trim())?;
                self.aliases
                    .push((name.trim().into(), expression.trim().into()));
            }
            ".comm" => {
                let fields: Vec<_> = line[parts[0].len()..].split(',').map(str::trim).collect();
                if !(2..=3).contains(&fields.len()) || fields[0].is_empty() {
                    return Err(RasError::ParseError("Invalid .comm operands".into()));
                }
                let size = u64::try_from(parse_integer(fields[1])?)
                    .map_err(|_| RasError::ParseError("Invalid common size".into()))?;
                let alignment = if let Some(value) = fields.get(2) {
                    u64::try_from(parse_integer(value)?)
                        .map_err(|_| RasError::ParseError("Invalid common alignment".into()))?
                } else {
                    size.next_power_of_two().min(16)
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
            }
            ".local" => {
                for name in line[parts[0].len()..].split(',').map(str::trim) {
                    if name.is_empty() {
                        return Err(RasError::ParseError(".local requires symbol".into()));
                    }
                    self.attributes.entry(name.into()).or_default().local = true;
                }
            }
            ".hidden" => {
                for name in line[parts[0].len()..].split(',').map(str::trim) {
                    if name.is_empty() {
                        return Err(RasError::ParseError(".hidden requires symbol".into()));
                    }
                    self.attributes.entry(name.into()).or_default().hidden = true;
                }
            }
            ".weak" => {
                for name in line[parts[0].len()..].split(',').map(str::trim) {
                    if name.is_empty() {
                        return Err(RasError::ParseError(".weak requires symbol".into()));
                    }
                    self.attributes.entry(name.into()).or_default().weak = true;
                }
            }
            ".global" | ".globl" => {
                if parts.len() < 2 {
                    return Err(RasError::ParseError(
                        ".global requires a symbol name".to_string(),
                    ));
                }
                for name in &parts[1..] {
                    self.pending_globals.insert((*name).to_string());
                }
            }
            ".asciz" | ".string" => {
                let rest = line[parts[0].len()..].trim();
                if let Some(bytes) = parse_quoted_string_bytes(rest, true) {
                    self.lines.push(Line::Data(bytes));
                }
            }
            ".ascii" => {
                let rest = line[parts[0].len()..].trim();
                if let Some(bytes) = parse_quoted_string_bytes(rest, false) {
                    self.lines.push(Line::Data(bytes));
                }
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
            ".byte" | ".short" | ".hword" | ".int" | ".long" | ".quad" | ".xword" => {
                let width = match parts[0] {
                    ".byte" => 1,
                    ".short" | ".hword" => 2,
                    ".quad" | ".xword" => 8,
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
                    .transpose()?;
                self.lines.push(Line::Align {
                    value,
                    directive: parts[0].into(),
                    fill,
                    max_skip,
                });
            }
            ".size" | ".type" | ".ident" | ".file" => {}
            _ => {
                return Err(RasError::ParseError(format!(
                    "Unsupported directive: {}",
                    parts[0]
                )));
            }
        }

        Ok(())
    }

    fn parse_instruction(&mut self, line: &str) -> Result<(), RasError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        let opcode = parts[0].to_string();
        let operands: Vec<String> = if parts.len() > 1 {
            split_operands(&parts[1..].join(" "))
        } else {
            Vec::new()
        };

        let inst = ParsedInstruction { opcode, operands };
        self.instructions.push(inst.clone());
        self.lines.push(Line::Instruction(inst));
        Ok(())
    }
}

fn is_numeric_label(name: &str) -> bool {
    name.len() == 1 && name.as_bytes()[0].is_ascii_digit()
}

fn resolve_numeric_references(
    text: &str,
    position: usize,
    definitions: &[(usize, String, String)],
) -> Result<String, RasError> {
    let mut result = String::new();
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
                result.push_str(&definition.2);
            } else {
                result.push_str(token);
            }
        }
        if end < text.len() {
            result.push(ch);
        }
        start = end + ch.len_utf8();
    }
    Ok(result)
}

pub(crate) enum DataExpression {
    Constant(i128),
    Symbol(String, i128),
    Difference(String, String),
}

pub(crate) fn parse_data_expression(text: &str) -> Result<DataExpression, RasError> {
    if let Ok(value) = parse_integer(text) {
        return Ok(DataExpression::Constant(value));
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
    if let Some((index, op)) = text.char_indices().find(|(_, c)| matches!(c, '+' | '-')) {
        let left = text[..index].trim();
        let right = text[index + 1..].trim();
        if operand(left) {
            if let Ok(value) = parse_integer(right) {
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
    let (sign, digits) = text.strip_prefix('-').map_or((1, text), |s| (-1, s));
    let result = if let Some(hex) = digits.strip_prefix("0x") {
        i128::from_str_radix(hex, 16)
    } else {
        digits.parse()
    };
    result
        .map(|n| sign * n)
        .map_err(|_| RasError::ParseError(format!("Invalid integer: {text}")))
}

fn parse_quoted_string_bytes(s: &str, null_terminate: bool) -> Option<Vec<u8>> {
    let s = s.trim();
    let s = s.strip_prefix('"')?.strip_suffix('"')?;
    let mut out = Vec::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                'n' => out.push(b'\n'),
                'r' => out.push(b'\r'),
                't' => out.push(b'\t'),
                '0' => out.push(0),
                '"' => out.push(b'"'),
                '\\' => out.push(b'\\'),
                c => out.push(c as u8),
            }
        } else {
            out.push(c as u8);
        }
    }
    if null_terminate {
        out.push(0);
    }
    Some(out)
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
        value >>= 7;
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

    #[test]
    fn empty_input_produces_empty_output() {
        let asm = parse("");
        assert!(asm.lines.is_empty());
        assert!(asm.symbols.is_empty());
        assert!(asm.instructions.is_empty());
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
    fn label_section_tracks_current_section() {
        let asm = parse(".data\nvar:");
        assert_eq!(asm.symbols[0].section, ".data");
    }

    #[test]
    fn known_no_op_directives_are_silently_ignored() {
        let asm = parse(".size foo, 4\n.type foo, @function");
        assert!(asm.lines.is_empty());
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
}
