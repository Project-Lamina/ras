//! Assembly text parser
//!
//! Parses gas-compatible assembly: .text, .data, .global/.globl, directives,
//! labels, and instructions. Outputs `lines` preserving label/instruction order
//! for two-pass assembly (jmp/call with label resolution).

use crate::encoder::traits::ParsedInstruction;
use crate::error::RasError;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Section {
    pub name: String,
    pub flags: SectionFlags,
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
    Label(Symbol),
    Instruction(ParsedInstruction),
}

pub struct ParsedAssembly {
    pub sections: Vec<Section>,
    pub symbols: Vec<Symbol>,
    pub instructions: Vec<ParsedInstruction>,
    pub lines: Vec<Line>,
}

pub struct AssemblyParser {
    current_section: String,
    sections: Vec<Section>,
    symbols: Vec<Symbol>,
    instructions: Vec<ParsedInstruction>,
    lines: Vec<Line>,
    pending_globals: HashSet<String>,
}

impl Default for AssemblyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AssemblyParser {
    pub fn new() -> Self {
        Self {
            current_section: ".text".to_string(),
            sections: Vec::new(),
            symbols: Vec::new(),
            instructions: Vec::new(),
            lines: Vec::new(),
            pending_globals: HashSet::new(),
        }
    }

    pub fn parse(&mut self, text: &str) -> Result<ParsedAssembly, RasError> {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }

            if line.ends_with(':') {
                let label = line.trim_end_matches(':').trim();
                let global = self.pending_globals.remove(label);
                let sym = Symbol {
                    name: label.to_string(),
                    global,
                    section: self.current_section.clone(),
                };
                self.symbols.push(sym.clone());
                self.lines.push(Line::Label(sym));
            } else if line.starts_with('.') {
                self.parse_directive(line)?;
            } else {
                self.parse_instruction(line)?;
            }
        }

        Ok(ParsedAssembly {
            sections: self.sections.clone(),
            symbols: self.symbols.clone(),
            instructions: self.instructions.clone(),
            lines: self.lines.clone(),
        })
    }

    fn parse_directive(&mut self, line: &str) -> Result<(), RasError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        match parts[0] {
            ".text" => {
                self.current_section = ".text".to_string();
                self.sections.push(Section {
                    name: ".text".to_string(),
                    flags: SectionFlags {
                        alloc: true,
                        exec: true,
                        write: false,
                    },
                });
            }
            ".data" => {
                self.current_section = ".data".to_string();
                self.sections.push(Section {
                    name: ".data".to_string(),
                    flags: SectionFlags {
                        alloc: true,
                        exec: false,
                        write: true,
                    },
                });
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
            ".section" | ".align" | ".balign" | ".p2align" | ".byte" | ".short" | ".int"
            | ".long" | ".quad" | ".ascii" | ".asciz" | ".string" | ".zero" | ".space"
            | ".skip" | ".cfi_startproc" | ".cfi_endproc" | ".cfi_def_cfa" | ".size" | ".type"
            | ".ident" | ".file" => {}
            _ => {}
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
            parts[1..]
                .join(" ")
                .split(',')
                .map(|s| s.trim().to_string())
                .collect()
        } else {
            Vec::new()
        };

        let inst = ParsedInstruction { opcode, operands };
        self.instructions.push(inst.clone());
        self.lines.push(Line::Instruction(inst));
        Ok(())
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
        let asm = parse(".align 16\n.size foo, 4\n.type foo, @function");
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
}
