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
            ".section" | ".align" | ".balign" | ".p2align" | ".byte" | ".short"
            | ".int" | ".long" | ".quad" | ".ascii" | ".asciz" | ".string"
            | ".zero" | ".space" | ".skip" | ".cfi_startproc" | ".cfi_endproc"
            | ".cfi_def_cfa" | ".size" | ".type" | ".ident" | ".file" => {}
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
