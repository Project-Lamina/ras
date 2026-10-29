//! DWARF 4 line table and compile unit built from `.file` and `.loc` rows.
//!
//! Every address-sized field that names a section is emitted as a relocation
//! against that section's symbol, so the linker can rebase the debug sections.

use crate::{
    error::RasError,
    object::ExternalReloc,
    parser::{DebugFile, Section, SectionFlags},
};
use lamina_platform::TargetArchitecture;
use std::collections::BTreeMap;

const DW_TAG_COMPILE_UNIT: u8 = 0x11;
const DW_AT_NAME: u8 = 0x03;
const DW_AT_STMT_LIST: u8 = 0x10;
const DW_AT_LOW_PC: u8 = 0x11;
const DW_AT_HIGH_PC: u8 = 0x12;
const DW_AT_RANGES: u8 = 0x55;
const DW_AT_COMP_DIR: u8 = 0x1b;
const DW_FORM_ADDR: u8 = 0x01;
const DW_FORM_DATA8: u8 = 0x07;
const DW_FORM_STRING: u8 = 0x08;
const DW_FORM_SEC_OFFSET: u8 = 0x17;

const DW_LNS_COPY: u8 = 1;
const DW_LNS_ADVANCE_PC: u8 = 2;
const DW_LNS_ADVANCE_LINE: u8 = 3;
const DW_LNS_SET_FILE: u8 = 4;
const DW_LNS_SET_COLUMN: u8 = 5;
const DW_LNS_NEGATE_STMT: u8 = 6;
const DW_LNE_END_SEQUENCE: u8 = 1;
const DW_LNE_SET_ADDRESS: u8 = 2;

const LINE_BASE: i8 = -5;
const LINE_RANGE: u8 = 14;
const OPCODE_BASE: u8 = 13;
const STANDARD_OPCODE_LENGTHS: [u8; 12] = [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1];

pub(crate) struct LocRow {
    pub section: String,
    pub offset: u64,
    pub file: u32,
    pub line: u32,
    pub column: u32,
    pub is_stmt: bool,
}

pub(crate) struct DebugSections {
    pub sections: Vec<Section>,
    pub relocs: Vec<ExternalReloc>,
}

fn absolute_kinds(arch: TargetArchitecture) -> Option<(u32, u32)> {
    match arch {
        TargetArchitecture::X86_64 => Some((1, 10)),
        TargetArchitecture::Aarch64 => Some((257, 258)),
        TargetArchitecture::Riscv64 => Some((2, 1)),
        _ => None,
    }
}

pub(crate) fn build(
    arch: TargetArchitecture,
    sections: &[Section],
    rows: &[LocRow],
    files: &BTreeMap<u32, DebugFile>,
    fallback_name: &str,
    always_emit: bool,
) -> Result<Option<DebugSections>, RasError> {
    let Some((abs64, abs32)) = absolute_kinds(arch) else {
        return Ok(None);
    };
    if rows.is_empty() && !always_emit {
        return Ok(None);
    }
    if sections.iter().any(|s| {
        matches!(
            s.name.as_str(),
            ".debug_line" | ".debug_info" | ".debug_abbrev"
        )
    }) {
        return Err(RasError::ObjectError(
            "source defines .debug_* sections; ras cannot add .loc DWARF beside them".into(),
        ));
    }
    let text = sections
        .iter()
        .find(|s| s.name == ".text")
        .ok_or_else(|| RasError::ObjectError("no .text section for DWARF".into()))?;

    let numbers: Vec<u32> = files.keys().copied().collect();
    let file_index = |number: u32| {
        numbers
            .binary_search(&number)
            .map(|position| position as u64 + 1)
            .map_err(|_| RasError::ObjectError(format!("undefined .file {number}")))
    };
    let mut directories: Vec<&str> = Vec::new();
    let mut file_entries: Vec<(&str, u64)> = Vec::new();
    for file in numbers.iter().map(|number| &files[number]) {
        let directory = match file.directory.as_deref() {
            None => 0,
            Some(name) => {
                let index = directories
                    .iter()
                    .position(|d| *d == name)
                    .unwrap_or_else(|| {
                        directories.push(name);
                        directories.len() - 1
                    });
                index as u64 + 1
            }
        };
        file_entries.push((file.name.as_str(), directory));
    }

    let mut line_header = vec![1, 1, 1];
    line_header.push(LINE_BASE as u8);
    line_header.push(LINE_RANGE);
    line_header.push(OPCODE_BASE);
    line_header.extend_from_slice(&STANDARD_OPCODE_LENGTHS);
    for directory in &directories {
        push_cstr(&mut line_header, directory);
    }
    line_header.push(0);
    for (name, directory) in &file_entries {
        push_cstr(&mut line_header, name);
        push_uleb(&mut line_header, *directory);
        push_uleb(&mut line_header, 0);
        push_uleb(&mut line_header, 0);
    }
    line_header.push(0);

    let mut program = Vec::new();
    let mut line_relocs = Vec::new();
    let row_sections: Vec<&Section> = sections
        .iter()
        .filter(|section| rows.iter().any(|row| row.section == section.name))
        .collect();
    for section in &row_sections {
        let mut section_rows: Vec<&LocRow> = rows
            .iter()
            .filter(|row| row.section == section.name)
            .collect();
        section_rows.sort_by_key(|row| row.offset);
        let mut picked: Vec<&LocRow> = Vec::with_capacity(section_rows.len());
        for row in section_rows {
            if picked.last().is_some_and(|last| last.offset == row.offset) {
                picked.pop();
            }
            picked.push(row);
        }
        program.push(0);
        push_uleb(&mut program, 9);
        program.push(DW_LNE_SET_ADDRESS);
        line_relocs.push((program.len(), section.name.clone(), 8, 0i64));
        program.extend_from_slice(&[0; 8]);

        let (mut file, mut line, mut column, mut is_stmt, mut address) =
            (1u64, 1u32, 0u32, true, 0u64);
        for row in picked {
            let row_file = file_index(row.file)?;
            if row_file != file {
                program.push(DW_LNS_SET_FILE);
                push_uleb(&mut program, row_file);
                file = row_file;
            }
            if row.column != column {
                program.push(DW_LNS_SET_COLUMN);
                push_uleb(&mut program, u64::from(row.column));
                column = row.column;
            }
            if row.is_stmt != is_stmt {
                program.push(DW_LNS_NEGATE_STMT);
                is_stmt = row.is_stmt;
            }
            if row.offset > address {
                program.push(DW_LNS_ADVANCE_PC);
                push_uleb(&mut program, row.offset - address);
                address = row.offset;
            }
            if row.line != line {
                program.push(DW_LNS_ADVANCE_LINE);
                push_sleb(&mut program, i64::from(row.line) - i64::from(line));
                line = row.line;
            }
            program.push(DW_LNS_COPY);
        }
        let end = section.data.len() as u64;
        if end > address {
            program.push(DW_LNS_ADVANCE_PC);
            push_uleb(&mut program, end - address);
        }
        program.push(0);
        push_uleb(&mut program, 1);
        program.push(DW_LNE_END_SEQUENCE);
    }

    let mut debug_line = Vec::new();
    debug_line.extend_from_slice(&4u16.to_le_bytes());
    let header_length = line_header.len() as u32;
    debug_line.extend_from_slice(&header_length.to_le_bytes());
    debug_line.extend_from_slice(&line_header);
    let program_start = debug_line.len();
    debug_line.extend_from_slice(&program);
    let line_unit_length = debug_line.len() as u32;
    let mut debug_line_unit = line_unit_length.to_le_bytes().to_vec();
    debug_line_unit.extend_from_slice(&debug_line);

    let mut ranges = Vec::new();
    let mut range_relocs = Vec::new();
    let ranged = row_sections.iter().any(|section| section.name != ".text");
    if ranged {
        for section in &row_sections {
            ranges.extend_from_slice(&u64::MAX.to_le_bytes());
            range_relocs.push((ranges.len(), section.name.clone(), 8, 0i64));
            ranges.extend_from_slice(&[0; 8]);
            ranges.extend_from_slice(&0u64.to_le_bytes());
            ranges.extend_from_slice(&(section.data.len() as u64).to_le_bytes());
        }
        ranges.extend_from_slice(&[0; 16]);
    }

    let primary = numbers.first().map(|number| &files[number]);
    let compile_unit_name = primary.map_or(fallback_name, |file| file.name.as_str());
    let comp_dir = primary.and_then(|file| file.directory.clone()).or_else(|| {
        (!compile_unit_name.starts_with('/'))
            .then(|| std::env::current_dir().ok())
            .flatten()
            .map(|dir| dir.display().to_string())
    });

    let mut abbrev = Vec::new();
    for code in 1u8..=4 {
        let with_comp_dir = code & 1 == 0;
        let with_ranges = code > 2;
        abbrev.push(code);
        abbrev.push(DW_TAG_COMPILE_UNIT);
        abbrev.push(0);
        let mut attributes = Vec::new();
        if with_comp_dir {
            attributes.push((DW_AT_COMP_DIR, DW_FORM_STRING));
        }
        attributes.extend([
            (DW_AT_NAME, DW_FORM_STRING),
            (DW_AT_STMT_LIST, DW_FORM_SEC_OFFSET),
            (DW_AT_LOW_PC, DW_FORM_ADDR),
        ]);
        attributes.push(if with_ranges {
            (DW_AT_RANGES, DW_FORM_SEC_OFFSET)
        } else {
            (DW_AT_HIGH_PC, DW_FORM_DATA8)
        });
        for (attribute, form) in attributes {
            abbrev.push(attribute);
            abbrev.push(form);
        }
        abbrev.extend_from_slice(&[0, 0]);
    }
    abbrev.push(0);

    let mut info = vec![0u8; 4];
    info.extend_from_slice(&4u16.to_le_bytes());
    let mut info_relocs = vec![(info.len(), ".debug_abbrev".to_string(), 4, 0i64)];
    info.extend_from_slice(&[0; 4]);
    info.push(8);
    let code = 1 + u8::from(comp_dir.is_some()) + 2 * u8::from(ranged);
    info.push(code);
    if let Some(dir) = comp_dir.as_deref() {
        push_cstr(&mut info, dir);
    }
    push_cstr(&mut info, compile_unit_name);
    info_relocs.push((info.len(), ".debug_line".to_string(), 4, 0i64));
    info.extend_from_slice(&[0; 4]);
    if ranged {
        info.extend_from_slice(&[0; 8]);
        info_relocs.push((info.len(), ".debug_ranges".to_string(), 4, 0i64));
        info.extend_from_slice(&[0; 4]);
    } else {
        info_relocs.push((info.len(), text.name.clone(), 8, 0i64));
        info.extend_from_slice(&[0; 8]);
        info.extend_from_slice(&(text.data.len() as u64).to_le_bytes());
    }
    let info_unit_length = (info.len() - 4) as u32;
    info[..4].copy_from_slice(&info_unit_length.to_le_bytes());

    let mut relocs = Vec::new();
    for (position, symbol, width, addend) in line_relocs {
        relocs.push(reloc(
            ".debug_line",
            4 + program_start + position,
            &symbol,
            width,
            addend,
            abs64,
            abs32,
        ));
    }
    for (position, symbol, width, addend) in info_relocs {
        relocs.push(reloc(
            ".debug_info",
            position,
            &symbol,
            width,
            addend,
            abs64,
            abs32,
        ));
    }
    for (position, symbol, width, addend) in range_relocs {
        relocs.push(reloc(
            ".debug_ranges",
            position,
            &symbol,
            width,
            addend,
            abs64,
            abs32,
        ));
    }
    let mut out_sections = vec![
        section(".debug_line", debug_line_unit),
        section(".debug_abbrev", abbrev),
        section(".debug_info", info),
    ];
    if ranged {
        out_sections.push(section(".debug_ranges", ranges));
    }
    Ok(Some(DebugSections {
        sections: out_sections,
        relocs,
    }))
}

fn reloc(
    section: &str,
    offset: usize,
    symbol: &str,
    width: u8,
    addend: i64,
    abs64: u32,
    abs32: u32,
) -> ExternalReloc {
    ExternalReloc {
        offset,
        symbol: symbol.to_string(),
        section: section.to_string(),
        kind: if width == 8 { abs64 } else { abs32 },
        addend,
        trailing: 0,
        branch: false,
        section_symbol: true,
    }
}

fn section(name: &str, data: Vec<u8>) -> Section {
    Section {
        group: None,
        name: name.into(),
        flags: SectionFlags {
            alloc: false,
            exec: false,
            write: false,
            tls: false,
        },
        data,
        alignment: 1,
    }
}

fn push_cstr(buf: &mut Vec<u8>, text: &str) {
    buf.extend_from_slice(text.as_bytes());
    buf.push(0);
}

fn push_uleb(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

fn push_sleb(buf: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
        if done {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}
