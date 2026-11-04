//! DWARF 4 line table and compile unit built from `.file` and `.loc` rows.
//!
//! Every address-sized field that names a section is emitted as a relocation
//! against that section's symbol, so the linker can rebase the debug sections.

pub use crate::debug::{
    BaseType, DebugInfoRequest, DebugType, DeclSite, FormalParameter, Subprogram,
};
use crate::{
    debug::{DebugSections, DebugSource, FunctionExtent, LocRow},
    error::RasError,
    object::ExternalReloc,
    parser::{MachOSectionType, Section, SectionFlags},
    target::ObjectFormat,
};
use lamina_platform::TargetArchitecture;
use std::collections::HashMap;

const DW_TAG_COMPILE_UNIT: u64 = 0x11;
const DW_TAG_FORMAL_PARAMETER: u64 = 0x05;
const DW_TAG_POINTER_TYPE: u64 = 0x0f;
const DW_TAG_BASE_TYPE: u64 = 0x24;
const DW_TAG_SUBPROGRAM: u64 = 0x2e;
const DW_TAG_UNSPECIFIED_PARAMETERS: u64 = 0x18;

const DW_AT_NAME: u64 = 0x03;
const DW_AT_BYTE_SIZE: u64 = 0x0b;
const DW_AT_STMT_LIST: u64 = 0x10;
const DW_AT_LOW_PC: u64 = 0x11;
const DW_AT_HIGH_PC: u64 = 0x12;
const DW_AT_LANGUAGE: u64 = 0x13;
const DW_AT_COMP_DIR: u64 = 0x1b;
const DW_AT_PRODUCER: u64 = 0x25;
const DW_AT_PROTOTYPED: u64 = 0x27;
const DW_AT_DECL_FILE: u64 = 0x3a;
const DW_AT_DECL_LINE: u64 = 0x3b;
const DW_AT_ENCODING: u64 = 0x3e;
const DW_AT_EXTERNAL: u64 = 0x3f;
const DW_AT_FRAME_BASE: u64 = 0x40;
const DW_AT_TYPE: u64 = 0x49;
const DW_AT_RANGES: u64 = 0x55;

const DW_FORM_ADDR: u64 = 0x01;
const DW_FORM_DATA2: u64 = 0x05;
const DW_FORM_DATA8: u64 = 0x07;
const DW_FORM_STRING: u64 = 0x08;
const DW_FORM_DATA1: u64 = 0x0b;
const DW_FORM_UDATA: u64 = 0x0f;
const DW_FORM_REF4: u64 = 0x13;
const DW_FORM_SEC_OFFSET: u64 = 0x17;
const DW_FORM_EXPRLOC: u64 = 0x18;
const DW_FORM_FLAG_PRESENT: u64 = 0x19;

const DW_ATE_BOOLEAN: u64 = 0x02;
const DW_ATE_FLOAT: u64 = 0x04;
const DW_ATE_SIGNED: u64 = 0x05;
const DW_ATE_UNSIGNED: u64 = 0x07;

const DW_OP_CALL_FRAME_CFA: u8 = 0x9c;

/// Every target this writer serves addresses with 64 bits.
const ADDRESS_SIZE: u8 = 8;

/// `S_ATTR_DEBUG`. Mach-O requires it on every section of the `__DWARF` segment, and
/// it is what tells `ld` to leave the section out of the linked image.
const MACHO_DEBUG_ATTRIBUTE: u32 = 0x0200_0000;

/// Lamina's scalars follow C's conventions for width, signedness and float format,
/// which is what a debugger reads this field for. Nothing here claims C semantics
/// beyond that.
const DW_LANG_C99: u64 = 0x0c;

const DW_LNS_COPY: u8 = 1;
const DW_LNS_ADVANCE_PC: u8 = 2;
const DW_LNS_ADVANCE_LINE: u8 = 3;
const DW_LNS_SET_FILE: u8 = 4;
const DW_LNS_SET_COLUMN: u8 = 5;
const DW_LNS_NEGATE_STMT: u8 = 6;
const DW_LNS_SET_BASIC_BLOCK: u8 = 7;
/// DWARF 4 section 6.2.5.2. A row carrying this is where a debugger should place a
/// function entry breakpoint, past the prologue, rather than guessing.
const DW_LNS_SET_PROLOGUE_END: u8 = 10;
const DW_LNS_SET_EPILOGUE_BEGIN: u8 = 11;
const DW_LNS_SET_ISA: u8 = 12;
const DW_LNE_END_SEQUENCE: u8 = 1;
const DW_LNE_SET_ADDRESS: u8 = 2;
const DW_LNE_SET_DISCRIMINATOR: u8 = 4;

const LINE_BASE: i8 = -5;
const LINE_RANGE: u8 = 14;
const OPCODE_BASE: u8 = 13;
const STANDARD_OPCODE_LENGTHS: [u8; 12] = [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1];

/// DWARF's encoding attribute for a base type: how a debugger should interpret the
/// bytes. DWARF 4 section 7.8, "Base type encodings".
fn dwarf_encoding(base: BaseType) -> u64 {
    match base {
        BaseType::Bool => DW_ATE_BOOLEAN,
        BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64 => DW_ATE_SIGNED,
        BaseType::U8 | BaseType::U16 | BaseType::U32 | BaseType::U64 => DW_ATE_UNSIGNED,
        BaseType::F32 | BaseType::F64 => DW_ATE_FLOAT,
    }
}

/// The abbreviation table, interned as entries are needed. A DIE asks for the exact
/// attribute list it is about to write and gets the code for it, so a new attribute
/// never has to be threaded through a table written by hand.
#[derive(PartialEq, Eq)]
struct Abbrev {
    tag: u64,
    children: bool,
    attributes: Vec<(u64, u64)>,
}

#[derive(Default)]
struct Abbrevs {
    entries: Vec<Abbrev>,
}

impl Abbrevs {
    fn code(&mut self, tag: u64, children: bool, attributes: &[(u64, u64)]) -> u64 {
        let wanted = Abbrev {
            tag,
            children,
            attributes: attributes.to_vec(),
        };
        match self.entries.iter().position(|entry| *entry == wanted) {
            Some(index) => index as u64 + 1,
            None => {
                self.entries.push(wanted);
                self.entries.len() as u64
            }
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            push_uleb(&mut out, index as u64 + 1);
            push_uleb(&mut out, entry.tag);
            out.push(u8::from(entry.children));
            for (attribute, form) in &entry.attributes {
                push_uleb(&mut out, *attribute);
                push_uleb(&mut out, *form);
            }
            out.extend_from_slice(&[0, 0]);
        }
        out.push(0);
        out
    }
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
    source: &DebugSource<'_>,
    fallback_name: &str,
    always_emit: bool,
) -> Result<Option<DebugSections>, RasError> {
    let DebugSource {
        format,
        sections,
        rows,
        files,
        extents,
        request,
    } = *source;
    let macho = format == ObjectFormat::MachO;
    let Some((abs64, abs32)) = absolute_kinds(arch) else {
        return Ok(None);
    };
    if rows.is_empty() && !always_emit {
        return Ok(None);
    }
    if sections
        .iter()
        .any(|s| s.name.starts_with(".debug_") || s.name.starts_with("__DWARF,"))
    {
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
        let mut isa = 0u64;
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
            if row.isa != isa {
                program.push(DW_LNS_SET_ISA);
                push_uleb(&mut program, row.isa);
                isa = row.isa;
            }
            if row.basic_block {
                program.push(DW_LNS_SET_BASIC_BLOCK);
            }
            if row.prologue_end {
                program.push(DW_LNS_SET_PROLOGUE_END);
            }
            if row.epilogue_begin {
                program.push(DW_LNS_SET_EPILOGUE_BEGIN);
            }
            if row.discriminator != 0 {
                program.push(0);
                let mut operand = Vec::new();
                push_uleb(&mut operand, row.discriminator);
                push_uleb(&mut program, operand.len() as u64 + 1);
                program.push(DW_LNE_SET_DISCRIMINATOR);
                program.extend_from_slice(&operand);
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

    let request = request.filter(|request| !request.subprograms.is_empty());
    let described: Vec<(&Subprogram, &FunctionExtent)> = request
        .map(|request| {
            request
                .subprograms
                .iter()
                .filter_map(|sub| Some((sub, extents.get(sub.object_symbol())?)))
                .collect()
        })
        .unwrap_or_default();

    let mut type_pool: Vec<DebugType> = Vec::new();
    for (sub, _) in &described {
        let used = sub
            .ret
            .iter()
            .chain(sub.params.iter().filter_map(|param| param.ty.as_ref()));
        for ty in used {
            if !type_pool.contains(ty) {
                type_pool.push(*ty);
            }
        }
    }

    let mut abbrevs = Abbrevs::default();
    let producer = request.and_then(|request| request.producer.as_deref());
    let mut cu_attributes: Vec<(u64, u64)> = Vec::new();
    if producer.is_some() {
        cu_attributes.push((DW_AT_PRODUCER, DW_FORM_STRING));
    }
    cu_attributes.push((DW_AT_LANGUAGE, DW_FORM_DATA2));
    if comp_dir.is_some() {
        cu_attributes.push((DW_AT_COMP_DIR, DW_FORM_STRING));
    }
    cu_attributes.extend([
        (DW_AT_NAME, DW_FORM_STRING),
        (DW_AT_STMT_LIST, DW_FORM_SEC_OFFSET),
        (DW_AT_LOW_PC, DW_FORM_ADDR),
    ]);
    cu_attributes.push(if ranged {
        (DW_AT_RANGES, DW_FORM_SEC_OFFSET)
    } else {
        (DW_AT_HIGH_PC, DW_FORM_DATA8)
    });
    let has_children = !described.is_empty() || !type_pool.is_empty();
    let cu_code = abbrevs.code(DW_TAG_COMPILE_UNIT, has_children, &cu_attributes);

    let mut info = vec![0u8; 4];
    info.extend_from_slice(&4u16.to_le_bytes());
    let mut info_relocs = vec![(info.len(), ".debug_abbrev".to_string(), 4, 0i64)];
    info.extend_from_slice(&[0; 4]);
    info.push(ADDRESS_SIZE);
    push_uleb(&mut info, cu_code);
    if let Some(producer) = producer {
        push_cstr(&mut info, producer);
    }
    info.extend_from_slice(&(DW_LANG_C99 as u16).to_le_bytes());
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

    // The type pool comes first, so every `DW_FORM_ref4` below points backwards and no
    // DIE needs patching after the fact.
    let mut type_offsets: HashMap<DebugType, u32> = HashMap::new();
    for ty in &type_pool {
        let offset = info.len() as u32;
        match ty {
            DebugType::Base(base) => {
                let code = abbrevs.code(
                    DW_TAG_BASE_TYPE,
                    false,
                    &[
                        (DW_AT_NAME, DW_FORM_STRING),
                        (DW_AT_ENCODING, DW_FORM_DATA1),
                        (DW_AT_BYTE_SIZE, DW_FORM_DATA1),
                    ],
                );
                push_uleb(&mut info, code);
                push_cstr(&mut info, base.name());
                info.push(dwarf_encoding(*base) as u8);
                info.push(base.byte_size());
            }
            DebugType::Pointer => {
                let code = abbrevs.code(
                    DW_TAG_POINTER_TYPE,
                    false,
                    &[(DW_AT_BYTE_SIZE, DW_FORM_DATA1)],
                );
                push_uleb(&mut info, code);
                info.push(ADDRESS_SIZE);
            }
        }
        type_offsets.insert(*ty, offset);
    }

    for (sub, extent) in &described {
        let decl = sub
            .decl
            .and_then(|decl| Some((file_index(decl.file).ok()?, u64::from(decl.line))));
        let ret = sub.ret.and_then(|ty| type_offsets.get(&ty).copied());
        let mut attributes = vec![(DW_AT_NAME, DW_FORM_STRING)];
        if sub.external {
            attributes.push((DW_AT_EXTERNAL, DW_FORM_FLAG_PRESENT));
        }
        if decl.is_some() {
            attributes.push((DW_AT_DECL_FILE, DW_FORM_UDATA));
            attributes.push((DW_AT_DECL_LINE, DW_FORM_UDATA));
        }
        attributes.push((DW_AT_PROTOTYPED, DW_FORM_FLAG_PRESENT));
        if ret.is_some() {
            attributes.push((DW_AT_TYPE, DW_FORM_REF4));
        }
        attributes.extend([
            (DW_AT_LOW_PC, DW_FORM_ADDR),
            (DW_AT_HIGH_PC, DW_FORM_DATA8),
            (DW_AT_FRAME_BASE, DW_FORM_EXPRLOC),
        ]);
        let children = !sub.params.is_empty() || sub.variadic;
        let code = abbrevs.code(DW_TAG_SUBPROGRAM, children, &attributes);
        push_uleb(&mut info, code);
        push_cstr(&mut info, &sub.name);
        if let Some((file, line)) = decl {
            push_uleb(&mut info, file);
            push_uleb(&mut info, line);
        }
        if let Some(offset) = ret {
            info.extend_from_slice(&offset.to_le_bytes());
        }
        info_relocs.push((info.len(), extent.section.clone(), 8, extent.start as i64));
        info.extend_from_slice(&[0; 8]);
        info.extend_from_slice(&extent.length.to_le_bytes());
        info.push(1);
        info.push(DW_OP_CALL_FRAME_CFA);

        // A parameter carries no `DW_AT_location`: nothing here knows where the
        // allocator put it, and a guessed location makes a debugger print a wrong
        // value rather than say it does not know.
        for param in &sub.params {
            let ty = param.ty.and_then(|ty| type_offsets.get(&ty).copied());
            let mut attributes = Vec::new();
            if param.name.is_some() {
                attributes.push((DW_AT_NAME, DW_FORM_STRING));
            }
            if ty.is_some() {
                attributes.push((DW_AT_TYPE, DW_FORM_REF4));
            }
            let code = abbrevs.code(DW_TAG_FORMAL_PARAMETER, false, &attributes);
            push_uleb(&mut info, code);
            if let Some(name) = param.name.as_deref() {
                push_cstr(&mut info, name);
            }
            if let Some(offset) = ty {
                info.extend_from_slice(&offset.to_le_bytes());
            }
        }
        if sub.variadic {
            let code = abbrevs.code(DW_TAG_UNSPECIFIED_PARAMETERS, false, &[]);
            push_uleb(&mut info, code);
        }
        if children {
            info.push(0);
        }
    }
    if has_children {
        info.push(0);
    }
    let abbrev = abbrevs.encode();
    let info_unit_length = (info.len() - 4) as u32;
    info[..4].copy_from_slice(&info_unit_length.to_le_bytes());

    let mut relocs = Vec::new();
    let mut add = |section: &str, position: usize, symbol: String, width: u8, addend: i64| {
        // Mach-O's linker does not relocate the `__DWARF` sections: it drops them, and
        // `dsymutil` reads the objects themselves. An offset from one debug section of
        // this object into another is therefore already final, and the only addresses
        // that still need a relocation are the ones that name code.
        if macho && symbol.starts_with(".debug_") {
            return;
        }
        relocs.push(reloc(
            &debug_section_name(macho, section),
            position,
            &symbol,
            width,
            addend,
            abs64,
            abs32,
        ));
    };
    for (position, symbol, width, addend) in line_relocs {
        add(
            ".debug_line",
            4 + program_start + position,
            symbol,
            width,
            addend,
        );
    }
    for (position, symbol, width, addend) in info_relocs {
        add(".debug_info", position, symbol, width, addend);
    }
    for (position, symbol, width, addend) in range_relocs {
        add(".debug_ranges", position, symbol, width, addend);
    }
    let mut out_sections = vec![
        section(macho, ".debug_line", debug_line_unit),
        section(macho, ".debug_abbrev", abbrev),
        section(macho, ".debug_info", info),
    ];
    if ranged {
        out_sections.push(section(macho, ".debug_ranges", ranges));
    }
    Ok(Some(DebugSections {
        sections: out_sections,
        relocs,
    }))
}

/// `.debug_info` is `__DWARF,__debug_info` in a Mach-O object. The bytes are the same.
fn debug_section_name(macho: bool, elf: &str) -> String {
    if macho {
        format!("__DWARF,__{}", &elf[1..])
    } else {
        elf.to_string()
    }
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

fn section(macho: bool, name: &str, data: Vec<u8>) -> Section {
    Section {
        group: None,
        name: debug_section_name(macho, name),
        flags: SectionFlags {
            alloc: false,
            exec: false,
            write: false,
            tls: false,
        },
        data,
        alignment: 1,
        macho: macho.then_some(MachOSectionType {
            kind: 0,
            attributes: MACHO_DEBUG_ATTRIBUTE,
        }),
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
