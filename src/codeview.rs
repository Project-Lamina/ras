//! CodeView `.debug$S` and `.debug$T` for COFF objects, built from the same
//! [`DebugInfoRequest`] the DWARF writer reads.
//!
//! No PDB is written here. A CodeView object carries symbol and type records and the
//! linker gathers them into the PDB, which is how every toolchain on Windows works and
//! what `link.exe` and `lld-link` already do with `.debug$S`/`.debug$T`.
//!
//! Nothing here states where a value lives at run time. The vreg-to-location map does
//! not outlive the backends, so no `S_LOCAL` and no `S_DEFRANGE_*` record is emitted;
//! a guessed range makes a debugger print a confident wrong value.

use crate::{
    debug::{BaseType, DebugSections, DebugSource, DebugType, FunctionExtent, LocRow, Subprogram},
    error::RasError,
    object::{COFF_SECREL32, COFF_SECTION, ExternalReloc},
    parser::{Section, SectionFlags},
};
use lamina_platform::TargetArchitecture;
use std::collections::HashMap;

/// `CV_SIGNATURE_C13`, the first dword of both CodeView sections. Microsoft's
/// `cvinfo.h`, `CV_SIGNATURE_C13 = 4`; nothing current reads an older signature.
const CV_SIGNATURE_C13: u32 = 4;

/// `DEBUG_S_SUBSECTION_TYPE` from `cvinfo.h`. The high bit (`DEBUG_S_IGNORE`) is never
/// set here: every subsection written is one a consumer should read.
const DEBUG_S_SYMBOLS: u32 = 0xf1;
const DEBUG_S_LINES: u32 = 0xf2;
const DEBUG_S_STRINGTABLE: u32 = 0xf3;
const DEBUG_S_FILECHKSMS: u32 = 0xf4;

/// `SYM_ENUM_e` from `cvinfo.h`. The `_ID` procedure records take an `LF_FUNC_ID` index
/// rather than an `LF_PROCEDURE` one, which is what puts the name in the type stream.
const S_OBJNAME: u16 = 0x1101;
const S_COMPILE3: u16 = 0x113c;
const S_LPROC32_ID: u16 = 0x1146;
const S_GPROC32_ID: u16 = 0x1147;
const S_PROC_ID_END: u16 = 0x114f;

/// `LEAF_ENUM_e` from `cvinfo.h`.
const LF_PROCEDURE: u16 = 0x1008;
const LF_ARGLIST: u16 = 0x1201;
const LF_FUNC_ID: u16 = 0x1601;

/// `CV_primitive_type` values from `cvconst.h`. A primitive has no record: the index
/// itself encodes the type, and indices below [`FIRST_TYPE_INDEX`] are reserved for them.
const T_NOTYPE: u32 = 0x0000;
const T_VOID: u32 = 0x0003;
const T_BOOL08: u32 = 0x0030;
const T_REAL32: u32 = 0x0040;
const T_REAL64: u32 = 0x0041;
const T_INT1: u32 = 0x0068;
const T_UINT1: u32 = 0x0069;
const T_INT2: u32 = 0x0072;
const T_UINT2: u32 = 0x0073;
const T_INT4: u32 = 0x0074;
const T_UINT4: u32 = 0x0075;
const T_INT8: u32 = 0x0076;
const T_UINT8: u32 = 0x0077;
/// `T_64PVOID`: a 64-bit pointer to void, which is all an untyped machine pointer is.
const T_64PVOID: u32 = 0x0603;

/// `CV_FIRST_NONPRIM`: the first index a record in `.debug$T` may be given.
const FIRST_TYPE_INDEX: u32 = 0x1000;

/// `CV_CALL_NEAR_C` from `CV_call_e`. Lamina's own calling convention is the platform
/// C one, which is what this field names.
const CV_CALL_NEAR_C: u8 = 0x00;

/// `CV_CFL_C` from `CV_CFL_LANG`. Lamina's scalars follow C's widths, signedness and
/// float formats, which is what this field is read for.
const CV_CFL_C: u32 = 0x00;

/// `CV_CPU_TYPE_e` from `cvconst.h`.
const CV_CFL_X64: u16 = 0xd0;
const CV_CFL_ARM64: u16 = 0xf6;

/// `CV_PROCFLAGS` is all optimizer and frame claims; Lamina makes none of them here.
const CV_PROCFLAGS_NONE: u8 = 0;

/// `CV_FileChecksumKind::None`. The assembler does not read the source file, so it
/// cannot hash it, and a wrong hash would make a debugger reject a correct source.
const CHECKSUM_NONE: u8 = 0;

fn cpu_type(arch: TargetArchitecture) -> Option<u16> {
    match arch {
        TargetArchitecture::X86_64 => Some(CV_CFL_X64),
        TargetArchitecture::Aarch64 => Some(CV_CFL_ARM64),
        _ => None,
    }
}

fn primitive(ty: DebugType) -> u32 {
    match ty {
        DebugType::Pointer => T_64PVOID,
        DebugType::Base(base) => match base {
            BaseType::Bool => T_BOOL08,
            BaseType::I8 => T_INT1,
            BaseType::U8 => T_UINT1,
            BaseType::I16 => T_INT2,
            BaseType::U16 => T_UINT2,
            BaseType::I32 => T_INT4,
            BaseType::U32 => T_UINT4,
            BaseType::I64 => T_INT8,
            BaseType::U64 => T_UINT8,
            BaseType::F32 => T_REAL32,
            BaseType::F64 => T_REAL64,
        },
    }
}

/// The type stream, interned: two functions with the same signature share one
/// `LF_PROCEDURE` and one `LF_ARGLIST`, as they do in every CodeView producer.
#[derive(Default)]
struct TypeStream {
    data: Vec<u8>,
    index_of: HashMap<Vec<u8>, u32>,
    next: u32,
}

impl TypeStream {
    fn new() -> Self {
        Self {
            data: CV_SIGNATURE_C13.to_le_bytes().to_vec(),
            index_of: HashMap::new(),
            next: FIRST_TYPE_INDEX,
        }
    }

    /// `body` is the record after its kind, without the length prefix. A record is
    /// padded to four bytes with `LF_PAD`n bytes, which a reader skips by value.
    fn intern(&mut self, kind: u16, body: Vec<u8>) -> Result<u32, RasError> {
        let mut record = kind.to_le_bytes().to_vec();
        record.extend_from_slice(&body);
        while !(record.len() + 2).is_multiple_of(4) {
            record.push(0xf0 + (4 - ((record.len() + 2) % 4)) as u8);
        }
        if let Some(index) = self.index_of.get(&record) {
            return Ok(*index);
        }
        let length = u16::try_from(record.len()).map_err(|_| {
            RasError::ObjectError("CodeView type record is longer than 65535 bytes".into())
        })?;
        let index = self.next;
        self.next += 1;
        self.data.extend_from_slice(&length.to_le_bytes());
        self.data.extend_from_slice(&record);
        self.index_of.insert(record, index);
        Ok(index)
    }
}

/// A parameter whose type the caller could not state becomes `T_NOTYPE`, the same
/// silence DWARF spells by leaving `DW_AT_type` off the formal parameter. CodeView
/// also spells a trailing `...` as `T_NOTYPE`, so a variadic function appends one of
/// its own: two in a row is a last untyped parameter followed by the ellipsis.
fn arglist_body(sub: &Subprogram) -> Vec<u8> {
    let mut args: Vec<u32> = sub
        .params
        .iter()
        .map(|param| param.ty.map_or(T_NOTYPE, primitive))
        .collect();
    if sub.variadic {
        args.push(T_NOTYPE);
    }
    let mut body = (args.len() as u32).to_le_bytes().to_vec();
    for arg in args {
        body.extend_from_slice(&arg.to_le_bytes());
    }
    body
}

fn func_id(types: &mut TypeStream, sub: &Subprogram) -> Result<u32, RasError> {
    let count = sub.params.len() + usize::from(sub.variadic);
    let arglist = types.intern(LF_ARGLIST, arglist_body(sub))?;
    let mut procedure = sub.ret.map_or(T_VOID, primitive).to_le_bytes().to_vec();
    procedure.push(CV_CALL_NEAR_C);
    procedure.push(0);
    procedure.extend_from_slice(&(count as u16).to_le_bytes());
    procedure.extend_from_slice(&arglist.to_le_bytes());
    let procedure = types.intern(LF_PROCEDURE, procedure)?;
    let mut body = 0u32.to_le_bytes().to_vec();
    body.extend_from_slice(&procedure.to_le_bytes());
    push_cstr(&mut body, &sub.name);
    types.intern(LF_FUNC_ID, body)
}

/// A `.debug$S` under construction: subsections, with the byte offsets of the fields a
/// COFF relocation has to fill in.
struct Symbols {
    data: Vec<u8>,
    relocs: Vec<(usize, String, u32)>,
}

impl Symbols {
    fn new() -> Self {
        Self {
            data: CV_SIGNATURE_C13.to_le_bytes().to_vec(),
            relocs: Vec::new(),
        }
    }

    fn subsection(&mut self, kind: u32, mut body: Vec<u8>, relocs: &[(usize, String, u32)]) {
        while !body.len().is_multiple_of(4) {
            body.push(0);
        }
        self.data.extend_from_slice(&kind.to_le_bytes());
        self.data
            .extend_from_slice(&(body.len() as u32).to_le_bytes());
        let base = self.data.len();
        for (offset, symbol, kind) in relocs {
            self.relocs.push((base + offset, symbol.clone(), *kind));
        }
        self.data.extend_from_slice(&body);
    }
}

/// `body` is the record after its kind, without the length prefix. A symbol record is
/// padded to four bytes with zeros, and its length covers the padding.
fn symbol(kind: u16, body: &[u8]) -> Vec<u8> {
    let mut record = kind.to_le_bytes().to_vec();
    record.extend_from_slice(body);
    while !(record.len() + 2).is_multiple_of(4) {
        record.push(0);
    }
    let mut out = (record.len() as u16).to_le_bytes().to_vec();
    out.extend_from_slice(&record);
    out
}

pub(crate) fn build(
    arch: TargetArchitecture,
    source: &DebugSource<'_>,
    object_name: &str,
) -> Result<Option<DebugSections>, RasError> {
    let DebugSource {
        format: _,
        sections,
        rows,
        files,
        extents,
        request,
    } = *source;
    let Some(request) = request else {
        return Ok(None);
    };
    let Some(machine) = cpu_type(arch) else {
        return Ok(None);
    };
    if let Some(s) = sections
        .iter()
        .find(|s| s.name == ".debug$S" || s.name == ".debug$T")
    {
        return Err(RasError::ObjectError(format!(
            "source defines {}; ras cannot add CodeView beside it",
            s.name
        )));
    }

    // CodeView has no compilation directory: the string table holds the path a
    // debugger will open, so a relative `.file` would be resolved against whatever
    // directory the linker happened to run in.
    let cwd = std::env::current_dir()
        .ok()
        .map(|dir| dir.display().to_string());
    let mut strings: Vec<u8> = vec![0];
    let mut checksums: Vec<u8> = Vec::new();
    let mut checksum_offset: HashMap<u32, u32> = HashMap::new();
    for (number, file) in files {
        let path = match file.directory.as_deref() {
            Some(directory) => format!("{directory}/{}", file.name),
            None => file.name.clone(),
        };
        let path = match (path.starts_with('/'), cwd.as_deref()) {
            (false, Some(cwd)) => format!("{cwd}/{path}"),
            _ => path,
        };
        let name_offset = strings.len() as u32;
        push_cstr(&mut strings, &path);
        checksum_offset.insert(*number, checksums.len() as u32);
        checksums.extend_from_slice(&name_offset.to_le_bytes());
        checksums.push(0);
        checksums.push(CHECKSUM_NONE);
        while !checksums.len().is_multiple_of(4) {
            checksums.push(0);
        }
    }

    let described: Vec<(&Subprogram, &FunctionExtent)> = request
        .subprograms
        .iter()
        .filter_map(|sub| Some((sub, extents.get(sub.object_symbol())?)))
        .collect();

    let mut types = TypeStream::new();
    let mut symbols = Symbols::new();

    let mut head = symbol(S_OBJNAME, &objname_body(object_name));
    head.extend_from_slice(&symbol(
        S_COMPILE3,
        &compile3_body(machine, request.producer.as_deref().unwrap_or("lamina")),
    ));
    symbols.subsection(DEBUG_S_SYMBOLS, head, &[]);

    for (sub, extent) in &described {
        let id = func_id(&mut types, sub)?;
        let kind = if sub.external {
            S_GPROC32_ID
        } else {
            S_LPROC32_ID
        };
        let mut body = Vec::new();
        for field in [0u32, 0, 0, extent.length as u32, 0, 0, id, 0] {
            body.extend_from_slice(&field.to_le_bytes());
        }
        body.extend_from_slice(&0u16.to_le_bytes());
        body.push(CV_PROCFLAGS_NONE);
        push_cstr(&mut body, &sub.name);
        // `CodeOffset` is the eighth dword and `Segment` the halfword after it; both
        // start at the record body, which starts two bytes of length and two of kind
        // into the record.
        let proc_relocs = vec![
            (4 + 7 * 4, sub.object_symbol().to_string(), COFF_SECREL32),
            (4 + 8 * 4, sub.object_symbol().to_string(), COFF_SECTION),
        ];
        let mut proc = symbol(kind, &body);
        proc.extend_from_slice(&symbol(S_PROC_ID_END, &[]));
        symbols.subsection(DEBUG_S_SYMBOLS, proc, &proc_relocs);

        if let Some(lines) = lines_body(sub, extent, rows, &checksum_offset)? {
            symbols.subsection(
                DEBUG_S_LINES,
                lines,
                &[
                    (0, sub.object_symbol().to_string(), COFF_SECREL32),
                    (4, sub.object_symbol().to_string(), COFF_SECTION),
                ],
            );
        }
    }

    symbols.subsection(DEBUG_S_FILECHKSMS, checksums, &[]);
    symbols.subsection(DEBUG_S_STRINGTABLE, strings, &[]);

    let relocs = symbols
        .relocs
        .iter()
        .map(|(offset, symbol, kind)| ExternalReloc {
            offset: *offset,
            symbol: symbol.clone(),
            section: ".debug$S".to_string(),
            kind: *kind,
            addend: 0,
            trailing: 0,
            branch: false,
            section_symbol: false,
        })
        .collect();

    Ok(Some(DebugSections {
        sections: vec![
            section(".debug$S", symbols.data),
            section(".debug$T", types.data),
        ],
        relocs,
    }))
}

/// A CodeView section is never loaded, and the records inside it are four-aligned.
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
        alignment: 4,
        macho: None,
    }
}

fn objname_body(name: &str) -> Vec<u8> {
    let mut body = 0u32.to_le_bytes().to_vec();
    push_cstr(&mut body, name);
    body
}

/// `COMPILESYM3`: a language and flag word, the CPU, then four-part front end and back
/// end versions, then the version string. Lamina has one version, so both halves carry
/// it and the string says which compiler it was.
fn compile3_body(machine: u16, producer: &str) -> Vec<u8> {
    let mut body = CV_CFL_C.to_le_bytes().to_vec();
    body.extend_from_slice(&machine.to_le_bytes());
    for _ in 0..8 {
        body.extend_from_slice(&0u16.to_le_bytes());
    }
    push_cstr(&mut body, producer);
    body
}

/// `CV_DebugSLinesHeader_t`, then one `CV_DebugSLinesFileBlockHeader_t` per run of rows
/// from one file, then `CV_Line_t` entries whose offsets are relative to the function.
/// No column subsection is written, so the header's `fHasColumns` flag stays clear.
fn lines_body(
    sub: &Subprogram,
    extent: &FunctionExtent,
    rows: &[LocRow],
    checksum_offset: &HashMap<u32, u32>,
) -> Result<Option<Vec<u8>>, RasError> {
    let end = extent.start + extent.length;
    let mut inside: Vec<&LocRow> = rows
        .iter()
        .filter(|row| {
            row.section == extent.section && row.offset >= extent.start && row.offset < end
        })
        .collect();
    inside.sort_by_key(|row| row.offset);
    let mut picked: Vec<&LocRow> = Vec::with_capacity(inside.len());
    for row in inside {
        if picked.last().is_some_and(|last| last.offset == row.offset) {
            picked.pop();
        }
        picked.push(row);
    }
    if picked.is_empty() {
        return Ok(None);
    }

    let mut body = 0u32.to_le_bytes().to_vec();
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&(extent.length as u32).to_le_bytes());

    let mut start = 0usize;
    while start < picked.len() {
        let file = picked[start].file;
        let mut stop = start;
        while stop < picked.len() && picked[stop].file == file {
            stop += 1;
        }
        let offset = *checksum_offset.get(&file).ok_or_else(|| {
            RasError::ObjectError(format!("{} has a .loc in undefined file {file}", sub.name))
        })?;
        let count = (stop - start) as u32;
        body.extend_from_slice(&offset.to_le_bytes());
        body.extend_from_slice(&count.to_le_bytes());
        body.extend_from_slice(&(12 + count * 8).to_le_bytes());
        for row in &picked[start..stop] {
            body.extend_from_slice(&((row.offset - extent.start) as u32).to_le_bytes());
            body.extend_from_slice(&line_entry(row.line, row.is_stmt).to_le_bytes());
        }
        start = stop;
    }
    Ok(Some(body))
}

/// `CV_Line_t`'s second word: the start line in the low 24 bits, the distance to the
/// end line in the next 7, and `fStatement` in the top bit.
fn line_entry(line: u32, is_stmt: bool) -> u32 {
    (line & 0x00ff_ffff) | (u32::from(is_stmt) << 31)
}

fn push_cstr(buf: &mut Vec<u8>, text: &str) {
    buf.extend_from_slice(text.as_bytes());
    buf.push(0);
}
