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

#[cfg(test)]
mod tests {
    /// CodeView `.debug$S` and `.debug$T` in a COFF object, read back from the bytes.
    ///
    /// The assertions are questions a debugger asks: which functions exist, how long each
    /// one is, what it returns, what its parameters are called, and which source line each
    /// instruction came from. None of them compares bytes with another toolchain's output.
    mod records {
        use crate::test_support::TempDir;
        use crate::{
            BaseType, DebugInfoRequest, DebugType, DeclSite, FormalParameter, ObjectWriteOptions,
            Ras, Subprogram,
        };
        use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
        use std::process::Command;

        /// `twice` is followed by `.p2align 4`, so the padding between it and `half` belongs
        /// to neither function and must not land in either one's code size.
        const SOURCE: &str = ".file 1 \"shapes.lamina\"\n\
        .text\n\
        .globl twice\n\
        twice:\n\
        .loc 1 1 1\n\
        addl %edi, %edi\n\
        .loc 1 2 1\n\
        movl %edi, %eax\n\
        ret\n\
        .p2align 4\n\
        half:\n\
        .loc 1 6 1\n\
        movl %edi, %eax\n\
        ret\n";

        /// The same two functions in aarch64 instructions, so the Windows-on-ARM relocation
        /// types get read back from a real object rather than only from x86_64 ones.
        const AARCH64_SOURCE: &str = ".file 1 \"shapes.lamina\"\n\
        .text\n\
        .globl twice\n\
        twice:\n\
        .loc 1 1 1\n\
        add w0, w0, w0\n\
        .loc 1 2 1\n\
        ret\n\
        .p2align 4\n\
        half:\n\
        .loc 1 6 1\n\
        asr w0, w0, #1\n\
        ret\n";

        const DEBUG_S_SYMBOLS: u32 = 0xf1;
        const DEBUG_S_LINES: u32 = 0xf2;
        const DEBUG_S_STRINGTABLE: u32 = 0xf3;
        const DEBUG_S_FILECHKSMS: u32 = 0xf4;
        const S_OBJNAME: u16 = 0x1101;
        const S_COMPILE3: u16 = 0x113c;
        const S_LPROC32_ID: u16 = 0x1146;
        const S_GPROC32_ID: u16 = 0x1147;
        const S_PROC_ID_END: u16 = 0x114f;
        const LF_PROCEDURE: u16 = 0x1008;
        const LF_ARGLIST: u16 = 0x1201;
        const LF_FUNC_ID: u16 = 0x1601;
        const T_INT4: u32 = 0x0074;

        /// `S_LOCAL` and every `S_DEFRANGE_*` from `SYM_ENUM_e`, which is the whole range
        /// 0x113e to 0x1145. Nothing outside the backends knows where a value lives, so a
        /// record from this range could only be a guess.
        const LOCATION_RECORDS: std::ops::RangeInclusive<u16> = 0x113e..=0x1145;

        const IMAGE_REL_AMD64_SECTION: u16 = 0x000a;
        const IMAGE_REL_AMD64_SECREL: u16 = 0x000b;
        const IMAGE_REL_ARM64_SECREL: u16 = 0x0008;
        const IMAGE_REL_ARM64_SECTION: u16 = 0x000d;

        fn request() -> DebugInfoRequest {
            DebugInfoRequest {
                producer: Some("ras test".into()),
                subprograms: vec![
                    Subprogram {
                        name: "twice".into(),
                        symbol: None,
                        external: true,
                        decl: Some(DeclSite { file: 1, line: 1 }),
                        ret: Some(DebugType::Base(BaseType::I32)),
                        params: vec![FormalParameter {
                            name: Some("n".into()),
                            ty: Some(DebugType::Base(BaseType::I32)),
                        }],
                        variadic: false,
                    },
                    Subprogram {
                        name: "half".into(),
                        symbol: None,
                        external: false,
                        decl: Some(DeclSite { file: 1, line: 6 }),
                        ret: Some(DebugType::Base(BaseType::I32)),
                        params: vec![FormalParameter {
                            name: Some("n".into()),
                            ty: Some(DebugType::Base(BaseType::I32)),
                        }],
                        variadic: false,
                    },
                    // Nothing in the source defines this, so it must be left undescribed
                    // rather than described without an address.
                    Subprogram {
                        name: "absent".into(),
                        symbol: None,
                        external: true,
                        decl: None,
                        ret: None,
                        params: Vec::new(),
                        variadic: false,
                    },
                ],
            }
        }

        fn object_for(arch: TargetArchitecture, source: &str) -> Vec<u8> {
            Ras::with_object_write_options(
                arch,
                TargetOperatingSystem::Windows,
                ObjectWriteOptions {
                    function_symbols: vec!["twice".into(), "half".into()],
                    debug_info: Some(request()),
                    debug_object_name: "shapes.obj".into(),
                    ..ObjectWriteOptions::default()
                },
            )
            .expect("assembler")
            .assemble_to_bytes(source)
            .expect("assemble")
        }

        fn u16_at(bytes: &[u8], at: usize) -> u16 {
            u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"))
        }

        fn u32_at(bytes: &[u8], at: usize) -> u32 {
            u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
        }

        struct Coff {
            bytes: Vec<u8>,
        }

        struct Reloc {
            offset: u32,
            symbol: String,
            kind: u16,
        }

        impl Coff {
            fn new() -> Self {
                Self {
                    bytes: object_for(TargetArchitecture::X86_64, SOURCE),
                }
            }

            fn aarch64() -> Self {
                Self {
                    bytes: object_for(TargetArchitecture::Aarch64, AARCH64_SOURCE),
                }
            }

            fn section_header(&self, name: &str) -> Option<usize> {
                let count = usize::from(u16_at(&self.bytes, 2));
                (0..count).map(|i| 20 + i * 40).find(|&at| {
                    let raw = &self.bytes[at..at + 8];
                    let end = raw.iter().position(|&b| b == 0).unwrap_or(8);
                    raw[..end] == *name.as_bytes()
                })
            }

            fn section(&self, name: &str) -> Vec<u8> {
                let at = self
                    .section_header(name)
                    .unwrap_or_else(|| panic!("no {name} section"));
                let size = u32_at(&self.bytes, at + 16) as usize;
                let start = u32_at(&self.bytes, at + 20) as usize;
                self.bytes[start..start + size].to_vec()
            }

            fn symbol_name(&self, index: usize) -> String {
                let table = u32_at(&self.bytes, 8) as usize;
                let count = u32_at(&self.bytes, 12) as usize;
                let strings = table + count * 18;
                let entry = table + index * 18;
                let raw = &self.bytes[entry..entry + 18];
                if u32_at(raw, 0) == 0 {
                    let at = strings + u32_at(raw, 4) as usize;
                    let end = self.bytes[at..]
                        .iter()
                        .position(|&b| b == 0)
                        .expect("nul terminated");
                    String::from_utf8_lossy(&self.bytes[at..at + end]).into_owned()
                } else {
                    let end = raw[..8].iter().position(|&b| b == 0).unwrap_or(8);
                    String::from_utf8_lossy(&raw[..end]).into_owned()
                }
            }

            fn relocs(&self, name: &str) -> Vec<Reloc> {
                let at = self
                    .section_header(name)
                    .unwrap_or_else(|| panic!("no {name} section"));
                let start = u32_at(&self.bytes, at + 24) as usize;
                let count = usize::from(u16_at(&self.bytes, at + 32));
                (0..count)
                    .map(|i| {
                        let entry = start + i * 10;
                        Reloc {
                            offset: u32_at(&self.bytes, entry),
                            symbol: self.symbol_name(u32_at(&self.bytes, entry + 4) as usize),
                            kind: u16_at(&self.bytes, entry + 8),
                        }
                    })
                    .collect()
            }
        }

        /// `(kind, content, where the content starts in the section)` of every subsection of
        /// a `.debug$S`, each padded out to four bytes before the next begins.
        fn subsections(section: &[u8]) -> Vec<(u32, Vec<u8>, usize)> {
            assert_eq!(u32_at(section, 0), 4, "CV_SIGNATURE_C13");
            let mut out = Vec::new();
            let mut at = 4;
            while at + 8 <= section.len() {
                let kind = u32_at(section, at);
                let size = u32_at(section, at + 4) as usize;
                out.push((kind, section[at + 8..at + 8 + size].to_vec(), at + 8));
                at += 8 + size.next_multiple_of(4);
            }
            out
        }

        /// `(kind, body, offset in the subsection)` of every record in a symbol subsection.
        fn records(content: &[u8]) -> Vec<(u16, Vec<u8>, usize)> {
            let mut out = Vec::new();
            let mut at = 0;
            while at + 4 <= content.len() {
                let size = usize::from(u16_at(content, at));
                let kind = u16_at(content, at + 2);
                out.push((kind, content[at + 4..at + 2 + size].to_vec(), at));
                at += 2 + size;
            }
            out
        }

        /// Every symbol record in the section, each with where its body starts in the section.
        fn symbol_records(section: &[u8]) -> Vec<(u16, Vec<u8>, usize)> {
            subsections(section)
                .into_iter()
                .filter(|(kind, ..)| *kind == DEBUG_S_SYMBOLS)
                .flat_map(|(_, content, base)| {
                    records(&content)
                        .into_iter()
                        .map(|(kind, body, at)| (kind, body, base + at + 4))
                        .collect::<Vec<_>>()
                })
                .collect()
        }

        /// `(index, kind, body)` of every record in a `.debug$T`.
        fn type_records(section: &[u8]) -> Vec<(u32, u16, Vec<u8>)> {
            assert_eq!(u32_at(section, 0), 4, "CV_SIGNATURE_C13");
            let mut out = Vec::new();
            let mut at = 4;
            let mut index = 0x1000;
            while at + 4 <= section.len() {
                let size = usize::from(u16_at(section, at));
                out.push((
                    index,
                    u16_at(section, at + 2),
                    section[at + 4..at + 2 + size].to_vec(),
                ));
                index += 1;
                at += 2 + size;
            }
            out
        }

        fn name_at(body: &[u8], at: usize) -> String {
            let end = body[at..].iter().position(|&b| b == 0).expect("nul");
            String::from_utf8_lossy(&body[at..at + end]).into_owned()
        }

        /// The `PROCSYM32` body: parent, end, next, code size, debug start, debug end, the
        /// `LF_FUNC_ID` index, the code offset, the segment, flags, then the name.
        fn procedure(body: &[u8]) -> (String, u32, u32) {
            (name_at(body, 35), u32_at(body, 12), u32_at(body, 24))
        }

        #[test]
        fn the_object_names_itself_and_its_producer() {
            let section = Coff::new().section(".debug$S");
            let records = symbol_records(&section);
            let objname = records
                .iter()
                .find(|(kind, ..)| *kind == S_OBJNAME)
                .expect("S_OBJNAME");
            assert_eq!(name_at(&objname.1, 4), "shapes.obj");
            let compile = records
                .iter()
                .find(|(kind, ..)| *kind == S_COMPILE3)
                .expect("S_COMPILE3");
            assert_eq!(name_at(&compile.1, 22), "ras test");
            // CV_CFL_X64, which is what a reader checks before trusting a register number.
            assert_eq!(u16_at(&compile.1, 4), 0xd0);
        }

        #[test]
        fn each_defined_function_is_a_procedure_sized_like_its_code() {
            let coff = Coff::new();
            let section = coff.section(".debug$S");
            let procedures: Vec<(u16, (String, u32, u32))> = symbol_records(&section)
                .iter()
                .filter(|(kind, ..)| *kind == S_GPROC32_ID || *kind == S_LPROC32_ID)
                .map(|(kind, body, _)| (*kind, procedure(body)))
                .collect();

            assert_eq!(procedures.len(), 2, "a name with no code is not described");
            // addl + movl + ret, and none of the alignment padding after them.
            assert_eq!(procedures[0].0, S_GPROC32_ID);
            assert_eq!(procedures[0].1.0, "twice");
            assert_eq!(procedures[0].1.1, 5);
            assert_eq!(procedures[1].0, S_LPROC32_ID, "half is not global");
            assert_eq!(procedures[1].1.0, "half");
            assert_eq!(procedures[1].1.1, 3);
        }

        #[test]
        fn every_procedure_is_closed() {
            let coff = Coff::new();
            let kinds: Vec<u16> = symbol_records(&coff.section(".debug$S"))
                .iter()
                .map(|(kind, ..)| *kind)
                .filter(|kind| matches!(*kind, S_GPROC32_ID | S_LPROC32_ID | S_PROC_ID_END))
                .collect();
            assert_eq!(
                kinds,
                vec![S_GPROC32_ID, S_PROC_ID_END, S_LPROC32_ID, S_PROC_ID_END]
            );
        }

        /// The vreg-to-location map does not outlive the backends, so no record that states
        /// where a value lives may be written. A future guess trips this before it reaches a
        /// debugger.
        #[test]
        fn no_record_states_where_a_value_lives() {
            let coff = Coff::new();
            for (kind, ..) in symbol_records(&coff.section(".debug$S")) {
                assert!(
                    !LOCATION_RECORDS.contains(&kind),
                    "symbol record {kind:#06x} states a variable location"
                );
            }
        }

        /// `PROCSYM32` holds no address: the code offset is the eighth dword of its body and
        /// the segment the halfword after it, and both are filled in by a COFF relocation
        /// against the function's own symbol.
        #[test]
        fn a_procedure_addresses_its_function_through_relocations() {
            let coff = Coff::new();
            let section = coff.section(".debug$S");
            let body = symbol_records(&section)
                .into_iter()
                .find(|(kind, body, _)| *kind == S_GPROC32_ID && procedure(body).0 == "twice")
                .expect("twice")
                .2;
            let relocs = coff.relocs(".debug$S");
            let at = |offset: usize| {
                relocs
                    .iter()
                    .find(|r| r.offset as usize == offset)
                    .unwrap_or_else(|| panic!("no relocation at {offset}"))
            };
            let code_offset = at(body + 7 * 4);
            assert_eq!(code_offset.kind, IMAGE_REL_AMD64_SECREL);
            assert_eq!(code_offset.symbol, "twice");
            let segment = at(body + 8 * 4);
            assert_eq!(segment.kind, IMAGE_REL_AMD64_SECTION);
            assert_eq!(segment.symbol, "twice");
        }

        /// The ARM64 machine type numbers its relocations differently from AMD64, and a
        /// procedure's segment halfword is the one type the two tables disagree about most
        /// easily: `IMAGE_REL_ARM64_SECTION` is `0x000D`, not an AMD64 value reused.
        #[test]
        fn a_procedure_on_aarch64_addresses_its_function_through_arm64_relocations() {
            let coff = Coff::aarch64();
            let section = coff.section(".debug$S");
            let body = symbol_records(&section)
                .into_iter()
                .find(|(kind, body, _)| *kind == S_GPROC32_ID && procedure(body).0 == "twice")
                .expect("twice")
                .2;
            let relocs = coff.relocs(".debug$S");
            let at = |offset: usize| {
                relocs
                    .iter()
                    .find(|r| r.offset as usize == offset)
                    .unwrap_or_else(|| panic!("no relocation at {offset}"))
            };
            let code_offset = at(body + 7 * 4);
            assert_eq!(code_offset.kind, IMAGE_REL_ARM64_SECREL);
            assert_eq!(code_offset.symbol, "twice");
            let segment = at(body + 8 * 4);
            assert_eq!(segment.kind, IMAGE_REL_ARM64_SECTION);
            assert_eq!(segment.symbol, "twice");
        }

        #[test]
        fn a_line_subsection_maps_code_offsets_to_source_lines() {
            let coff = Coff::new();
            let section = coff.section(".debug$S");
            let lines: Vec<Vec<u8>> = subsections(&section)
                .into_iter()
                .filter(|(kind, ..)| *kind == DEBUG_S_LINES)
                .map(|(_, content, _)| content)
                .collect();
            assert_eq!(lines.len(), 2, "one fragment per function with rows");

            // CV_DebugSLinesHeader_t: offset, segment, flags, then the byte length covered.
            assert_eq!(u16_at(&lines[0], 6), 0, "no column subsection is written");
            assert_eq!(u32_at(&lines[0], 8), 5, "the fragment covers twice");

            // CV_DebugSLinesFileBlockHeader_t, then CV_Line_t pairs.
            assert_eq!(u32_at(&lines[0], 16), 2, "two rows");
            assert_eq!(
                u32_at(&lines[0], 20),
                12 + 2 * 8,
                "the block size counts them"
            );
            let rows: Vec<(u32, u32)> = (0..2)
                .map(|i| {
                    let at = 24 + i * 8;
                    (u32_at(&lines[0], at), u32_at(&lines[0], at + 4))
                })
                .collect();
            assert_eq!(rows[0].0, 0, "the first row is at the function start");
            assert_eq!(rows[0].1 & 0x00ff_ffff, 1);
            assert_eq!(rows[1].0, 2, "addl is two bytes");
            assert_eq!(rows[1].1 & 0x00ff_ffff, 2);
            assert_eq!(rows[0].1 >> 31, 1, "a .loc row is a statement");

            // The block names its file by an offset into the checksum subsection, and the
            // one file in this source is the first entry there.
            let checksums = subsections(&section)
                .into_iter()
                .find(|(kind, ..)| *kind == DEBUG_S_FILECHKSMS)
                .expect("a checksum subsection")
                .1;
            assert_eq!(u32_at(&lines[0], 12), 0);
            assert!(!checksums.is_empty());
        }

        #[test]
        fn the_source_file_is_named_by_a_path_a_debugger_can_open() {
            let coff = Coff::new();
            let section = coff.section(".debug$S");
            let strings = subsections(&section)
                .into_iter()
                .find(|(kind, ..)| *kind == DEBUG_S_STRINGTABLE)
                .expect("a string table")
                .1;
            assert_eq!(strings[0], 0, "offset zero is the empty string");
            let path = name_at(&strings, 1);
            assert!(path.ends_with("shapes.lamina"), "{path}");
            assert!(
                std::path::Path::new(&path).is_absolute(),
                "CodeView has no compilation directory, so the path has to be absolute: {path}"
            );
        }

        #[test]
        fn a_primitive_type_gets_no_record_of_its_own() {
            let coff = Coff::new();
            let records = type_records(&coff.section(".debug$T"));
            let kinds: Vec<u16> = records.iter().map(|(_, kind, _)| *kind).collect();
            // Two functions of the same shape share one arglist and one procedure.
            assert_eq!(
                kinds,
                vec![LF_ARGLIST, LF_PROCEDURE, LF_FUNC_ID, LF_FUNC_ID]
            );

            let (arglist_index, _, arglist) = &records[0];
            assert_eq!(u32_at(arglist, 0), 1);
            assert_eq!(u32_at(arglist, 4), T_INT4, "i32 is a built-in index");

            let (procedure_index, _, procedure) = &records[1];
            assert_eq!(u32_at(procedure, 0), T_INT4, "the return type");
            assert_eq!(u16_at(procedure, 6), 1, "one parameter");
            assert_eq!(u32_at(procedure, 8), *arglist_index);

            let names: Vec<String> = records[2..]
                .iter()
                .map(|(_, _, body)| name_at(body, 8))
                .collect();
            assert_eq!(names, vec!["twice", "half"]);
            assert_eq!(u32_at(&records[2].2, 4), *procedure_index);
        }

        /// CodeView spells a trailing `...` as a `T_NOTYPE` argument that `parmcount` counts,
        /// which is the same thing it spells an unstatable parameter type with. A variadic
        /// function therefore ends in two of them when its last parameter is untyped.
        #[test]
        fn a_variadic_function_ends_its_arglist_with_t_notype() {
            let object = Ras::with_object_write_options(
                TargetArchitecture::X86_64,
                TargetOperatingSystem::Windows,
                ObjectWriteOptions {
                    function_symbols: vec!["twice".into()],
                    debug_info: Some(DebugInfoRequest {
                        producer: Some("ras test".into()),
                        subprograms: vec![Subprogram {
                            name: "twice".into(),
                            symbol: None,
                            external: true,
                            decl: None,
                            ret: Some(DebugType::Base(BaseType::I32)),
                            params: vec![
                                FormalParameter {
                                    name: Some("n".into()),
                                    ty: Some(DebugType::Base(BaseType::I32)),
                                },
                                FormalParameter {
                                    name: Some("shape".into()),
                                    ty: None,
                                },
                            ],
                            variadic: true,
                        }],
                    }),
                    ..ObjectWriteOptions::default()
                },
            )
            .expect("assembler")
            .assemble_to_bytes(SOURCE)
            .expect("assemble");

            let section = {
                let coff = Coff { bytes: object };
                coff.section(".debug$T")
            };
            let records = type_records(&section);
            let (_, kind, arglist) = &records[0];
            assert_eq!(*kind, LF_ARGLIST);
            assert_eq!(u32_at(arglist, 0), 3, "two parameters and the ellipsis");
            assert_eq!(u32_at(arglist, 4), T_INT4);
            assert_eq!(u32_at(arglist, 8), 0, "an unstatable parameter type");
            assert_eq!(u32_at(arglist, 12), 0, "the ellipsis");
            let (_, kind, procedure) = &records[1];
            assert_eq!(*kind, LF_PROCEDURE);
            assert_eq!(u16_at(procedure, 6), 3, "parmcount counts the ellipsis");
        }

        #[test]
        fn llvm_pdbutil_reads_the_records_back() {
            let Some(pdbutil) = tool(&["llvm-pdbutil", "llvm-pdbutil-18", "llvm-pdbutil-19"])
            else {
                return;
            };
            let dir = TempDir::new("codeview_pdbutil");
            let path = dir.path().join("shapes.obj");
            std::fs::write(&path, object_for(TargetArchitecture::X86_64, SOURCE)).expect("write");
            let output = Command::new(pdbutil)
                .args(["dump", "--symbols", "--types", "-l"])
                .arg(&path)
                .output()
                .expect("llvm-pdbutil");
            let text = String::from_utf8_lossy(&output.stdout);
            assert!(output.status.success(), "{text}");
            assert!(text.contains("S_GPROC32_ID"), "{text}");
            assert!(text.contains("`twice`"), "{text}");
            assert!(text.contains("LF_FUNC_ID"), "{text}");
            assert!(!text.contains("S_DEFRANGE"), "{text}");
        }

        fn tool(names: &[&str]) -> Option<String> {
            names
                .iter()
                .find(|name| {
                    Command::new(name)
                        .arg("--version")
                        .output()
                        .is_ok_and(|out| out.status.success())
                })
                .map(|name| (*name).to_string())
        }
    }
}
