//! CodeView `.debug$S` and `.debug$T` in a COFF object, read back from the bytes.
//!
//! The assertions are questions a debugger asks: which functions exist, how long each
//! one is, what it returns, what its parameters are called, and which source line each
//! instruction came from. None of them compares bytes with another toolchain's output.

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::{
    BaseType, DebugInfoRequest, DebugType, DeclSite, FormalParameter, ObjectWriteOptions, Ras,
    Subprogram,
};
use std::process::Command;

mod common;
use common::TempDir;

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

fn object() -> Vec<u8> {
    Ras::with_object_write_options(
        TargetArchitecture::X86_64,
        TargetOperatingSystem::Windows,
        ObjectWriteOptions {
            function_symbols: vec!["twice".into(), "half".into()],
            debug_info: Some(request()),
            debug_object_name: "shapes.obj".into(),
            ..ObjectWriteOptions::default()
        },
    )
    .expect("assembler")
    .assemble_to_bytes(SOURCE)
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
        Self { bytes: object() }
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
        path.starts_with('/'),
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
    let Some(pdbutil) = tool(&["llvm-pdbutil", "llvm-pdbutil-18", "llvm-pdbutil-19"]) else {
        return;
    };
    let dir = TempDir::new("codeview_pdbutil");
    let path = dir.path().join("shapes.obj");
    std::fs::write(&path, object()).expect("write");
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
