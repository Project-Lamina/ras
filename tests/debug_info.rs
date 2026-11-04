//! Function symbols and `.debug_info` subprograms, read back with readelf.
//!
//! The assertions are questions a debugger asks: which functions exist, where each
//! one starts and ends, what it returns and what its parameters are called. None of
//! them compares bytes with another toolchain's output.
#![cfg(target_os = "linux")]

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::{
    BaseType, DebugInfoRequest, DebugType, DeclSite, FormalParameter, ObjectWriteOptions, Ras,
    Subprogram,
};
use std::path::Path;
use std::process::Command;

mod common;
use common::TempDir;

/// `twice` is followed by `.p2align 4`, so the padding between it and `half` belongs
/// to neither function and must not land in either one's size.
const SOURCE: &str = ".file 1 \"shapes.lamina\"\n\
.text\n\
.globl twice\n\
twice:\n\
.loc 1 1 1\n\
addl %edi, %edi\n\
movl %edi, %eax\n\
ret\n\
.p2align 4\n\
half:\n\
.loc 1 6 1\n\
movl %edi, %eax\n\
ret\n";

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

fn assemble(path: &Path, options: ObjectWriteOptions) {
    Ras::with_object_write_options(
        TargetArchitecture::X86_64,
        TargetOperatingSystem::Linux,
        options,
    )
    .expect("assembler")
    .assemble(SOURCE, path)
    .expect("assemble");
}

fn readelf(path: &Path, switch: &str) -> String {
    let output = Command::new("readelf")
        .arg(switch)
        .arg(path)
        .output()
        .expect("readelf");
    String::from_utf8(output.stdout).expect("readelf output is utf-8")
}

/// `(type, size)` of a symbol in the symbol table.
fn symbol(dump: &str, name: &str) -> Option<(String, u64)> {
    dump.lines().find_map(|line| {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 8 || fields.last() != Some(&name) {
            return None;
        }
        Some((fields[3].to_string(), fields[2].parse().ok()?))
    })
}

#[test]
fn named_functions_are_sized_without_the_padding_that_follows_them() {
    let dir = TempDir::new("debug_info_symbols");
    let path = dir.path().join("out.o");
    assemble(
        &path,
        ObjectWriteOptions {
            function_symbols: vec!["twice".into(), "half".into()],
            ..ObjectWriteOptions::default()
        },
    );
    let dump = readelf(&path, "-sW");

    // addl + movl + ret, and none of the alignment padding after them.
    assert_eq!(symbol(&dump, "twice"), Some(("FUNC".into(), 5)));
    assert_eq!(symbol(&dump, "half"), Some(("FUNC".into(), 3)));
}

#[test]
fn a_symbol_not_named_as_a_function_keeps_its_type() {
    let dir = TempDir::new("debug_info_untyped");
    let path = dir.path().join("out.o");
    assemble(&path, ObjectWriteOptions::default());
    let dump = readelf(&path, "-sW");

    assert_eq!(symbol(&dump, "twice"), Some(("NOTYPE".into(), 0)));
}

#[test]
fn subprograms_carry_their_range_declaration_and_parameters() {
    let dir = TempDir::new("debug_info_dies");
    let path = dir.path().join("out.o");
    assemble(
        &path,
        ObjectWriteOptions {
            function_symbols: vec!["twice".into(), "half".into()],
            debug_info: Some(request()),
            ..ObjectWriteOptions::default()
        },
    );
    let dump = readelf(&path, "--debug-dump=info");

    assert_eq!(dump.matches("DW_TAG_subprogram").count(), 2, "{dump}");
    assert!(!dump.contains("absent"), "{dump}");
    assert_eq!(dump.matches("DW_TAG_base_type").count(), 1, "{dump}");
    assert_eq!(dump.matches("DW_TAG_formal_parameter").count(), 2, "{dump}");

    // The externally visible one says so; the other must not.
    assert_eq!(dump.matches("DW_AT_external").count(), 1, "{dump}");

    for (name, line, length) in [("twice", "1", "0x5"), ("half", "6", "0x3")] {
        let die = dump
            .split("DW_TAG_subprogram")
            .find(|part| part.contains(&format!("DW_AT_name        : {name}")))
            .unwrap_or_else(|| panic!("no subprogram for {name} in {dump}"));
        assert!(
            die.contains(&format!("DW_AT_decl_line   : {line}")),
            "{die}"
        );
        assert!(
            die.contains(&format!("DW_AT_high_pc     : {length}")),
            "{die}"
        );
        assert!(die.contains("DW_OP_call_frame_cfa"), "{die}");
        assert!(die.contains("DW_AT_name        : n"), "{die}");
    }
}

/// A parameter whose location the caller could not supply must carry no location at
/// all. A debugger reads the absence as "optimized out"; a guess would make it print
/// a wrong value instead.
#[test]
fn a_parameter_without_a_known_location_is_given_none() {
    let dir = TempDir::new("debug_info_no_location");
    let path = dir.path().join("out.o");
    assemble(
        &path,
        ObjectWriteOptions {
            function_symbols: vec!["twice".into(), "half".into()],
            debug_info: Some(request()),
            ..ObjectWriteOptions::default()
        },
    );
    let dump = readelf(&path, "--debug-dump=info");

    assert!(!dump.contains("DW_AT_location"), "{dump}");
}

#[test]
fn the_tree_passes_the_dwarf_verifier() {
    let Ok(verifier) = which_dwarfdump() else {
        return;
    };
    let dir = TempDir::new("debug_info_verify");
    let path = dir.path().join("out.o");
    assemble(
        &path,
        ObjectWriteOptions {
            function_symbols: vec!["twice".into(), "half".into()],
            debug_info: Some(request()),
            ..ObjectWriteOptions::default()
        },
    );
    let output = Command::new(verifier)
        .arg("--verify")
        .arg(&path)
        .output()
        .expect("llvm-dwarfdump");
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("No errors."), "{text}");
}

fn which_dwarfdump() -> Result<String, ()> {
    for name in ["llvm-dwarfdump", "llvm-dwarfdump-18", "llvm-dwarfdump-19"] {
        let found = Command::new(name)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success());
        if found {
            return Ok(name.to_string());
        }
    }
    Err(())
}
