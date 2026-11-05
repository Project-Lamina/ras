//! Assemble what clang writes on macOS, link it with clang and run it.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::fs;
use std::process::Command;

mod common;
use common::TempDir;

fn exit_code(tag: &str, source: &str) -> i32 {
    let guard = TempDir::new(tag);
    let dir = guard.path();
    let input = dir.join("input.s");
    let object = dir.join("input.o");
    let exe = dir.join("input");
    fs::write(&input, source).expect("write assembly");
    let assemble = Command::new(env!("CARGO_BIN_EXE_ras"))
        .arg(&input)
        .arg("-o")
        .arg(&object)
        .output()
        .expect("run ras");
    assert!(
        assemble.status.success(),
        "{}",
        String::from_utf8_lossy(&assemble.stderr)
    );
    let link = Command::new("clang")
        .arg(&object)
        .arg("-o")
        .arg(&exe)
        .output()
        .expect("run clang");
    assert!(
        link.status.success(),
        "{}",
        String::from_utf8_lossy(&link.stderr)
    );
    Command::new(&exe)
        .status()
        .expect("run program")
        .code()
        .expect("exit code")
}

#[test]
fn darwin_hint_directives_and_w29_cfi_assemble() {
    let source = "\t.build_version macos, 27, 0\tsdk_version 27, 0\n\
        \t.section __TEXT,__text,regular,pure_instructions\n\
        \t.globl _main\n\
        \t.p2align 2\n\
        _main:\n\
        \t.cfi_startproc\n\
        \tstp x29, x30, [sp, #-16]!\n\
        \tmov x29, sp\n\
        \t.cfi_def_cfa w29, 16\n\
        Lloh0:\n\
        \tmov w0, #42\n\
        \tldp x29, x30, [sp], #16\n\
        \tret\n\
        \t.loh AdrpAdd Lloh0 Lloh0\n\
        \t.cfi_endproc\n\
        .subsections_via_symbols\n";
    assert_eq!(exit_code("ras-darwin-hints", source), 42);
}

#[test]
fn darwin_exception_tables_are_refused_by_name() {
    let guard = TempDir::new("ras-darwin-eh");
    let input = guard.path().join("input.s");
    fs::write(
        &input,
        "_main:\n.cfi_startproc\n.cfi_personality 155, ___gxx_personality_v0\nret\n.cfi_endproc\n",
    )
    .expect("write assembly");
    let assemble = Command::new(env!("CARGO_BIN_EXE_ras"))
        .arg(&input)
        .arg("-o")
        .arg(guard.path().join("input.o"))
        .output()
        .expect("run ras");
    assert!(!assemble.status.success());
    let stderr = String::from_utf8_lossy(&assemble.stderr);
    assert!(stderr.contains("Mach-O exception tables"), "{stderr}");
}

#[test]
fn darwin_semicolon_starts_a_comment() {
    let source = "\t.globl _main ; -- Begin function main\n\
        \t.p2align 2 ; -- Begin\n\
        _main: ; @main\n\
        \tmov w0, #7 ; =0x7\n\
        \tret\n";
    assert_eq!(exit_code("ras-darwin-semicolon", source), 7);
}

#[test]
fn darwin_literal_sections_link_and_load() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \t.globl _main\n\
        \t.p2align 2\n\
        _main:\n\
        \tadrp x8, lCPI@PAGE\n\
        \tldr x0, [x8, lCPI@PAGEOFF]\n\
        \tadrp x8, l_.str@PAGE\n\
        \tadd x8, x8, l_.str@PAGEOFF\n\
        \tldrb w8, [x8]\n\
        \tadd w0, w0, w8\n\
        \tret\n\
        \t.section __TEXT,__literal8,8byte_literals\n\
        \t.p2align 3\n\
        lCPI:\n\
        \t.quad 2\n\
        \t.section __TEXT,__cstring,cstring_literals\n\
        l_.str:\n\
        \t.asciz \"(\"\n";
    assert_eq!(exit_code("ras-darwin-literals", source), 42);
}

/// `otool -t` of the only section, as hex words.
fn text_words(object: &std::path::Path) -> String {
    let output = Command::new("otool")
        .arg("-t")
        .arg(object)
        .output()
        .expect("run otool");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .skip(2)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn register_offset_loads_match_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \tldr w0, [x9, w8, uxtw #2]\n\
        \tldr w0, [x9, w8, sxtw]\n\
        \tldr x1, [x2, x3]\n\
        \tldr x1, [x2, x3, lsl #3]\n\
        \tstr x1, [x2, x3, lsl #3]\n\
        \tldrb w1, [x2, x3]\n\
        \tldrh w1, [x2, x3, lsl #1]\n\
        \tldrsw x1, [x2, x3, sxtx #2]\n\
        \tstrb w1, [x2, w3, uxtw]\n\
        \tldr d0, [x2, x3, lsl #3]\n\
        \tldr s0, [x2, w3, sxtw #2]\n\
        \tldr q0, [x2, x3, lsl #4]\n\
        \tldr x1, [x2, #16]\n";
    assert_matches_clang("ras-register-offset", source);
}

#[test]
fn fmov_constants_match_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \tfmov d0, #1.00000000\n\
        \tfmov d1, #2.00000000\n\
        \tfmov s2, #-0.5\n\
        \tfmov d3, #31.0\n\
        \tfmov s4, #0.125\n";
    assert_matches_clang("ras-fmov-constant", source);
}

#[test]
fn dotted_neon_spelling_matches_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \tadd.4s v0, v1, v2\n\
        \tadd.16b v3, v4, v5\n\
        \tsub.2d v0, v1, v2\n\
        \tmul.8h v0, v1, v2\n\
        \tand.16b v0, v1, v2\n\
        \teor.8b v3, v4, v5\n\
        \tfmul.2s v0, v1, v2\n\
        \tfmul.2d v2, v3, v4\n\
        \tdup.4s v0, w12\n\
        \tdup.2d v2, x3\n\
        \tdup.16b v1, w0\n\
        \tmov.s v0[1], w10\n\
        \tmov.d v2[1], x3\n\
        \tmov.s w12, v0[3]\n";
    assert_matches_clang("ras-dotted-neon", source);
}

fn assert_matches_clang(tag: &str, source: &str) {
    let guard = TempDir::new(tag);
    let dir = guard.path();
    let input = dir.join("input.s");
    fs::write(&input, source).expect("write assembly");
    let ras_object = dir.join("ras.o");
    let clang_object = dir.join("clang.o");
    let assemble = Command::new(env!("CARGO_BIN_EXE_ras"))
        .arg(&input)
        .arg("-o")
        .arg(&ras_object)
        .output()
        .expect("run ras");
    assert!(
        assemble.status.success(),
        "{}",
        String::from_utf8_lossy(&assemble.stderr)
    );
    let reference = Command::new("clang")
        .args(["-c", "-x", "assembler"])
        .arg(&input)
        .arg("-o")
        .arg(&clang_object)
        .output()
        .expect("run clang");
    assert!(reference.status.success());
    assert_eq!(text_words(&ras_object), text_words(&clang_object));
}

#[test]
fn zerofill_defines_zeroed_symbol_and_returns_to_section() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        .zerofill __DATA,__bss,_slot,16,3\n\
        \t.globl _main\n\
        \t.p2align 2\n\
        _main:\n\
        \tadrp x8, _slot@PAGE\n\
        \tldr x0, [x8, _slot@PAGEOFF]\n\
        \tadd x0, x0, #9\n\
        \tret\n";
    assert_eq!(exit_code("ras-zerofill", source), 9);
}

#[test]
fn assignment_names_an_absolute_value() {
    let source = "SIZE = 42\n\
        \t.globl _main\n\
        \t.p2align 2\n\
        _main:\n\
        \tadrp x8, _value@PAGE\n\
        \tldr w0, [x8, _value@PAGEOFF]\n\
        \tret\n\
        \t.section __DATA,__data\n\
        \t.p2align 2\n\
        _value:\n\
        \t.long SIZE\n";
    assert_eq!(exit_code("ras-assignment", source), 42);
}
