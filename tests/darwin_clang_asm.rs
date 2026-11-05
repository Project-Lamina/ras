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

#[test]
fn neon_float_and_logic_match_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \tfadd.4s v0, v1, v2\n\
        \tfadd.2d v3, v4, v5\n\
        \tfsub.2s v6, v7, v8\n\
        \tfsub.2d v9, v10, v11\n\
        \tfdiv.2s v12, v13, v14\n\
        \tfdiv.4s v15, v16, v17\n\
        \tfdiv.2d v18, v19, v20\n\
        \torr.16b v0, v1, v2\n\
        \torr.8b v3, v4, v5\n\
        \tmov.16b v6, v7\n\
        \tmov.8b v8, v9\n\
        \tfmla.4s v0, v1, v2\n\
        \tfmla.2d v3, v4, v5\n\
        \tfmls.2s v6, v7, v8\n\
        \tfmls.2d v9, v10, v11\n";
    assert_matches_clang("ras-neon-float", source);
}

#[test]
fn neon_reductions_match_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \taddv.8b b0, v1\n\
        \taddv.16b b2, v3\n\
        \taddv.4h h4, v5\n\
        \taddv.8h h6, v7\n\
        \taddv.4s s8, v9\n\
        \taddv s10, v11.4s\n\
        \tfaddp.4s v0, v1, v2\n\
        \tfaddp.2s v3, v4, v5\n\
        \tfaddp.2d v6, v7, v8\n\
        \tfaddp.2s s9, v10\n\
        \tfaddp d11, v12.2d\n";
    assert_matches_clang("ras-neon-reduce", source);
}

#[test]
fn neon_ext_matches_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \text.16b v0, v1, v2, #8\n\
        \text.8b v3, v4, v5, #3\n\
        \text.16b v6, v7, v8, #0\n\
        \text.16b v9, v10, v11, #15\n\
        \text.8b v12, v13, v14, #7\n";
    assert_matches_clang("ras-neon-ext", source);
}

#[test]
fn neon_fmov_high_doubleword_matches_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \tfmov.d x0, v1[1]\n\
        \tfmov.d v2[1], x3\n\
        \tfmov x4, v5.d[1]\n\
        \tfmov v6.d[1], x7\n";
    assert_matches_clang("ras-neon-fmov", source);
}

#[test]
fn neon_movi_matches_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \tmovi.2d v0, #0\n\
        \tmovi.16b v1, #0\n\
        \tmovi.8b v2, #0xff\n\
        \tmovi.4s v3, #1\n\
        \tmovi.4s v4, #0x1, lsl #8\n\
        \tmovi.4s v5, #0x3, lsl #16\n\
        \tmovi.2s v6, #0x3, lsl #24\n\
        \tmovi.8h v7, #0x2, lsl #8\n\
        \tmovi.4h v8, #0x2\n\
        \tmovi.2d v9, #0xff00ff00ff00ff00\n\
        \tmovi.2d v10, #0xffffffffffffffff\n\
        \tmovi.2d v11, #0x00ff00ff00ff00ff\n\
        \tmovi.16b v12, #0x42\n\
        \tmovi.4s v13, #0\n";
    assert_matches_clang("ras-neon-movi", source);
}

#[test]
fn neon_structure_loads_and_stores_match_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \tld1.4s { v0 }, [x0]\n\
        \tld1.4s { v1, v2 }, [x2]\n\
        \tld1.16b { v3, v4, v5 }, [x1]\n\
        \tld1.8h { v6, v7, v8, v9 }, [x3]\n\
        \tld1.2d { v10 }, [sp]\n\
        \tld1.8b { v11 }, [x12]\n\
        \tst1.2s { v13 }, [x4]\n\
        \tst1.4h { v31 }, [sp]\n\
        \tst1.2d { v14, v15 }, [x5]\n\
        \tst1.16b { v16, v17, v18, v19 }, [x6]\n\
        \tld1.4s { v31, v0 }, [x7]\n";
    assert_matches_clang("ras-neon-ld1", source);
}

#[test]
fn neon_dup_element_matches_clang_bytes() {
    let source = "\t.section __TEXT,__text,regular,pure_instructions\n\
        \tdup.4s v0, v1[2]\n\
        \tdup.2d v2, v3[1]\n\
        \tdup.8b v4, v5[7]\n\
        \tdup.16b v6, v7[15]\n\
        \tdup.8h v8, v9[3]\n\
        \tdup.4h v10, v11[0]\n\
        \tdup.2s v12, v13[1]\n";
    assert_matches_clang("ras-neon-dup", source);
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
