//! These compare ras against GNU binutils and clang on ELF objects, so they
//! only run where both the reference assembler and the format are available.
#![cfg(target_os = "linux")]

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;
use std::process::Command;

fn text(bytes: &[u8]) -> &[u8] {
    let u64_at = |at| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
    let headers = u64_at(40);
    let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
    let strings = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;
    let names = &bytes[u64_at(headers + strings * 64 + 24)..];
    for i in 1..count {
        let h = headers + i * 64;
        let name = u32::from_le_bytes(bytes[h..h + 4].try_into().unwrap()) as usize;
        if names[name..].starts_with(b".text\0") {
            return &bytes[u64_at(h + 24)..u64_at(h + 24) + u64_at(h + 32)];
        }
    }
    panic!("Missing .text")
}

/// Run the RV64I subset that `li` expands into and report `a0`.
fn evaluate(text: &[u8]) -> i64 {
    let mut regs = [0i64; 32];
    for word in text.chunks_exact(4) {
        let word = u32::from_le_bytes(word.try_into().unwrap());
        let rd = ((word >> 7) & 31) as usize;
        let rs1 = regs[((word >> 15) & 31) as usize];
        let shamt = (word >> 20) & 63;
        let imm = i64::from((word as i32) >> 20);
        let value = match (word & 0x7f, (word >> 12) & 7) {
            (0x37, _) => i64::from((word & 0xffff_f000) as i32),
            (0x13, 0) => rs1.wrapping_add(imm),
            (0x13, 1) => ((rs1 as u64) << shamt) as i64,
            (0x13, 5) => ((rs1 as u64) >> shamt) as i64,
            (0x13, 6) => rs1 | imm,
            (0x1b, 0) => i64::from(rs1.wrapping_add(imm) as i32),
            (0x33, 0) => rs1.wrapping_add(regs[((word >> 20) & 31) as usize]),
            _ => panic!("Unexpected instruction in li expansion: {word:#010x}"),
        };
        if rd != 0 {
            regs[rd] = value;
        }
    }
    regs[10]
}

#[test]
fn li_materializes_every_boundary_immediate() {
    let cases: &[(&str, i64)] = &[
        ("0", 0),
        ("1", 1),
        ("-1", -1),
        ("2047", 2047),
        ("2048", 2048),
        ("-2048", -2048),
        ("-2049", -2049),
        ("4095", 4095),
        ("4096", 4096),
        ("-4096", -4096),
        ("0x7ff", 0x7ff),
        ("0x800", 0x800),
        ("0xfff", 0xfff),
        ("0x1000", 0x1000),
        ("0x1800", 0x1800),
        ("0x7fffffff", 0x7fff_ffff),
        ("-0x80000000", -0x8000_0000),
        ("0x80000000", 0x8000_0000),
        ("0xffffffff", 0xffff_ffff),
        ("0x100000000", 0x1_0000_0000),
        ("0x123456789abcdef0", 0x1234_5678_9abc_def0),
        ("-9223372036854775808", i64::MIN),
        ("9223372036854775807", i64::MAX),
    ];
    for (literal, expected) in cases {
        let object = assemble_riscv("li_boundary", &format!(".text\nli a0, {literal}\n"));
        assert_eq!(evaluate(text(&object)), *expected, "li a0, {literal}");
    }
}

/// (offset, relocation type, symbol name) for every RELA entry in an object file.
fn relocations(bytes: &[u8]) -> Vec<(u64, u32, String)> {
    let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let shoff = u64_at(40) as usize;
    let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
    let section = |i: usize| shoff + i * 64;
    let mut out = Vec::new();
    for i in 0..count {
        let header = section(i);
        if u32_at(header + 4) != 4 {
            continue;
        }
        let symtab = section(u32_at(header + 40) as usize);
        let symbols = u64_at(symtab + 24) as usize;
        let strings = u64_at(section(u32_at(symtab + 40) as usize) + 24) as usize;
        let entries = u64_at(header + 24) as usize;
        for entry in (entries..entries + u64_at(header + 32) as usize).step_by(24) {
            let info = u64_at(entry + 8);
            let name_at = strings + u32_at(symbols + (info >> 32) as usize * 24) as usize;
            let name_len = bytes[name_at..].iter().position(|&b| b == 0).unwrap();
            let name = String::from_utf8_lossy(&bytes[name_at..name_at + name_len]).into_owned();
            out.push((u64_at(entry), info as u32, name));
        }
    }
    out
}

fn assemble_riscv(name: &str, asm: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("ras_riscv_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let output = dir.join("ras.o");
    Ras::new(TargetArchitecture::Riscv64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(asm, &output)
        .unwrap();
    let bytes = std::fs::read(&output).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    bytes
}

fn clang_riscv(name: &str, asm: &str, extra: &[&str]) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("ras_riscv_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("input.s");
    let reference = dir.join("reference.o");
    std::fs::write(&source, asm).unwrap();
    let result = Command::new("clang")
        .args(["--target=riscv64-linux-gnu", "-march=rv64g", "-c"])
        .args(extra)
        .arg(&source)
        .arg("-o")
        .arg(&reference)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&reference).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    bytes
}

/// R_RISCV_RELAX hints only matter to linker relaxation, which ras does not emit.
fn without_relax(relocs: Vec<(u64, u32, String)>) -> Vec<(u64, u32, String)> {
    relocs
        .into_iter()
        .filter(|(_, kind, _)| *kind != 51)
        .collect()
}

#[test]
fn call_and_tail_to_external_symbols_match_clang_with_call_plt() {
    let asm = ".text\n.globl _start\n_start:\n  call foo\n  tail foo\n";
    let reference = clang_riscv("call_oracle", asm, &[]);
    let ras = assemble_riscv("call_ras", asm);
    assert_eq!(text(&ras), text(&reference));
    assert_eq!(
        without_relax(relocations(&ras)),
        without_relax(relocations(&reference))
    );
}

#[test]
fn la_to_external_symbol_matches_clang_non_pic_pcrel_pair() {
    let asm = ".text\n.globl _start\n_start:\n  la a0, foo\n";
    let reference = clang_riscv("la_oracle", asm, &["-fno-pic"]);
    let ras = assemble_riscv("la_ras", asm);
    assert_eq!(text(&ras), text(&reference));
    let kinds = |bytes: &[u8]| {
        without_relax(relocations(bytes))
            .into_iter()
            .map(|(offset, kind, _)| (offset, kind))
            .collect::<Vec<_>>()
    };
    assert_eq!(kinds(&ras), kinds(&reference));
}

#[test]
fn local_call_tail_and_la_land_on_their_label_across_the_twelve_bit_boundary() {
    let pc_relative = |pc: i64, auipc: u32, second: u32| {
        let hi = (auipc & !0xFFF) as i32 as i64;
        let lo = (second as i32 >> 20) as i64;
        pc + hi + lo
    };
    for skip in [0usize, 2044, 2046, 2047, 2048, 2049, 4094, 4096] {
        let asm = format!(
            ".text\n.globl _start\n_start:\n  call local\n  tail local\n  la a0, local\n  .skip {skip}\nlocal:\n  ret\n"
        );
        let bytes = assemble_riscv("local_pairs", &asm);
        let code = text(&bytes);
        let word = |at: usize| u32::from_le_bytes(code[at..at + 4].try_into().unwrap());
        let target = 24 + skip as i64;
        for pc in [0i64, 8, 16] {
            let at = pc as usize;
            assert_eq!(
                pc_relative(pc, word(at), word(at + 4)),
                target,
                "skip {skip} pair at {pc}"
            );
        }
    }
}
