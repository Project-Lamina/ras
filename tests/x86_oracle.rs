//! These compare ras against GNU binutils and clang on ELF objects, so they
//! only run where both the reference assembler and the format are available.
#![cfg(target_os = "linux")]

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;
use std::process::Command;

mod common;
use common::TempDir;

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

/// Length of the nop starting at `bytes`, covering the `66` and `cs` prefixed
/// forms GNU as pads with, or `None` when it does not start with one.
fn nop_length(bytes: &[u8]) -> Option<usize> {
    let mut at = 0;
    while bytes.get(at) == Some(&0x66) {
        at += 1;
    }
    if bytes.get(at) == Some(&0x2e) {
        at += 1;
    }
    match bytes.get(at)? {
        0x90 => Some(at + 1),
        0x0f if bytes.get(at + 1) == Some(&0x1f) => {
            let modrm = *bytes.get(at + 2)?;
            if (modrm >> 3) & 7 != 0 {
                return None;
            }
            let (mode, rm) = (modrm >> 6, modrm & 7);
            let mut length = at + 3 + usize::from(mode != 3 && rm == 4);
            length += match mode {
                1 => 1,
                2 => 4,
                0 if rm == 5 => 4,
                _ => 0,
            };
            (length <= bytes.len()).then_some(length)
        }
        _ => None,
    }
}

/// Rewrite every nop as that many one-byte nops. Which forms an assembler pads
/// with, and in what order, differs between binutils releases; where the
/// padding starts and ends does not, and that is what these comparisons are
/// about.
fn flatten_nops(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    let mut at = 0;
    while at < text.len() {
        match nop_length(&text[at..]) {
            Some(length) => {
                out.resize(out.len() + length, 0x90);
                at += length;
            }
            None => {
                out.push(text[at]);
                at += 1;
            }
        }
    }
    out
}

#[test]
fn padding_comparison_ignores_nop_form() {
    let long_first = [
        0xc3, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0, 0x0f, 0x1f, 0x40, 0x00, 0xc3,
    ];
    let short_first = [
        0xc3, 0x0f, 0x1f, 0x40, 0x00, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0, 0xc3,
    ];
    assert_eq!(flatten_nops(&long_first), flatten_nops(&short_first));
    let shorter = [0xc3, 0x0f, 0x1f, 0x40, 0x00, 0xc3];
    assert_ne!(flatten_nops(&long_first), flatten_nops(&shorter));
}

#[test]
fn move_widths_extensions_and_scalar_operations_match_gas() {
    let tmp = TempDir::new("x86_oracle");
    let dir = tmp.path();
    let source = dir.join("input.s");
    let reference = dir.join("reference.o");
    let output = dir.join("ras.o");
    let mut asm = String::from(".text\nendbr64\n");
    for (width, reg) in [("b", "%al"), ("w", "%ax"), ("l", "%eax"), ("q", "%rax")] {
        for base in ["%rsp", "%rbp", "%r12", "%r13", "%r11"] {
            for offset in [0, 8, -8, 256, -256] {
                asm.push_str(&format!("mov{width} {reg}, {offset}({base})\nmov{width} {offset}({base}), {reg}\nmov{width} $42, {offset}({base})\n"));
            }
        }
    }
    asm.push_str("movl %r9d, %r10d\nmovw %r9w, %r10w\nmovb %r9b, %r10b\nmovb %ah, %cl\nmovb %spl, %dil\nmovq $42, %rax\nmovq $4294967295, %rax\nmovq $-42, %r9\nmovzbl (%rsp), %ecx\nmovzbq (%rax), %r9\nmovzwq (%r13), %r10\nmovzwl 8(%r12), %ecx\ncmovel %eax, %ecx\ncmovel %r9d, %r10d\nmulss %xmm1, %xmm0\nmulsd %xmm10, %xmm9\nsqrtsd %xmm1, %xmm0\nsqrtss %xmm11, %xmm12\nminsd %xmm2, %xmm3\nminss %xmm4, %xmm5\nmaxsd %xmm13, %xmm14\nmaxss %xmm6, %xmm7\nmovq $0x8000000000000000, %rax\nret\n");
    asm.push_str(".globl callee\ncallee: call callee\ncall .Llocal\n.Llocal: ret\n");
    for padding in 1..32 {
        asm.push_str(".p2align 5\n");
        for _ in 0..32 - padding {
            asm.push_str("ret\n");
        }
        asm.push_str(".p2align 5\n");
    }
    std::fs::write(&source, &asm).unwrap();
    let result = Command::new("as")
        .arg("--64")
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
    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(&asm, &output)
        .unwrap();
    assert_eq!(
        flatten_nops(text(&std::fs::read(&output).unwrap())),
        flatten_nops(text(&std::fs::read(&reference).unwrap()))
    );
}

#[test]
fn memory_move_rejects_an_unrepresentable_immediate() {
    let tmp = TempDir::new("invalid_immediate");
    let path = tmp.path().join("invalid_immediate.o");
    assert!(
        Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .unwrap()
            .assemble("movq $4294967295, (%rax)\n", &path)
            .is_err()
    );
}

fn oracle(asm: &str, name: &str) -> (Vec<u8>, Vec<u8>) {
    let tmp = TempDir::new(&format!("x86_{name}"));
    let dir = tmp.path();
    let source = dir.join("input.s");
    let reference = dir.join("reference.o");
    let output = dir.join("ras.o");
    std::fs::write(&source, asm).unwrap();
    let result = Command::new("as")
        .args(["--64", "-o"])
        .arg(&reference)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(asm, &output)
        .unwrap();
    let bytes = (
        std::fs::read(output).unwrap(),
        std::fs::read(reference).unwrap(),
    );
    bytes
}

fn section<'a>(bytes: &'a [u8], name: &str) -> &'a [u8] {
    let at = |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    let headers = at(40);
    let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
    let strings = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;
    let names = &bytes[at(headers + strings * 64 + 24)..];
    for i in 1..count {
        let h = headers + i * 64;
        let n = u32::from_le_bytes(bytes[h..h + 4].try_into().unwrap()) as usize;
        if names[n..].split(|b| *b == 0).next().unwrap() == name.as_bytes() {
            return &bytes[at(h + 24)..at(h + 24) + at(h + 32)];
        }
    }
    panic!("Missing section {name}")
}

fn relocations(bytes: &[u8], name: &str) -> Vec<(u64, u32, String, i64)> {
    let at =
        |bytes: &[u8], offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    let symbols = section(bytes, ".symtab");
    let strings = section(bytes, ".strtab");
    section(bytes, name)
        .chunks_exact(24)
        .map(|entry| {
            let info = at(entry, 8);
            let symbol = &symbols[(info >> 32) as usize * 24..];
            let offset = u32::from_le_bytes(symbol[..4].try_into().unwrap()) as usize;
            let name = strings[offset..].split(|b| *b == 0).next().unwrap();
            (
                at(entry, 0),
                info as u32,
                String::from_utf8(name.to_vec()).unwrap(),
                at(entry, 16) as i64,
            )
        })
        .collect()
}

#[test]
fn gcc_instruction_forms_match_gas() {
    let (output, reference) = oracle(include_str!("fixtures/x86_gcc.s"), "gcc_forms");
    assert_eq!(text(&output), text(&reference));
    assert_eq!(
        relocations(&output, ".rela.text"),
        relocations(&reference, ".rela.text")
    );
}

#[test]
fn truncating_float_to_integer_conversions_match_gas() {
    let asm = ".text\ncvttss2sil %xmm0, %eax\ncvttss2siq %xmm9, %r10\ncvttss2siq 8(%rsp), %rax\ncvttss2sil -4(%r13), %r9d\ncvttsd2siq %xmm1, %rcx\ncvttsd2sil 16(%r12), %edx\nret\n";
    let (output, reference) = oracle(asm, "cvtt_to_int");
    assert_eq!(text(&output), text(&reference));
}

#[test]
fn scalar_float_comparisons_match_gas() {
    let asm = ".text\nucomiss %xmm0, %xmm1\nucomiss %xmm15, %xmm3\ncomiss %xmm2, %xmm7\nucomiss (%rax), %xmm0\ncomiss 8(%rbp,%rcx,4), %xmm9\nucomisd %xmm0, %xmm1\ncomisd %xmm4, %xmm5\nret\n";
    let (output, reference) = oracle(asm, "scalar_cmp");
    assert_eq!(text(&output), text(&reference));
}

#[test]
fn float_width_conversions_match_gas() {
    let asm = ".text\ncvtsd2ss %xmm1, %xmm0\ncvtsd2ss %xmm9, %xmm10\ncvtsd2ss (%rsp), %xmm1\ncvtsd2ss 8(%r13), %xmm2\ncvtss2sd %xmm9, %xmm1\ncvtss2sd -4(%rbx), %xmm15\nret\n";
    let (output, reference) = oracle(asm, "float_width");
    assert_eq!(text(&output), text(&reference));
}

#[test]
fn integer_to_single_conversions_match_gas() {
    let asm = ".text\ncvtsi2ssl %eax, %xmm0\ncvtsi2ssl %r9d, %xmm10\ncvtsi2ssl (%rsp), %xmm1\ncvtsi2ssl -12(%r12), %xmm7\ncvtsi2ssq %r9, %xmm10\ncvtsi2ssq 8(%rbp), %xmm2\nret\n";
    let (output, reference) = oracle(asm, "int_to_single");
    assert_eq!(text(&output), text(&reference));
}

#[test]
fn ret_with_stack_adjustment_matches_gas() {
    let asm = ".text\nret $5\nret $0x10\nret $65535\nret\nret $0\n";
    let (output, reference) = oracle(asm, "ret_imm");
    assert_eq!(text(&output), text(&reference));
}

#[test]
fn movabs_immediates_match_gas() {
    let asm = ".text\nmovabs $0x123456789abcdef0, %rcx\nmovabs $1, %rax\nmovabs $-2147483649, %r12\nmovabsq $9223372036854775807, %r9\nret\n";
    let (output, reference) = oracle(asm, "movabs");
    assert_eq!(text(&output), text(&reference));
}

#[test]
fn absolute_symbols_in_data_directives_match_gas() {
    let asm = ".text\n.byte foo\n.long foo\n.quad later\n.byte later+1\n.short bar-2\n.set foo, 5\n.equ bar, 0x20\n.set later, 9\n.set alias, foo\n.byte alias\n.set alias2, alias+3\n.quad alias2\nret\n";
    let (output, reference) = oracle(asm, "abs_data");
    assert_eq!(text(&output), text(&reference));
}

#[test]
fn direct_branches_to_symbols_match_gas() {
    let asm = ".text\ncall foo@PLT\njmp bar+8\nje baz-4\njne .Lend\ncall foo\n.Lend: ret\n";
    let (output, reference) = oracle(asm, "direct_branch");
    assert_eq!(text(&output), text(&reference));
}

#[test]
fn rip_addends_and_prefixed_fixups_match_gas() {
    let asm = ".text\n.globl entry\nentry: movq 120+external(%rip), %rax\nmovq external+120(%rip), %r9\ncmpw $3, -8+external(%rip)\nmovq 8+.Llocal(%rip), %rax\n.Llocal: ret\nleaq 4+entry(%rip), %r9\nlock addw $1, external(%rip)\nnotrack jmp *external(%rip)\npshufd $3, 16+external(%rip), %xmm9\nlock addw $1, %fs:8(%rax)\n";
    let (output, reference) = oracle(asm, "rip_addends");
    assert_eq!(text(&output), text(&reference));
    assert_eq!(
        relocations(&output, ".rela.text"),
        relocations(&reference, ".rela.text")
    );
    let entries = relocations(&output, ".rela.text");
    assert_eq!(entries[0].3, 116);
    assert_eq!(entries[2].3, -13);
}

#[test]
fn alignment_sensitive_branches_match_gas() {
    let mut asm = String::from(".text\n");
    for i in 100..160 {
        asm.push_str(&format!(".p2align 4\n.Lstart{i}: je .Lend{i}\n.space {i}, 0x90\njne .Lstart{i}\n.p2align 4,,10\n.p2align 3\njmp .Lend{i}\n.space 8, 0x90\n.Lend{i}: ret\n"));
    }
    let (output, reference) = oracle(&asm, "branch_alignment");
    assert_eq!(flatten_nops(text(&output)), flatten_nops(text(&reference)));
}

#[test]
fn zero_max_skip_aligns_like_gas() {
    let asm = ".data\n.byte 1\n.p2align 4,,0\n.byte 2\n.p2align 4,,4\n.byte 3\n";
    let (output, reference) = oracle(asm, "zero_max_skip");
    assert_eq!(section(&output, ".data"), section(&reference, ".data"));
}

#[test]
fn compiler_string_escapes_match_gas() {
    let asm = r#".text
.ascii "\001\012\123\377\0x\b\f\v\a\n\t\r\\\""
.asciz "\0007\01a\1234"
"#;
    let (output, reference) = oracle(asm, "string_escapes");
    assert_eq!(text(&output), text(&reference));
}

#[test]
fn invalid_compiler_operands_fail() {
    let tmp = TempDir::new("invalid_compiler");
    let path = tmp.path().join("invalid_compiler.o");
    for asm in [
        "cmpw %eax, %ax",
        "movd %ax, %xmm0",
        "mov %rax, $1",
        "mov %rax, %xmm0",
        "mov (%rax), %xmm0",
        "movswl %al, %eax",
        "movsbl %ecx, %eax",
        "movzwq %r9d, %rax",
        "movhps %xmm0, %xmm1",
        "pshufd $256, %xmm0, %xmm1",
        "cvtsi2sdq %eax, %xmm0",
        "ret $65536",
        "ret %rax",
        "endbr64 %rax",
        "leave %rax",
        "cltq %eax",
        "cqto %rax",
        "syscall $1",
        "ud2 %eax",
        "call *%eax",
        "jmp *%r9d",
        ".p2align 4,,,",
        "call %rax",
        "jmp 8(%rax)",
        "je %rax",
        "call foo bar",
        "call $5",
        ".set x y, 5",
        ".type foo bar, @function",
        ".size foo bar, 4",
        ".comm foo bar, 8",
        ".local foo bar",
        ".weak foo bar",
        "notrack jmp %rax",
        "lock cmpq $1, (%rax)",
        "rep addq %rax, %rbx",
        "movq external+other(%rip), %rax",
        "movq 2147483648+external(%rip), %rax",
    ] {
        assert!(
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(asm, &path)
                .is_err(),
            "{asm}"
        );
    }
}

#[test]
fn cross_section_branch_relocation_kinds_match_gas() {
    let asm = ".text\ncall .Lcold\njmp .Lcold\ncall cold_global\n.section .text.unlikely,\"ax\",@progbits\n.Lcold: ret\n.globl cold_global\ncold_global: ret\n";
    let (output, reference) = oracle(asm, "cold_branches");
    assert_eq!(text(&output), text(&reference));
    let fields = |bytes: &[u8]| {
        let mut entries = relocations(bytes, ".rela.text")
            .into_iter()
            .map(|(offset, kind, _, addend)| (offset, kind, addend))
            .collect::<Vec<_>>();
        entries.sort();
        entries
    };
    assert_eq!(fields(&output), fields(&reference));
    assert_eq!(
        fields(&output)
            .iter()
            .map(|entry| entry.1)
            .collect::<Vec<_>>(),
        [2, 2, 4]
    );
}

/// Every packed-SSE form the x86_64 backend can emit, against GNU as.
#[test]
fn packed_sse_forms_match_gas() {
    const BINARY: &[&str] = &[
        "paddb",
        "paddw",
        "paddd",
        "paddq",
        "psubb",
        "psubw",
        "psubd",
        "psubq",
        "pmullw",
        "pmuludq",
        "pmulld",
        "pand",
        "pandn",
        "por",
        "pxor",
        "psllw",
        "pslld",
        "psllq",
        "psrlw",
        "psrld",
        "psrlq",
        "psraw",
        "psrad",
        "punpcklbw",
        "punpcklwd",
        "punpckldq",
        "punpcklqdq",
        "punpckhbw",
        "punpckhwd",
        "punpckhdq",
        "punpckhqdq",
        "pcmpeqb",
        "pcmpeqw",
        "pcmpeqd",
        "pcmpgtb",
        "pcmpgtw",
        "pcmpgtd",
        "addps",
        "subps",
        "mulps",
        "divps",
        "minps",
        "maxps",
        "sqrtps",
        "andps",
        "andnps",
        "orps",
        "xorps",
        "unpcklps",
        "unpckhps",
        "addpd",
        "subpd",
        "mulpd",
        "divpd",
        "minpd",
        "maxpd",
        "sqrtpd",
        "andpd",
        "andnpd",
        "orpd",
        "xorpd",
        "unpcklpd",
        "unpckhpd",
    ];
    const SOURCES: &[&str] = &[
        "%xmm0",
        "%xmm7",
        "%xmm8",
        "%xmm15",
        "(%rax)",
        "(%r12)",
        "(%r13)",
        "(%rsp)",
        "(%rbp)",
        "1024(%rbx)",
        "-128(%r14)",
        "(%rax,%r9,8)",
        "(%r12,%r13,2)",
    ];
    const DESTINATIONS: &[&str] = &["%xmm0", "%xmm6", "%xmm8", "%xmm14"];
    let mut asm = String::from(".text\n");
    for mnemonic in BINARY {
        for source in SOURCES {
            for destination in DESTINATIONS {
                asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
            }
        }
    }
    for mnemonic in [
        "psllw", "pslld", "psllq", "psrlw", "psrld", "psrlq", "psraw", "psrad", "pslldq", "psrldq",
    ] {
        for count in [0, 1, 31, 255] {
            for destination in DESTINATIONS {
                asm.push_str(&format!("{mnemonic} ${count}, {destination}\n"));
            }
        }
    }
    for mnemonic in ["pshufd", "pshuflw", "pshufhw", "shufps", "shufpd"] {
        for source in SOURCES {
            for destination in DESTINATIONS {
                asm.push_str(&format!("{mnemonic} $27, {source}, {destination}\n"));
            }
        }
    }
    for mnemonic in ["movdqu", "movdqa", "movaps", "movapd", "movups"] {
        for source in SOURCES {
            for destination in DESTINATIONS {
                asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
                asm.push_str(&format!("{mnemonic} {destination}, {source}\n"));
            }
        }
    }
    let (output, reference) = oracle(&asm, "packed_sse");
    assert_eq!(text(&output), text(&reference));
}

/// SSE4.1 lane transfers across both escape maps, register widths and REX.
#[test]
fn lane_transfer_forms_match_gas() {
    const WORD: &[&str] = &[
        "%eax", "%ecx", "%edi", "%esp", "%ebp", "%r8d", "%r13d", "%r15d",
    ];
    const QUAD: &[&str] = &[
        "%rax", "%rcx", "%rdi", "%rsp", "%rbp", "%r8", "%r13", "%r15",
    ];
    const MEMORY: &[&str] = &["(%rax)", "(%r12,%r13,2)", "-9(%rbp)", "(%rsp)"];
    const VECTORS: &[&str] = &["%xmm0", "%xmm7", "%xmm8", "%xmm15"];
    let mut asm = String::from(".text\n");
    for (suffix, registers, last) in [
        ("b", WORD, 15),
        ("w", WORD, 7),
        ("d", WORD, 3),
        ("q", QUAD, 1),
    ] {
        for vector in VECTORS {
            for register in registers {
                asm.push_str(&format!("pextr{suffix} $0, {vector}, {register}\n"));
                asm.push_str(&format!("pinsr{suffix} ${last}, {register}, {vector}\n"));
            }
            if suffix == "w" {
                continue;
            }
            for memory in MEMORY {
                asm.push_str(&format!("pextr{suffix} ${last}, {vector}, {memory}\n"));
                asm.push_str(&format!("pinsr{suffix} $0, {memory}, {vector}\n"));
            }
        }
    }
    let (output, reference) = oracle(&asm, "lane_transfer");
    assert_eq!(text(&output), text(&reference));
}

/// The encoded bytes of each packed form, pinned so a regression cannot hide
/// behind a reference assembler that moved with it.
#[test]
fn packed_sse_encodings_are_pinned() {
    const CASES: &[(&str, &str)] = &[
        ("paddb %xmm1, %xmm2", "660ffcd1"),
        ("paddb %xmm9, %xmm12", "66450ffce1"),
        ("paddb 8(%rbp), %xmm3", "660ffc5d08"),
        ("paddw %xmm1, %xmm2", "660ffdd1"),
        ("paddw %xmm9, %xmm12", "66450ffde1"),
        ("paddw 8(%rbp), %xmm3", "660ffd5d08"),
        ("paddd %xmm1, %xmm2", "660ffed1"),
        ("paddd %xmm9, %xmm12", "66450ffee1"),
        ("paddd 8(%rbp), %xmm3", "660ffe5d08"),
        ("paddq %xmm1, %xmm2", "660fd4d1"),
        ("paddq %xmm9, %xmm12", "66450fd4e1"),
        ("paddq 8(%rbp), %xmm3", "660fd45d08"),
        ("psubb %xmm1, %xmm2", "660ff8d1"),
        ("psubb %xmm9, %xmm12", "66450ff8e1"),
        ("psubb 8(%rbp), %xmm3", "660ff85d08"),
        ("psubw %xmm1, %xmm2", "660ff9d1"),
        ("psubw %xmm9, %xmm12", "66450ff9e1"),
        ("psubw 8(%rbp), %xmm3", "660ff95d08"),
        ("psubd %xmm1, %xmm2", "660ffad1"),
        ("psubd %xmm9, %xmm12", "66450ffae1"),
        ("psubd 8(%rbp), %xmm3", "660ffa5d08"),
        ("psubq %xmm1, %xmm2", "660ffbd1"),
        ("psubq %xmm9, %xmm12", "66450ffbe1"),
        ("psubq 8(%rbp), %xmm3", "660ffb5d08"),
        ("pmullw %xmm1, %xmm2", "660fd5d1"),
        ("pmullw %xmm9, %xmm12", "66450fd5e1"),
        ("pmullw 8(%rbp), %xmm3", "660fd55d08"),
        ("pmuludq %xmm1, %xmm2", "660ff4d1"),
        ("pmuludq %xmm9, %xmm12", "66450ff4e1"),
        ("pmuludq 8(%rbp), %xmm3", "660ff45d08"),
        ("pmulld %xmm1, %xmm2", "660f3840d1"),
        ("pmulld %xmm9, %xmm12", "66450f3840e1"),
        ("pmulld 8(%rbp), %xmm3", "660f38405d08"),
        ("pand %xmm1, %xmm2", "660fdbd1"),
        ("pand %xmm9, %xmm12", "66450fdbe1"),
        ("pand 8(%rbp), %xmm3", "660fdb5d08"),
        ("pandn %xmm1, %xmm2", "660fdfd1"),
        ("pandn %xmm9, %xmm12", "66450fdfe1"),
        ("pandn 8(%rbp), %xmm3", "660fdf5d08"),
        ("por %xmm1, %xmm2", "660febd1"),
        ("por %xmm9, %xmm12", "66450febe1"),
        ("por 8(%rbp), %xmm3", "660feb5d08"),
        ("pxor %xmm1, %xmm2", "660fefd1"),
        ("pxor %xmm9, %xmm12", "66450fefe1"),
        ("pxor 8(%rbp), %xmm3", "660fef5d08"),
        ("punpcklbw %xmm1, %xmm2", "660f60d1"),
        ("punpcklbw %xmm9, %xmm12", "66450f60e1"),
        ("punpcklbw 8(%rbp), %xmm3", "660f605d08"),
        ("punpcklwd %xmm1, %xmm2", "660f61d1"),
        ("punpcklwd %xmm9, %xmm12", "66450f61e1"),
        ("punpcklwd 8(%rbp), %xmm3", "660f615d08"),
        ("punpckldq %xmm1, %xmm2", "660f62d1"),
        ("punpckldq %xmm9, %xmm12", "66450f62e1"),
        ("punpckldq 8(%rbp), %xmm3", "660f625d08"),
        ("punpcklqdq %xmm1, %xmm2", "660f6cd1"),
        ("punpcklqdq %xmm9, %xmm12", "66450f6ce1"),
        ("punpcklqdq 8(%rbp), %xmm3", "660f6c5d08"),
        ("punpckhbw %xmm1, %xmm2", "660f68d1"),
        ("punpckhbw %xmm9, %xmm12", "66450f68e1"),
        ("punpckhbw 8(%rbp), %xmm3", "660f685d08"),
        ("punpckhwd %xmm1, %xmm2", "660f69d1"),
        ("punpckhwd %xmm9, %xmm12", "66450f69e1"),
        ("punpckhwd 8(%rbp), %xmm3", "660f695d08"),
        ("punpckhdq %xmm1, %xmm2", "660f6ad1"),
        ("punpckhdq %xmm9, %xmm12", "66450f6ae1"),
        ("punpckhdq 8(%rbp), %xmm3", "660f6a5d08"),
        ("punpckhqdq %xmm1, %xmm2", "660f6dd1"),
        ("punpckhqdq %xmm9, %xmm12", "66450f6de1"),
        ("punpckhqdq 8(%rbp), %xmm3", "660f6d5d08"),
        ("pcmpeqb %xmm1, %xmm2", "660f74d1"),
        ("pcmpeqb %xmm9, %xmm12", "66450f74e1"),
        ("pcmpeqb 8(%rbp), %xmm3", "660f745d08"),
        ("pcmpeqw %xmm1, %xmm2", "660f75d1"),
        ("pcmpeqw %xmm9, %xmm12", "66450f75e1"),
        ("pcmpeqw 8(%rbp), %xmm3", "660f755d08"),
        ("pcmpeqd %xmm1, %xmm2", "660f76d1"),
        ("pcmpeqd %xmm9, %xmm12", "66450f76e1"),
        ("pcmpeqd 8(%rbp), %xmm3", "660f765d08"),
        ("pcmpgtb %xmm1, %xmm2", "660f64d1"),
        ("pcmpgtb %xmm9, %xmm12", "66450f64e1"),
        ("pcmpgtb 8(%rbp), %xmm3", "660f645d08"),
        ("pcmpgtw %xmm1, %xmm2", "660f65d1"),
        ("pcmpgtw %xmm9, %xmm12", "66450f65e1"),
        ("pcmpgtw 8(%rbp), %xmm3", "660f655d08"),
        ("pcmpgtd %xmm1, %xmm2", "660f66d1"),
        ("pcmpgtd %xmm9, %xmm12", "66450f66e1"),
        ("pcmpgtd 8(%rbp), %xmm3", "660f665d08"),
        ("addps %xmm1, %xmm2", "0f58d1"),
        ("addps %xmm9, %xmm12", "450f58e1"),
        ("addps 8(%rbp), %xmm3", "0f585d08"),
        ("subps %xmm1, %xmm2", "0f5cd1"),
        ("subps %xmm9, %xmm12", "450f5ce1"),
        ("subps 8(%rbp), %xmm3", "0f5c5d08"),
        ("mulps %xmm1, %xmm2", "0f59d1"),
        ("mulps %xmm9, %xmm12", "450f59e1"),
        ("mulps 8(%rbp), %xmm3", "0f595d08"),
        ("divps %xmm1, %xmm2", "0f5ed1"),
        ("divps %xmm9, %xmm12", "450f5ee1"),
        ("divps 8(%rbp), %xmm3", "0f5e5d08"),
        ("minps %xmm1, %xmm2", "0f5dd1"),
        ("minps %xmm9, %xmm12", "450f5de1"),
        ("minps 8(%rbp), %xmm3", "0f5d5d08"),
        ("maxps %xmm1, %xmm2", "0f5fd1"),
        ("maxps %xmm9, %xmm12", "450f5fe1"),
        ("maxps 8(%rbp), %xmm3", "0f5f5d08"),
        ("sqrtps %xmm1, %xmm2", "0f51d1"),
        ("sqrtps %xmm9, %xmm12", "450f51e1"),
        ("sqrtps 8(%rbp), %xmm3", "0f515d08"),
        ("andps %xmm1, %xmm2", "0f54d1"),
        ("andps %xmm9, %xmm12", "450f54e1"),
        ("andps 8(%rbp), %xmm3", "0f545d08"),
        ("andnps %xmm1, %xmm2", "0f55d1"),
        ("andnps %xmm9, %xmm12", "450f55e1"),
        ("andnps 8(%rbp), %xmm3", "0f555d08"),
        ("orps %xmm1, %xmm2", "0f56d1"),
        ("orps %xmm9, %xmm12", "450f56e1"),
        ("orps 8(%rbp), %xmm3", "0f565d08"),
        ("xorps %xmm1, %xmm2", "0f57d1"),
        ("xorps %xmm9, %xmm12", "450f57e1"),
        ("xorps 8(%rbp), %xmm3", "0f575d08"),
        ("unpcklps %xmm1, %xmm2", "0f14d1"),
        ("unpcklps %xmm9, %xmm12", "450f14e1"),
        ("unpcklps 8(%rbp), %xmm3", "0f145d08"),
        ("unpckhps %xmm1, %xmm2", "0f15d1"),
        ("unpckhps %xmm9, %xmm12", "450f15e1"),
        ("unpckhps 8(%rbp), %xmm3", "0f155d08"),
        ("addpd %xmm1, %xmm2", "660f58d1"),
        ("addpd %xmm9, %xmm12", "66450f58e1"),
        ("addpd 8(%rbp), %xmm3", "660f585d08"),
        ("subpd %xmm1, %xmm2", "660f5cd1"),
        ("subpd %xmm9, %xmm12", "66450f5ce1"),
        ("subpd 8(%rbp), %xmm3", "660f5c5d08"),
        ("mulpd %xmm1, %xmm2", "660f59d1"),
        ("mulpd %xmm9, %xmm12", "66450f59e1"),
        ("mulpd 8(%rbp), %xmm3", "660f595d08"),
        ("divpd %xmm1, %xmm2", "660f5ed1"),
        ("divpd %xmm9, %xmm12", "66450f5ee1"),
        ("divpd 8(%rbp), %xmm3", "660f5e5d08"),
        ("minpd %xmm1, %xmm2", "660f5dd1"),
        ("minpd %xmm9, %xmm12", "66450f5de1"),
        ("minpd 8(%rbp), %xmm3", "660f5d5d08"),
        ("maxpd %xmm1, %xmm2", "660f5fd1"),
        ("maxpd %xmm9, %xmm12", "66450f5fe1"),
        ("maxpd 8(%rbp), %xmm3", "660f5f5d08"),
        ("sqrtpd %xmm1, %xmm2", "660f51d1"),
        ("sqrtpd %xmm9, %xmm12", "66450f51e1"),
        ("sqrtpd 8(%rbp), %xmm3", "660f515d08"),
        ("andpd %xmm1, %xmm2", "660f54d1"),
        ("andpd %xmm9, %xmm12", "66450f54e1"),
        ("andpd 8(%rbp), %xmm3", "660f545d08"),
        ("andnpd %xmm1, %xmm2", "660f55d1"),
        ("andnpd %xmm9, %xmm12", "66450f55e1"),
        ("andnpd 8(%rbp), %xmm3", "660f555d08"),
        ("orpd %xmm1, %xmm2", "660f56d1"),
        ("orpd %xmm9, %xmm12", "66450f56e1"),
        ("orpd 8(%rbp), %xmm3", "660f565d08"),
        ("xorpd %xmm1, %xmm2", "660f57d1"),
        ("xorpd %xmm9, %xmm12", "66450f57e1"),
        ("xorpd 8(%rbp), %xmm3", "660f575d08"),
        ("unpcklpd %xmm1, %xmm2", "660f14d1"),
        ("unpcklpd %xmm9, %xmm12", "66450f14e1"),
        ("unpcklpd 8(%rbp), %xmm3", "660f145d08"),
        ("unpckhpd %xmm1, %xmm2", "660f15d1"),
        ("unpckhpd %xmm9, %xmm12", "66450f15e1"),
        ("unpckhpd 8(%rbp), %xmm3", "660f155d08"),
        ("psllw %xmm1, %xmm2", "660ff1d1"),
        ("psllw %xmm9, %xmm12", "66450ff1e1"),
        ("psllw $5, %xmm2", "660f71f205"),
        ("psllw $5, %xmm12", "66410f71f405"),
        ("pslld %xmm1, %xmm2", "660ff2d1"),
        ("pslld %xmm9, %xmm12", "66450ff2e1"),
        ("pslld $5, %xmm2", "660f72f205"),
        ("pslld $5, %xmm12", "66410f72f405"),
        ("psllq %xmm1, %xmm2", "660ff3d1"),
        ("psllq %xmm9, %xmm12", "66450ff3e1"),
        ("psllq $5, %xmm2", "660f73f205"),
        ("psllq $5, %xmm12", "66410f73f405"),
        ("psrlw %xmm1, %xmm2", "660fd1d1"),
        ("psrlw %xmm9, %xmm12", "66450fd1e1"),
        ("psrlw $5, %xmm2", "660f71d205"),
        ("psrlw $5, %xmm12", "66410f71d405"),
        ("psrld %xmm1, %xmm2", "660fd2d1"),
        ("psrld %xmm9, %xmm12", "66450fd2e1"),
        ("psrld $5, %xmm2", "660f72d205"),
        ("psrld $5, %xmm12", "66410f72d405"),
        ("psrlq %xmm1, %xmm2", "660fd3d1"),
        ("psrlq %xmm9, %xmm12", "66450fd3e1"),
        ("psrlq $5, %xmm2", "660f73d205"),
        ("psrlq $5, %xmm12", "66410f73d405"),
        ("psraw %xmm1, %xmm2", "660fe1d1"),
        ("psraw %xmm9, %xmm12", "66450fe1e1"),
        ("psraw $5, %xmm2", "660f71e205"),
        ("psraw $5, %xmm12", "66410f71e405"),
        ("psrad %xmm1, %xmm2", "660fe2d1"),
        ("psrad %xmm9, %xmm12", "66450fe2e1"),
        ("psrad $5, %xmm2", "660f72e205"),
        ("psrad $5, %xmm12", "66410f72e405"),
        ("pslldq $5, %xmm2", "660f73fa05"),
        ("pslldq $5, %xmm12", "66410f73fc05"),
        ("psrldq $5, %xmm2", "660f73da05"),
        ("psrldq $5, %xmm12", "66410f73dc05"),
        ("shufps $27, %xmm1, %xmm2", "0fc6d11b"),
        ("shufps $27, %xmm9, %xmm12", "450fc6e11b"),
        ("pshufhw $27, %xmm1, %xmm2", "f30f70d11b"),
        ("pextrb $1, %xmm2, %eax", "660f3a14d001"),
        ("pextrb $1, %xmm12, %r9d", "66450f3a14e101"),
        ("pextrb $1, %xmm12, 8(%rbp)", "66440f3a14650801"),
        ("pinsrb $1, %eax, %xmm2", "660f3a20d001"),
        ("pinsrb $1, %r9d, %xmm12", "66450f3a20e101"),
        ("pinsrb $1, 8(%rbp), %xmm12", "66440f3a20650801"),
        ("pextrd $1, %xmm2, %eax", "660f3a16d001"),
        ("pextrd $1, %xmm12, %r9d", "66450f3a16e101"),
        ("pextrd $1, %xmm12, 8(%rbp)", "66440f3a16650801"),
        ("pinsrd $1, %eax, %xmm2", "660f3a22d001"),
        ("pinsrd $1, %r9d, %xmm12", "66450f3a22e101"),
        ("pinsrd $1, 8(%rbp), %xmm12", "66440f3a22650801"),
        ("pextrq $1, %xmm2, %rax", "66480f3a16d001"),
        ("pextrq $1, %xmm12, %r9", "664d0f3a16e101"),
        ("pextrq $1, %xmm12, 8(%rbp)", "664c0f3a16650801"),
        ("pinsrq $1, %rax, %xmm2", "66480f3a22d001"),
        ("pinsrq $1, %r9, %xmm12", "664d0f3a22e101"),
        ("pinsrq $1, 8(%rbp), %xmm12", "664c0f3a22650801"),
    ];
    for (asm, expected) in CASES {
        let tmp = TempDir::new("pinned_sse");
        let path = tmp.path().join("pinned.o");
        Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .unwrap()
            .assemble(&format!(".text\n{asm}\n"), &path)
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let encoded: String = text(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(&encoded, expected, "{asm}");
    }
}

/// A lane transfer whose register width disagrees with the mnemonic is not
/// silently encoded as the other width.
#[test]
fn lane_transfer_rejects_a_mismatched_register_width() {
    for asm in [
        "pextrd $0, %xmm0, %rax",
        "pextrq $0, %xmm0, %eax",
        "pinsrd $0, %rax, %xmm0",
        "pinsrq $0, %eax, %xmm0",
    ] {
        let tmp = TempDir::new("lane_width");
        let path = tmp.path().join("lane.o");
        assert!(
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(&format!(".text\n{asm}\n"), &path)
                .is_err(),
            "{asm}"
        );
    }
}
