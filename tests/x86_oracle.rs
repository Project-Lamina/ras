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

/// The SSE2, SSSE3, SSE4.1 and SSE4.2 forms GNU as accepts, across both
/// escape maps, every register slot and the awkward addressing bases.
#[test]
fn extended_sse_forms_match_gas() {
    const BINARY: &[&str] = &[
        "pmaddwd",
        "pshufb",
        "packuswb",
        "packsswb",
        "packssdw",
        "packusdw",
        "pavgb",
        "pavgw",
        "psignb",
        "psignw",
        "psignd",
        "pabsb",
        "pabsw",
        "pabsd",
        "pminsb",
        "pminsw",
        "pminsd",
        "pmaxsb",
        "pmaxsw",
        "pmaxsd",
        "pminub",
        "pminuw",
        "pminud",
        "pmaxub",
        "pmaxuw",
        "pmaxud",
        "pcmpeqq",
        "pcmpgtq",
        "pmuldq",
        "pmulhw",
        "pmulhuw",
        "pmulhrsw",
        "pmaddubsw",
        "psadbw",
        "phaddw",
        "phaddd",
        "phaddsw",
        "phsubw",
        "phsubd",
        "phsubsw",
        "pmovsxbw",
        "pmovsxbd",
        "pmovsxbq",
        "pmovsxwd",
        "pmovsxwq",
        "pmovsxdq",
        "pmovzxbw",
        "pmovzxbd",
        "pmovzxbq",
        "pmovzxwd",
        "pmovzxwq",
        "pmovzxdq",
        "ptest",
        "phminposuw",
        "paddsb",
        "paddsw",
        "paddusb",
        "paddusw",
        "psubsb",
        "psubsw",
        "psubusb",
        "psubusw",
        "addsubps",
        "addsubpd",
        "haddps",
        "haddpd",
        "hsubps",
        "hsubpd",
        "movshdup",
        "movsldup",
        "movddup",
        "rcpps",
        "rsqrtps",
        "cvtdq2ps",
        "cvtps2dq",
        "cvttps2dq",
        "cvtdq2pd",
        "cvtpd2ps",
        "cvtps2pd",
        "cvttpd2dq",
        "cvtpd2dq",
        "aesenc",
        "aesenclast",
        "aesdec",
        "aesdeclast",
        "aesimc",
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
        "(%r12,%r13,2)",
        "(%rax,%r9,8)",
        "1024(%rbx)",
        "-128(%r14)",
        "pinned(%rip)",
        "8(%r13)",
    ];
    const DESTINATIONS: &[&str] = &["%xmm0", "%xmm7", "%xmm8", "%xmm15"];
    let mut asm = String::from(".text\npinned: .quad 0\n");
    for mnemonic in BINARY {
        for source in SOURCES {
            for destination in DESTINATIONS {
                asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
            }
        }
    }
    for mnemonic in ["blendvps", "blendvpd", "pblendvb"] {
        for source in SOURCES {
            for destination in DESTINATIONS {
                asm.push_str(&format!("{mnemonic} %xmm0, {source}, {destination}\n"));
            }
        }
    }
    for (mnemonic, counts) in [
        ("pblendw", &[0, 1, 127, 128, 255][..]),
        ("palignr", &[0, 1, 15, 16, 255]),
        ("pclmulqdq", &[0, 1, 16, 17, 255]),
        ("blendps", &[0, 15, 255]),
        ("blendpd", &[0, 3, 255]),
        ("insertps", &[0, 78, 255]),
        ("dpps", &[0, 255]),
        ("dppd", &[0, 51, 255]),
        ("mpsadbw", &[0, 7, 255]),
        ("roundps", &[0, 1, 11, 255]),
        ("roundpd", &[0, 2, 255]),
        ("roundss", &[0, 3, 255]),
        ("roundsd", &[0, 4, 255]),
        ("cmpps", &[0, 7, 31, 255]),
        ("cmppd", &[0, 7, 255]),
        ("aeskeygenassist", &[0, 255]),
        ("pcmpistri", &[0, 255]),
        ("pcmpistrm", &[0, 255]),
        ("pcmpestri", &[0, 255]),
        ("pcmpestrm", &[0, 255]),
    ] {
        for count in counts {
            for source in SOURCES {
                for destination in DESTINATIONS {
                    asm.push_str(&format!("{mnemonic} ${count}, {source}, {destination}\n"));
                }
            }
        }
    }
    for source in SOURCES.iter().filter(|source| source.contains('(')) {
        for destination in DESTINATIONS {
            for mnemonic in ["movntdq", "movntps", "movntpd"] {
                asm.push_str(&format!("{mnemonic} {destination}, {source}\n"));
            }
            for mnemonic in ["lddqu", "movntdqa"] {
                asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
            }
            for mnemonic in ["movlps", "movhps", "movlpd", "movhpd", "movupd"] {
                asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
                asm.push_str(&format!("{mnemonic} {destination}, {source}\n"));
            }
        }
        asm.push_str(&format!("prefetchnta {source}\nprefetcht0 {source}\n"));
        asm.push_str(&format!("prefetcht1 {source}\nprefetcht2 {source}\n"));
        asm.push_str(&format!("ldmxcsr {source}\nstmxcsr {source}\n"));
    }
    for vector in DESTINATIONS {
        for register in ["%eax", "%esp", "%ebp", "%r9d", "%r12d", "%r15d"] {
            for mnemonic in ["pmovmskb", "movmskps", "movmskpd"] {
                asm.push_str(&format!("{mnemonic} {vector}, {register}\n"));
            }
            asm.push_str(&format!("cvtss2si {vector}, {register}\n"));
            asm.push_str(&format!("cvtsd2si {vector}, {register}\n"));
            asm.push_str(&format!("cvttss2si {vector}, {register}\n"));
        }
        for other in DESTINATIONS {
            asm.push_str(&format!("movlhps {vector}, {other}\n"));
            asm.push_str(&format!("movhlps {vector}, {other}\n"));
            asm.push_str(&format!("rcpss {vector}, {other}\n"));
            asm.push_str(&format!("rsqrtss {vector}, {other}\n"));
        }
    }
    asm.push_str("sfence\nlfence\nmfence\npause\n");
    let (output, reference) = oracle(&asm, "extended_sse");
    assert_eq!(text(&output), text(&reference));
}

/// `pextrw` picks its opcode map from the kind of its destination, and
/// `extractps` shares the lane-transfer path.
#[test]
fn word_and_lane_extraction_forms_match_gas() {
    const MEMORY: &[&str] = &[
        "(%rax)",
        "(%r12)",
        "(%r13)",
        "(%rsp)",
        "(%rbp)",
        "(%r12,%r13,2)",
        "pinned(%rip)",
        "-128(%r14)",
        "1024(%rbx)",
    ];
    const VECTORS: &[&str] = &["%xmm0", "%xmm7", "%xmm8", "%xmm15"];
    let mut asm = String::from(".text\npinned: .quad 0\n");
    for lane in [0, 1, 7, 255] {
        for vector in VECTORS {
            for register in ["%eax", "%esp", "%ebp", "%r9d", "%r12d", "%r15d"] {
                asm.push_str(&format!("pextrw ${lane}, {vector}, {register}\n"));
                asm.push_str(&format!("pinsrw ${lane}, {register}, {vector}\n"));
                asm.push_str(&format!("extractps ${lane}, {vector}, {register}\n"));
            }
            for memory in MEMORY {
                asm.push_str(&format!("pextrw ${lane}, {vector}, {memory}\n"));
                asm.push_str(&format!("pinsrw ${lane}, {memory}, {vector}\n"));
                asm.push_str(&format!("extractps ${lane}, {vector}, {memory}\n"));
            }
        }
    }
    let (output, reference) = oracle(&asm, "word_extraction");
    assert_eq!(text(&output), text(&reference));
}

/// SSE4.2 CRC32 over every source width, including the byte registers that
/// need a REX of their own.
#[test]
fn crc32_and_non_temporal_stores_match_gas() {
    const MEMORY: &[&str] = &["(%rax)", "(%r12)", "(%r13)", "(%rsp)", "(%rbp)", "-9(%r14)"];
    let mut asm = String::from(".text\n");
    for (suffix, sources, destinations) in [
        (
            "b",
            &[
                "%al", "%cl", "%spl", "%bpl", "%sil", "%dil", "%r9b", "%r15b",
            ][..],
            &["%eax", "%ecx", "%esp", "%ebp", "%r9d", "%r15d"][..],
        ),
        (
            "w",
            &["%ax", "%cx", "%sp", "%bp", "%r9w", "%r15w"],
            &["%eax", "%ecx", "%esp", "%ebp", "%r9d", "%r15d"],
        ),
        (
            "l",
            &["%eax", "%ecx", "%esp", "%ebp", "%r9d", "%r15d"],
            &["%eax", "%ecx", "%esp", "%ebp", "%r9d", "%r15d"],
        ),
        (
            "q",
            &["%rax", "%rcx", "%rsp", "%rbp", "%r9", "%r15"],
            &["%rax", "%rcx", "%rsp", "%rbp", "%r9", "%r15"],
        ),
    ] {
        for destination in destinations {
            for source in sources {
                asm.push_str(&format!("crc32{suffix} {source}, {destination}\n"));
            }
            for memory in MEMORY {
                asm.push_str(&format!("crc32{suffix} {memory}, {destination}\n"));
            }
        }
    }
    for memory in MEMORY {
        for register in ["%eax", "%ebp", "%r12d"] {
            asm.push_str(&format!("movntil {register}, {memory}\n"));
        }
        for register in ["%rax", "%rbp", "%r12"] {
            asm.push_str(&format!("movntiq {register}, {memory}\n"));
        }
    }
    let (output, reference) = oracle(&asm, "crc32");
    assert_eq!(text(&output), text(&reference));
}

/// The AVX2 per-lane shifts, which select the three-byte VEX form because
/// they live in the `0f 38` map.
#[test]
fn variable_vector_shifts_match_gas() {
    const MEMORY: &[&str] = &[
        "(%rax)",
        "(%r12)",
        "(%r13)",
        "(%rsp)",
        "(%rbp)",
        "(%r12,%r13,2)",
        "pinned(%rip)",
        "1024(%rbx)",
    ];
    let mut asm = String::from(".text\npinned: .quad 0\n");
    for mnemonic in ["vpsllvd", "vpsrlvd", "vpsravd", "vpsllvq", "vpsrlvq"] {
        for width in ["xmm", "ymm"] {
            let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
            for count in &registers {
                for source in &registers {
                    for destination in &registers {
                        asm.push_str(&format!("{mnemonic} {count}, {source}, {destination}\n"));
                    }
                }
            }
            for memory in MEMORY {
                for source in &registers {
                    asm.push_str(&format!("{mnemonic} {memory}, {source}, %{width}3\n"));
                }
            }
        }
    }
    let (output, reference) = oracle(&asm, "variable_shifts");
    assert_eq!(text(&output), text(&reference));
}

/// The VEX counterparts of the packed set, over both vector widths and the
/// register numbers that force the three-byte prefix.
#[test]
fn avx_packed_forms_match_gas() {
    const NDS: &[&str] = &[
        "vpaddb",
        "vpaddw",
        "vpaddd",
        "vpaddq",
        "vpsubb",
        "vpsubq",
        "vpmullw",
        "vpmuludq",
        "vpmaddwd",
        "vpsadbw",
        "vpavgb",
        "vpacksswb",
        "vpackuswb",
        "vpand",
        "vpandn",
        "vpor",
        "vpxor",
        "vpunpcklbw",
        "vpunpckhqdq",
        "vpcmpeqb",
        "vpcmpgtd",
        "vaddps",
        "vmulpd",
        "vdivps",
        "vminpd",
        "vmaxps",
        "vandnpd",
        "vunpcklps",
        "vaddsubps",
        "vhaddpd",
        "vpshufb",
        "vphaddw",
        "vpmaddubsw",
        "vpsignd",
        "vpmulhrsw",
        "vpmuldq",
        "vpcmpeqq",
        "vpackusdw",
        "vpcmpgtq",
        "vpminsb",
        "vpmaxud",
        "vpmulld",
        "vaddss",
        "vdivsd",
        "vsqrtss",
        "vcvtss2sd",
        "vmovhlps",
        "vmovlhps",
    ];
    const RM: &[&str] = &[
        "vsqrtps",
        "vsqrtpd",
        "vrcpps",
        "vmovshdup",
        "vmovddup",
        "vcvtdq2ps",
        "vcvttps2dq",
        "vptest",
        "vpabsb",
        "vpabsd",
        "vpmovsxbw",
        "vpmovzxdq",
        "vucomiss",
        "vcomisd",
        "vbroadcastss",
        "vpbroadcastb",
        "vpbroadcastq",
    ];
    const MEMORY: &[&str] = &[
        "(%rax)",
        "(%r12)",
        "(%r13)",
        "(%rsp)",
        "(%rbp)",
        "(%r12,%r13,2)",
        "pinned(%rip)",
        "-128(%r14)",
    ];
    let scalar = |name: &str| {
        name.ends_with("ss") || name.ends_with("sd") || matches!(name, "vmovhlps" | "vmovlhps")
    };
    let mut asm = String::from(".text\npinned: .quad 0\n");
    for mnemonic in NDS {
        for width in ["xmm", "ymm"] {
            if width == "ymm" && scalar(mnemonic) {
                continue;
            }
            let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
            for second in &registers {
                for first in &registers {
                    for destination in &registers {
                        asm.push_str(&format!("{mnemonic} {second}, {first}, {destination}\n"));
                    }
                }
            }
            if matches!(*mnemonic, "vmovhlps" | "vmovlhps") {
                continue;
            }
            for memory in MEMORY {
                asm.push_str(&format!("{mnemonic} {memory}, %{width}2, %{width}3\n"));
            }
        }
    }
    for mnemonic in RM {
        for source in [0, 7, 8, 15].map(|n| format!("%xmm{n}")) {
            for destination in [0, 7, 8, 15].map(|n| format!("%xmm{n}")) {
                asm.push_str(&format!("{mnemonic} {source}, {destination}\n"));
            }
        }
        for memory in MEMORY {
            asm.push_str(&format!("{mnemonic} {memory}, %xmm3\n"));
        }
    }
    for (mnemonic, count) in [
        ("vpblendw", 255),
        ("vpalignr", 15),
        ("vblendps", 15),
        ("vdpps", 255),
        ("vmpsadbw", 7),
        ("vcmpps", 7),
        ("vshufps", 27),
        ("vshufpd", 3),
    ] {
        for width in ["xmm", "ymm"] {
            let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
            for second in &registers {
                for destination in &registers {
                    asm.push_str(&format!(
                        "{mnemonic} ${count}, {second}, %{width}2, {destination}\n"
                    ));
                }
            }
            for memory in MEMORY {
                asm.push_str(&format!(
                    "{mnemonic} ${count}, {memory}, %{width}2, %{width}3\n"
                ));
            }
        }
    }
    for (mnemonic, count) in [
        ("vpshufd", 27),
        ("vpshufhw", 27),
        ("vpshuflw", 27),
        ("vroundps", 11),
        ("vpermq", 27),
        ("vpermilps", 27),
    ] {
        for width in ["xmm", "ymm"] {
            if width == "xmm" && mnemonic == "vpermq" {
                continue;
            }
            let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
            for source in &registers {
                for destination in &registers {
                    asm.push_str(&format!("{mnemonic} ${count}, {source}, {destination}\n"));
                }
            }
            for memory in MEMORY {
                asm.push_str(&format!("{mnemonic} ${count}, {memory}, %{width}3\n"));
            }
        }
    }
    for mnemonic in ["vblendvps", "vblendvpd", "vpblendvb"] {
        for width in ["xmm", "ymm"] {
            let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
            for mask in &registers {
                for second in &registers {
                    asm.push_str(&format!(
                        "{mnemonic} {mask}, {second}, %{width}2, %{width}3\n"
                    ));
                }
            }
            for memory in MEMORY {
                asm.push_str(&format!(
                    "{mnemonic} %{width}4, {memory}, %{width}2, %{width}3\n"
                ));
            }
        }
    }
    for mnemonic in [
        "vpsllw", "vpslld", "vpsllq", "vpsrlw", "vpsrld", "vpsrlq", "vpsraw", "vpsrad", "vpslldq",
        "vpsrldq",
    ] {
        for count in [0, 1, 31, 255] {
            for width in ["xmm", "ymm"] {
                for source in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                    asm.push_str(&format!("{mnemonic} ${count}, {source}, %{width}3\n"));
                }
            }
        }
    }
    for mnemonic in [
        "vmovaps", "vmovapd", "vmovups", "vmovupd", "vmovdqa", "vmovdqu",
    ] {
        for width in ["xmm", "ymm"] {
            for register in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                for memory in MEMORY {
                    asm.push_str(&format!("{mnemonic} {memory}, {register}\n"));
                    asm.push_str(&format!("{mnemonic} {register}, {memory}\n"));
                }
                for other in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                    asm.push_str(&format!("{mnemonic} {register}, {other}\n"));
                }
            }
        }
    }
    for mnemonic in ["vmovntdq", "vmovntps", "vmovntpd"] {
        for width in ["xmm", "ymm"] {
            for register in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                for memory in MEMORY {
                    asm.push_str(&format!("{mnemonic} {register}, {memory}\n"));
                }
            }
        }
    }
    asm.push_str("vzeroall\nvzeroupper\n");
    let (output, reference) = oracle(&asm, "avx_packed");
    assert_eq!(text(&output), text(&reference));
}

/// The VEX lane transfers, scalar moves and FMA3 set, where the two-byte
/// prefix is chosen only when no field it cannot carry is in use.
#[test]
fn avx_transfers_and_fma_match_gas() {
    const MEMORY: &[&str] = &[
        "(%rax)",
        "(%r12)",
        "(%r13)",
        "(%rsp)",
        "(%rbp)",
        "(%r12,%r13,2)",
        "pinned(%rip)",
        "-128(%r14)",
    ];
    let vectors = || [0, 7, 8, 15].map(|n| format!("%xmm{n}"));
    let mut asm = String::from(".text\npinned: .quad 0\n");
    for lane in [0, 1, 15, 255] {
        for vector in vectors() {
            for register in ["%eax", "%esp", "%ebp", "%r9d", "%r12d", "%r15d"] {
                for mnemonic in ["vpextrb", "vpextrw", "vpextrd", "vextractps"] {
                    asm.push_str(&format!("{mnemonic} ${lane}, {vector}, {register}\n"));
                }
                for mnemonic in ["vpinsrb", "vpinsrw", "vpinsrd"] {
                    for first in vectors() {
                        asm.push_str(&format!(
                            "{mnemonic} ${lane}, {register}, {first}, {vector}\n"
                        ));
                    }
                }
            }
            for register in ["%rax", "%rsp", "%rbp", "%r9", "%r12", "%r15"] {
                asm.push_str(&format!("vpextrq ${lane}, {vector}, {register}\n"));
                for first in vectors() {
                    asm.push_str(&format!("vpinsrq ${lane}, {register}, {first}, {vector}\n"));
                }
            }
            for memory in MEMORY {
                for mnemonic in ["vpextrb", "vpextrw", "vpextrd", "vpextrq", "vextractps"] {
                    asm.push_str(&format!("{mnemonic} ${lane}, {vector}, {memory}\n"));
                }
                for mnemonic in ["vpinsrb", "vpinsrw", "vpinsrd", "vpinsrq"] {
                    asm.push_str(&format!("{mnemonic} ${lane}, {memory}, %xmm2, {vector}\n"));
                }
            }
        }
    }
    for width in ["xmm", "ymm"] {
        for vector in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
            for register in ["%eax", "%esp", "%ebp", "%r9d", "%r12d", "%r15d"] {
                for mnemonic in ["vpmovmskb", "vmovmskps", "vmovmskpd"] {
                    asm.push_str(&format!("{mnemonic} {vector}, {register}\n"));
                }
            }
        }
    }
    for vector in vectors() {
        for register in ["%rax", "%rsp", "%rbp", "%r9", "%r12", "%r15"] {
            asm.push_str(&format!("vmovq {register}, {vector}\n"));
            asm.push_str(&format!("vmovq {vector}, {register}\n"));
        }
        for register in ["%eax", "%esp", "%r12d"] {
            asm.push_str(&format!("vmovd {register}, {vector}\n"));
            asm.push_str(&format!("vmovd {vector}, {register}\n"));
        }
        for other in vectors() {
            asm.push_str(&format!("vmovq {other}, {vector}\n"));
            for mnemonic in ["vmovss", "vmovsd"] {
                for first in vectors() {
                    asm.push_str(&format!("{mnemonic} {other}, {first}, {vector}\n"));
                }
            }
        }
        for memory in MEMORY {
            asm.push_str(&format!("vmovq {memory}, {vector}\n"));
            asm.push_str(&format!("vmovq {vector}, {memory}\n"));
            for mnemonic in ["vmovss", "vmovsd"] {
                asm.push_str(&format!("{mnemonic} {memory}, {vector}\n"));
                asm.push_str(&format!("{mnemonic} {vector}, {memory}\n"));
            }
            for mnemonic in ["vmovhps", "vmovlps", "vmovhpd", "vmovlpd"] {
                asm.push_str(&format!("{mnemonic} {memory}, %xmm2, {vector}\n"));
                asm.push_str(&format!("{mnemonic} {vector}, {memory}\n"));
            }
        }
        for register in ["%eax", "%esp", "%r12d"] {
            for first in vectors() {
                asm.push_str(&format!("vcvtsi2ssl {register}, {first}, {vector}\n"));
                asm.push_str(&format!("vcvtsi2sdl {register}, {first}, {vector}\n"));
            }
        }
        for register in ["%rax", "%rsp", "%r12"] {
            asm.push_str(&format!("vcvtsi2ssq {register}, %xmm2, {vector}\n"));
            asm.push_str(&format!("vcvtsi2sdq {register}, %xmm2, {vector}\n"));
        }
        for register in ["%eax", "%r12d", "%rax", "%r12"] {
            for mnemonic in ["vcvtss2si", "vcvtsd2si", "vcvttss2si", "vcvttsd2si"] {
                asm.push_str(&format!("{mnemonic} {vector}, {register}\n"));
            }
        }
    }
    for operation in ["madd", "msub", "nmadd", "nmsub"] {
        for order in ["132", "213", "231"] {
            for kind in ["ps", "pd", "ss", "sd"] {
                let mnemonic = format!("vf{operation}{order}{kind}");
                let width = if kind.starts_with('p') { "ymm" } else { "xmm" };
                let registers = [0, 7, 8, 15].map(|n| format!("%{width}{n}"));
                for second in &registers {
                    for destination in &registers {
                        asm.push_str(&format!("{mnemonic} {second}, %{width}2, {destination}\n"));
                    }
                }
                for memory in MEMORY {
                    asm.push_str(&format!("{mnemonic} {memory}, %{width}2, %{width}3\n"));
                }
            }
        }
    }
    for operation in ["maddsub", "msubadd"] {
        for order in ["132", "213", "231"] {
            for kind in ["ps", "pd"] {
                let mnemonic = format!("vf{operation}{order}{kind}");
                for second in [0, 7, 8, 15].map(|n| format!("%ymm{n}")) {
                    asm.push_str(&format!("{mnemonic} {second}, %ymm2, %ymm15\n"));
                }
            }
        }
    }
    for mnemonic in ["vmaskmovps", "vmaskmovpd", "vpmaskmovd", "vpmaskmovq"] {
        for width in ["xmm", "ymm"] {
            for register in [0, 7, 8, 15].map(|n| format!("%{width}{n}")) {
                for memory in MEMORY {
                    asm.push_str(&format!("{mnemonic} {memory}, {register}, %{width}3\n"));
                    asm.push_str(&format!("{mnemonic} {register}, %{width}2, {memory}\n"));
                }
            }
        }
    }
    for lane in [0, 1] {
        for width in [("xmm", "ymm")] {
            for register in [0, 7, 8, 15].map(|n| format!("%{}{n}", width.0)) {
                asm.push_str(&format!("vinserti128 ${lane}, {register}, %ymm2, %ymm3\n"));
                asm.push_str(&format!("vinsertf128 ${lane}, {register}, %ymm2, %ymm3\n"));
                asm.push_str(&format!("vextracti128 ${lane}, %ymm9, {register}\n"));
                asm.push_str(&format!("vextractf128 ${lane}, %ymm9, {register}\n"));
            }
            for memory in MEMORY {
                asm.push_str(&format!("vextracti128 ${lane}, %ymm9, {memory}\n"));
                asm.push_str(&format!("vinserti128 ${lane}, {memory}, %ymm2, %ymm3\n"));
                asm.push_str(&format!("vperm2i128 ${lane}, {memory}, %ymm2, %ymm3\n"));
            }
        }
    }
    for second in [0, 7, 8, 15].map(|n| format!("%ymm{n}")) {
        asm.push_str(&format!("vpermd {second}, %ymm2, %ymm3\n"));
        asm.push_str(&format!("vpermps {second}, %ymm2, %ymm3\n"));
        asm.push_str(&format!("vpermilps {second}, %ymm2, %ymm3\n"));
        asm.push_str(&format!("vpermilpd {second}, %ymm2, %ymm3\n"));
        asm.push_str(&format!("vperm2f128 $1, {second}, %ymm2, %ymm3\n"));
        asm.push_str(&format!("vpermpd $27, {second}, %ymm3\n"));
        asm.push_str(&format!(
            "vbroadcastsd {}, %ymm3\n",
            second.replace('y', "x")
        ));
    }
    for memory in MEMORY {
        asm.push_str(&format!("vbroadcasti128 {memory}, %ymm3\n"));
        asm.push_str(&format!("vbroadcastf128 {memory}, %ymm3\n"));
        asm.push_str(&format!("vbroadcastsd {memory}, %ymm3\n"));
        asm.push_str(&format!("vlddqu {memory}, %ymm3\n"));
        asm.push_str(&format!("vmovntdqa {memory}, %xmm3\n"));
    }
    let (output, reference) = oracle(&asm, "avx_transfers");
    assert_eq!(text(&output), text(&reference));
}

/// The encoded bytes of the forms this pass added, pinned so a regression
/// cannot hide behind a reference assembler that moved with it.
#[test]
fn extended_encodings_are_pinned() {
    const CASES: &[(&str, &str)] = &[
        ("pmaddwd %xmm1, %xmm2", "660ff5d1"),
        ("pmaddwd %xmm9, %xmm12", "66450ff5e1"),
        ("pmaddwd %xmm15, %xmm0", "66410ff5c7"),
        ("pmaddwd 8(%rbp), %xmm3", "660ff55d08"),
        ("pmaddwd (%r12), %xmm8", "66450ff50424"),
        ("pmaddwd (%r13,%rax,4), %xmm15", "66450ff57c8500"),
        ("pshufb %xmm1, %xmm2", "660f3800d1"),
        ("pshufb %xmm9, %xmm12", "66450f3800e1"),
        ("pshufb %xmm15, %xmm0", "66410f3800c7"),
        ("pshufb 8(%rbp), %xmm3", "660f38005d08"),
        ("pshufb (%r12), %xmm8", "66450f38000424"),
        ("pshufb (%r13,%rax,4), %xmm15", "66450f38007c8500"),
        ("packuswb %xmm1, %xmm2", "660f67d1"),
        ("packuswb %xmm9, %xmm12", "66450f67e1"),
        ("packuswb %xmm15, %xmm0", "66410f67c7"),
        ("packuswb 8(%rbp), %xmm3", "660f675d08"),
        ("packuswb (%r12), %xmm8", "66450f670424"),
        ("packuswb (%r13,%rax,4), %xmm15", "66450f677c8500"),
        ("packsswb %xmm1, %xmm2", "660f63d1"),
        ("packsswb %xmm9, %xmm12", "66450f63e1"),
        ("packsswb %xmm15, %xmm0", "66410f63c7"),
        ("packsswb 8(%rbp), %xmm3", "660f635d08"),
        ("packsswb (%r12), %xmm8", "66450f630424"),
        ("packsswb (%r13,%rax,4), %xmm15", "66450f637c8500"),
        ("packssdw %xmm1, %xmm2", "660f6bd1"),
        ("packssdw %xmm9, %xmm12", "66450f6be1"),
        ("packssdw %xmm15, %xmm0", "66410f6bc7"),
        ("packssdw 8(%rbp), %xmm3", "660f6b5d08"),
        ("packssdw (%r12), %xmm8", "66450f6b0424"),
        ("packssdw (%r13,%rax,4), %xmm15", "66450f6b7c8500"),
        ("packusdw %xmm1, %xmm2", "660f382bd1"),
        ("packusdw %xmm9, %xmm12", "66450f382be1"),
        ("packusdw %xmm15, %xmm0", "66410f382bc7"),
        ("packusdw 8(%rbp), %xmm3", "660f382b5d08"),
        ("packusdw (%r12), %xmm8", "66450f382b0424"),
        ("packusdw (%r13,%rax,4), %xmm15", "66450f382b7c8500"),
        ("pavgb %xmm1, %xmm2", "660fe0d1"),
        ("pavgb %xmm9, %xmm12", "66450fe0e1"),
        ("pavgb %xmm15, %xmm0", "66410fe0c7"),
        ("pavgb 8(%rbp), %xmm3", "660fe05d08"),
        ("pavgb (%r12), %xmm8", "66450fe00424"),
        ("pavgb (%r13,%rax,4), %xmm15", "66450fe07c8500"),
        ("pavgw %xmm1, %xmm2", "660fe3d1"),
        ("pavgw %xmm9, %xmm12", "66450fe3e1"),
        ("pavgw %xmm15, %xmm0", "66410fe3c7"),
        ("pavgw 8(%rbp), %xmm3", "660fe35d08"),
        ("pavgw (%r12), %xmm8", "66450fe30424"),
        ("pavgw (%r13,%rax,4), %xmm15", "66450fe37c8500"),
        ("psignb %xmm1, %xmm2", "660f3808d1"),
        ("psignb %xmm9, %xmm12", "66450f3808e1"),
        ("psignb %xmm15, %xmm0", "66410f3808c7"),
        ("psignb 8(%rbp), %xmm3", "660f38085d08"),
        ("psignb (%r12), %xmm8", "66450f38080424"),
        ("psignb (%r13,%rax,4), %xmm15", "66450f38087c8500"),
        ("psignw %xmm1, %xmm2", "660f3809d1"),
        ("psignw %xmm9, %xmm12", "66450f3809e1"),
        ("psignw %xmm15, %xmm0", "66410f3809c7"),
        ("psignw 8(%rbp), %xmm3", "660f38095d08"),
        ("psignw (%r12), %xmm8", "66450f38090424"),
        ("psignw (%r13,%rax,4), %xmm15", "66450f38097c8500"),
        ("psignd %xmm1, %xmm2", "660f380ad1"),
        ("psignd %xmm9, %xmm12", "66450f380ae1"),
        ("psignd %xmm15, %xmm0", "66410f380ac7"),
        ("psignd 8(%rbp), %xmm3", "660f380a5d08"),
        ("psignd (%r12), %xmm8", "66450f380a0424"),
        ("psignd (%r13,%rax,4), %xmm15", "66450f380a7c8500"),
        ("pabsb %xmm1, %xmm2", "660f381cd1"),
        ("pabsb %xmm9, %xmm12", "66450f381ce1"),
        ("pabsb %xmm15, %xmm0", "66410f381cc7"),
        ("pabsb 8(%rbp), %xmm3", "660f381c5d08"),
        ("pabsb (%r12), %xmm8", "66450f381c0424"),
        ("pabsb (%r13,%rax,4), %xmm15", "66450f381c7c8500"),
        ("pabsw %xmm1, %xmm2", "660f381dd1"),
        ("pabsw %xmm9, %xmm12", "66450f381de1"),
        ("pabsw %xmm15, %xmm0", "66410f381dc7"),
        ("pabsw 8(%rbp), %xmm3", "660f381d5d08"),
        ("pabsw (%r12), %xmm8", "66450f381d0424"),
        ("pabsw (%r13,%rax,4), %xmm15", "66450f381d7c8500"),
        ("pabsd %xmm1, %xmm2", "660f381ed1"),
        ("pabsd %xmm9, %xmm12", "66450f381ee1"),
        ("pabsd %xmm15, %xmm0", "66410f381ec7"),
        ("pabsd 8(%rbp), %xmm3", "660f381e5d08"),
        ("pabsd (%r12), %xmm8", "66450f381e0424"),
        ("pabsd (%r13,%rax,4), %xmm15", "66450f381e7c8500"),
        ("pminsb %xmm1, %xmm2", "660f3838d1"),
        ("pminsb %xmm9, %xmm12", "66450f3838e1"),
        ("pminsb %xmm15, %xmm0", "66410f3838c7"),
        ("pminsb 8(%rbp), %xmm3", "660f38385d08"),
        ("pminsb (%r12), %xmm8", "66450f38380424"),
        ("pminsb (%r13,%rax,4), %xmm15", "66450f38387c8500"),
        ("pminsw %xmm1, %xmm2", "660fead1"),
        ("pminsw %xmm9, %xmm12", "66450feae1"),
        ("pminsw %xmm15, %xmm0", "66410feac7"),
        ("pminsw 8(%rbp), %xmm3", "660fea5d08"),
        ("pminsw (%r12), %xmm8", "66450fea0424"),
        ("pminsw (%r13,%rax,4), %xmm15", "66450fea7c8500"),
        ("pminsd %xmm1, %xmm2", "660f3839d1"),
        ("pminsd %xmm9, %xmm12", "66450f3839e1"),
        ("pminsd %xmm15, %xmm0", "66410f3839c7"),
        ("pminsd 8(%rbp), %xmm3", "660f38395d08"),
        ("pminsd (%r12), %xmm8", "66450f38390424"),
        ("pminsd (%r13,%rax,4), %xmm15", "66450f38397c8500"),
        ("pmaxsb %xmm1, %xmm2", "660f383cd1"),
        ("pmaxsb %xmm9, %xmm12", "66450f383ce1"),
        ("pmaxsb %xmm15, %xmm0", "66410f383cc7"),
        ("pmaxsb 8(%rbp), %xmm3", "660f383c5d08"),
        ("pmaxsb (%r12), %xmm8", "66450f383c0424"),
        ("pmaxsb (%r13,%rax,4), %xmm15", "66450f383c7c8500"),
        ("pmaxsw %xmm1, %xmm2", "660feed1"),
        ("pmaxsw %xmm9, %xmm12", "66450feee1"),
        ("pmaxsw %xmm15, %xmm0", "66410feec7"),
        ("pmaxsw 8(%rbp), %xmm3", "660fee5d08"),
        ("pmaxsw (%r12), %xmm8", "66450fee0424"),
        ("pmaxsw (%r13,%rax,4), %xmm15", "66450fee7c8500"),
        ("pmaxsd %xmm1, %xmm2", "660f383dd1"),
        ("pmaxsd %xmm9, %xmm12", "66450f383de1"),
        ("pmaxsd %xmm15, %xmm0", "66410f383dc7"),
        ("pmaxsd 8(%rbp), %xmm3", "660f383d5d08"),
        ("pmaxsd (%r12), %xmm8", "66450f383d0424"),
        ("pmaxsd (%r13,%rax,4), %xmm15", "66450f383d7c8500"),
        ("pminub %xmm1, %xmm2", "660fdad1"),
        ("pminub %xmm9, %xmm12", "66450fdae1"),
        ("pminub %xmm15, %xmm0", "66410fdac7"),
        ("pminub 8(%rbp), %xmm3", "660fda5d08"),
        ("pminub (%r12), %xmm8", "66450fda0424"),
        ("pminub (%r13,%rax,4), %xmm15", "66450fda7c8500"),
        ("pminuw %xmm1, %xmm2", "660f383ad1"),
        ("pminuw %xmm9, %xmm12", "66450f383ae1"),
        ("pminuw %xmm15, %xmm0", "66410f383ac7"),
        ("pminuw 8(%rbp), %xmm3", "660f383a5d08"),
        ("pminuw (%r12), %xmm8", "66450f383a0424"),
        ("pminuw (%r13,%rax,4), %xmm15", "66450f383a7c8500"),
        ("pminud %xmm1, %xmm2", "660f383bd1"),
        ("pminud %xmm9, %xmm12", "66450f383be1"),
        ("pminud %xmm15, %xmm0", "66410f383bc7"),
        ("pminud 8(%rbp), %xmm3", "660f383b5d08"),
        ("pminud (%r12), %xmm8", "66450f383b0424"),
        ("pminud (%r13,%rax,4), %xmm15", "66450f383b7c8500"),
        ("pmaxub %xmm1, %xmm2", "660fded1"),
        ("pmaxub %xmm9, %xmm12", "66450fdee1"),
        ("pmaxub %xmm15, %xmm0", "66410fdec7"),
        ("pmaxub 8(%rbp), %xmm3", "660fde5d08"),
        ("pmaxub (%r12), %xmm8", "66450fde0424"),
        ("pmaxub (%r13,%rax,4), %xmm15", "66450fde7c8500"),
        ("pmaxuw %xmm1, %xmm2", "660f383ed1"),
        ("pmaxuw %xmm9, %xmm12", "66450f383ee1"),
        ("pmaxuw %xmm15, %xmm0", "66410f383ec7"),
        ("pmaxuw 8(%rbp), %xmm3", "660f383e5d08"),
        ("pmaxuw (%r12), %xmm8", "66450f383e0424"),
        ("pmaxuw (%r13,%rax,4), %xmm15", "66450f383e7c8500"),
        ("pmaxud %xmm1, %xmm2", "660f383fd1"),
        ("pmaxud %xmm9, %xmm12", "66450f383fe1"),
        ("pmaxud %xmm15, %xmm0", "66410f383fc7"),
        ("pmaxud 8(%rbp), %xmm3", "660f383f5d08"),
        ("pmaxud (%r12), %xmm8", "66450f383f0424"),
        ("pmaxud (%r13,%rax,4), %xmm15", "66450f383f7c8500"),
        ("pcmpeqq %xmm1, %xmm2", "660f3829d1"),
        ("pcmpeqq %xmm9, %xmm12", "66450f3829e1"),
        ("pcmpeqq %xmm15, %xmm0", "66410f3829c7"),
        ("pcmpeqq 8(%rbp), %xmm3", "660f38295d08"),
        ("pcmpeqq (%r12), %xmm8", "66450f38290424"),
        ("pcmpeqq (%r13,%rax,4), %xmm15", "66450f38297c8500"),
        ("pcmpgtq %xmm1, %xmm2", "660f3837d1"),
        ("pcmpgtq %xmm9, %xmm12", "66450f3837e1"),
        ("pcmpgtq %xmm15, %xmm0", "66410f3837c7"),
        ("pcmpgtq 8(%rbp), %xmm3", "660f38375d08"),
        ("pcmpgtq (%r12), %xmm8", "66450f38370424"),
        ("pcmpgtq (%r13,%rax,4), %xmm15", "66450f38377c8500"),
        ("pmuldq %xmm1, %xmm2", "660f3828d1"),
        ("pmuldq %xmm9, %xmm12", "66450f3828e1"),
        ("pmuldq %xmm15, %xmm0", "66410f3828c7"),
        ("pmuldq 8(%rbp), %xmm3", "660f38285d08"),
        ("pmuldq (%r12), %xmm8", "66450f38280424"),
        ("pmuldq (%r13,%rax,4), %xmm15", "66450f38287c8500"),
        ("pmulhw %xmm1, %xmm2", "660fe5d1"),
        ("pmulhw %xmm9, %xmm12", "66450fe5e1"),
        ("pmulhw %xmm15, %xmm0", "66410fe5c7"),
        ("pmulhw 8(%rbp), %xmm3", "660fe55d08"),
        ("pmulhw (%r12), %xmm8", "66450fe50424"),
        ("pmulhw (%r13,%rax,4), %xmm15", "66450fe57c8500"),
        ("pmulhuw %xmm1, %xmm2", "660fe4d1"),
        ("pmulhuw %xmm9, %xmm12", "66450fe4e1"),
        ("pmulhuw %xmm15, %xmm0", "66410fe4c7"),
        ("pmulhuw 8(%rbp), %xmm3", "660fe45d08"),
        ("pmulhuw (%r12), %xmm8", "66450fe40424"),
        ("pmulhuw (%r13,%rax,4), %xmm15", "66450fe47c8500"),
        ("pmulhrsw %xmm1, %xmm2", "660f380bd1"),
        ("pmulhrsw %xmm9, %xmm12", "66450f380be1"),
        ("pmulhrsw %xmm15, %xmm0", "66410f380bc7"),
        ("pmulhrsw 8(%rbp), %xmm3", "660f380b5d08"),
        ("pmulhrsw (%r12), %xmm8", "66450f380b0424"),
        ("pmulhrsw (%r13,%rax,4), %xmm15", "66450f380b7c8500"),
        ("pmaddubsw %xmm1, %xmm2", "660f3804d1"),
        ("pmaddubsw %xmm9, %xmm12", "66450f3804e1"),
        ("pmaddubsw %xmm15, %xmm0", "66410f3804c7"),
        ("pmaddubsw 8(%rbp), %xmm3", "660f38045d08"),
        ("pmaddubsw (%r12), %xmm8", "66450f38040424"),
        ("pmaddubsw (%r13,%rax,4), %xmm15", "66450f38047c8500"),
        ("psadbw %xmm1, %xmm2", "660ff6d1"),
        ("psadbw %xmm9, %xmm12", "66450ff6e1"),
        ("psadbw %xmm15, %xmm0", "66410ff6c7"),
        ("psadbw 8(%rbp), %xmm3", "660ff65d08"),
        ("psadbw (%r12), %xmm8", "66450ff60424"),
        ("psadbw (%r13,%rax,4), %xmm15", "66450ff67c8500"),
        ("phaddw %xmm1, %xmm2", "660f3801d1"),
        ("phaddw %xmm9, %xmm12", "66450f3801e1"),
        ("phaddw %xmm15, %xmm0", "66410f3801c7"),
        ("phaddw 8(%rbp), %xmm3", "660f38015d08"),
        ("phaddw (%r12), %xmm8", "66450f38010424"),
        ("phaddw (%r13,%rax,4), %xmm15", "66450f38017c8500"),
        ("phaddd %xmm1, %xmm2", "660f3802d1"),
        ("phaddd %xmm9, %xmm12", "66450f3802e1"),
        ("phaddd %xmm15, %xmm0", "66410f3802c7"),
        ("phaddd 8(%rbp), %xmm3", "660f38025d08"),
        ("phaddd (%r12), %xmm8", "66450f38020424"),
        ("phaddd (%r13,%rax,4), %xmm15", "66450f38027c8500"),
        ("phaddsw %xmm1, %xmm2", "660f3803d1"),
        ("phaddsw %xmm9, %xmm12", "66450f3803e1"),
        ("phaddsw %xmm15, %xmm0", "66410f3803c7"),
        ("phaddsw 8(%rbp), %xmm3", "660f38035d08"),
        ("phaddsw (%r12), %xmm8", "66450f38030424"),
        ("phaddsw (%r13,%rax,4), %xmm15", "66450f38037c8500"),
        ("phsubw %xmm1, %xmm2", "660f3805d1"),
        ("phsubw %xmm9, %xmm12", "66450f3805e1"),
        ("phsubw %xmm15, %xmm0", "66410f3805c7"),
        ("phsubw 8(%rbp), %xmm3", "660f38055d08"),
        ("phsubw (%r12), %xmm8", "66450f38050424"),
        ("phsubw (%r13,%rax,4), %xmm15", "66450f38057c8500"),
        ("phsubd %xmm1, %xmm2", "660f3806d1"),
        ("phsubd %xmm9, %xmm12", "66450f3806e1"),
        ("phsubd %xmm15, %xmm0", "66410f3806c7"),
        ("phsubd 8(%rbp), %xmm3", "660f38065d08"),
        ("phsubd (%r12), %xmm8", "66450f38060424"),
        ("phsubd (%r13,%rax,4), %xmm15", "66450f38067c8500"),
        ("phsubsw %xmm1, %xmm2", "660f3807d1"),
        ("phsubsw %xmm9, %xmm12", "66450f3807e1"),
        ("phsubsw %xmm15, %xmm0", "66410f3807c7"),
        ("phsubsw 8(%rbp), %xmm3", "660f38075d08"),
        ("phsubsw (%r12), %xmm8", "66450f38070424"),
        ("phsubsw (%r13,%rax,4), %xmm15", "66450f38077c8500"),
        ("pmovsxbw %xmm1, %xmm2", "660f3820d1"),
        ("pmovsxbw %xmm9, %xmm12", "66450f3820e1"),
        ("pmovsxbw %xmm15, %xmm0", "66410f3820c7"),
        ("pmovsxbw 8(%rbp), %xmm3", "660f38205d08"),
        ("pmovsxbw (%r12), %xmm8", "66450f38200424"),
        ("pmovsxbw (%r13,%rax,4), %xmm15", "66450f38207c8500"),
        ("pmovsxdq %xmm1, %xmm2", "660f3825d1"),
        ("pmovsxdq %xmm9, %xmm12", "66450f3825e1"),
        ("pmovsxdq %xmm15, %xmm0", "66410f3825c7"),
        ("pmovsxdq 8(%rbp), %xmm3", "660f38255d08"),
        ("pmovsxdq (%r12), %xmm8", "66450f38250424"),
        ("pmovsxdq (%r13,%rax,4), %xmm15", "66450f38257c8500"),
        ("pmovzxbw %xmm1, %xmm2", "660f3830d1"),
        ("pmovzxbw %xmm9, %xmm12", "66450f3830e1"),
        ("pmovzxbw %xmm15, %xmm0", "66410f3830c7"),
        ("pmovzxbw 8(%rbp), %xmm3", "660f38305d08"),
        ("pmovzxbw (%r12), %xmm8", "66450f38300424"),
        ("pmovzxbw (%r13,%rax,4), %xmm15", "66450f38307c8500"),
        ("pmovzxdq %xmm1, %xmm2", "660f3835d1"),
        ("pmovzxdq %xmm9, %xmm12", "66450f3835e1"),
        ("pmovzxdq %xmm15, %xmm0", "66410f3835c7"),
        ("pmovzxdq 8(%rbp), %xmm3", "660f38355d08"),
        ("pmovzxdq (%r12), %xmm8", "66450f38350424"),
        ("pmovzxdq (%r13,%rax,4), %xmm15", "66450f38357c8500"),
        ("ptest %xmm1, %xmm2", "660f3817d1"),
        ("ptest %xmm9, %xmm12", "66450f3817e1"),
        ("ptest %xmm15, %xmm0", "66410f3817c7"),
        ("ptest 8(%rbp), %xmm3", "660f38175d08"),
        ("ptest (%r12), %xmm8", "66450f38170424"),
        ("ptest (%r13,%rax,4), %xmm15", "66450f38177c8500"),
        ("phminposuw %xmm1, %xmm2", "660f3841d1"),
        ("phminposuw %xmm9, %xmm12", "66450f3841e1"),
        ("phminposuw %xmm15, %xmm0", "66410f3841c7"),
        ("phminposuw 8(%rbp), %xmm3", "660f38415d08"),
        ("phminposuw (%r12), %xmm8", "66450f38410424"),
        ("phminposuw (%r13,%rax,4), %xmm15", "66450f38417c8500"),
        ("paddsb %xmm1, %xmm2", "660fecd1"),
        ("paddsb %xmm9, %xmm12", "66450fece1"),
        ("paddsb %xmm15, %xmm0", "66410fecc7"),
        ("paddsb 8(%rbp), %xmm3", "660fec5d08"),
        ("paddsb (%r12), %xmm8", "66450fec0424"),
        ("paddsb (%r13,%rax,4), %xmm15", "66450fec7c8500"),
        ("paddusw %xmm1, %xmm2", "660fddd1"),
        ("paddusw %xmm9, %xmm12", "66450fdde1"),
        ("paddusw %xmm15, %xmm0", "66410fddc7"),
        ("paddusw 8(%rbp), %xmm3", "660fdd5d08"),
        ("paddusw (%r12), %xmm8", "66450fdd0424"),
        ("paddusw (%r13,%rax,4), %xmm15", "66450fdd7c8500"),
        ("psubsb %xmm1, %xmm2", "660fe8d1"),
        ("psubsb %xmm9, %xmm12", "66450fe8e1"),
        ("psubsb %xmm15, %xmm0", "66410fe8c7"),
        ("psubsb 8(%rbp), %xmm3", "660fe85d08"),
        ("psubsb (%r12), %xmm8", "66450fe80424"),
        ("psubsb (%r13,%rax,4), %xmm15", "66450fe87c8500"),
        ("psubusw %xmm1, %xmm2", "660fd9d1"),
        ("psubusw %xmm9, %xmm12", "66450fd9e1"),
        ("psubusw %xmm15, %xmm0", "66410fd9c7"),
        ("psubusw 8(%rbp), %xmm3", "660fd95d08"),
        ("psubusw (%r12), %xmm8", "66450fd90424"),
        ("psubusw (%r13,%rax,4), %xmm15", "66450fd97c8500"),
        ("aesenc %xmm1, %xmm2", "660f38dcd1"),
        ("aesenc %xmm9, %xmm12", "66450f38dce1"),
        ("aesenc %xmm15, %xmm0", "66410f38dcc7"),
        ("aesenc 8(%rbp), %xmm3", "660f38dc5d08"),
        ("aesenc (%r12), %xmm8", "66450f38dc0424"),
        ("aesenc (%r13,%rax,4), %xmm15", "66450f38dc7c8500"),
        ("aesdeclast %xmm1, %xmm2", "660f38dfd1"),
        ("aesdeclast %xmm9, %xmm12", "66450f38dfe1"),
        ("aesdeclast %xmm15, %xmm0", "66410f38dfc7"),
        ("aesdeclast 8(%rbp), %xmm3", "660f38df5d08"),
        ("aesdeclast (%r12), %xmm8", "66450f38df0424"),
        ("aesdeclast (%r13,%rax,4), %xmm15", "66450f38df7c8500"),
        ("aesimc %xmm1, %xmm2", "660f38dbd1"),
        ("aesimc %xmm9, %xmm12", "66450f38dbe1"),
        ("aesimc %xmm15, %xmm0", "66410f38dbc7"),
        ("aesimc 8(%rbp), %xmm3", "660f38db5d08"),
        ("aesimc (%r12), %xmm8", "66450f38db0424"),
        ("aesimc (%r13,%rax,4), %xmm15", "66450f38db7c8500"),
        ("maskmovdqu %xmm1, %xmm2", "660ff7d1"),
        ("maskmovdqu %xmm9, %xmm12", "66450ff7e1"),
        ("maskmovdqu %xmm15, %xmm0", "66410ff7c7"),
        ("blendvps %xmm0, %xmm1, %xmm2", "660f3814d1"),
        ("blendvps %xmm0, %xmm9, %xmm12", "66450f3814e1"),
        ("blendvps %xmm0, (%r13), %xmm8", "66450f38144500"),
        ("blendvpd %xmm0, %xmm1, %xmm2", "660f3815d1"),
        ("blendvpd %xmm0, %xmm9, %xmm12", "66450f3815e1"),
        ("blendvpd %xmm0, (%r13), %xmm8", "66450f38154500"),
        ("pblendvb %xmm0, %xmm1, %xmm2", "660f3810d1"),
        ("pblendvb %xmm0, %xmm9, %xmm12", "66450f3810e1"),
        ("pblendvb %xmm0, (%r13), %xmm8", "66450f38104500"),
        ("pblendw $255, %xmm1, %xmm2", "660f3a0ed1ff"),
        ("pblendw $255, %xmm9, %xmm12", "66450f3a0ee1ff"),
        ("pblendw $255, (%r12), %xmm8", "66450f3a0e0424ff"),
        ("palignr $15, %xmm1, %xmm2", "660f3a0fd10f"),
        ("palignr $15, %xmm9, %xmm12", "66450f3a0fe10f"),
        ("palignr $15, (%r12), %xmm8", "66450f3a0f04240f"),
        ("pclmulqdq $17, %xmm1, %xmm2", "660f3a44d111"),
        ("pclmulqdq $17, %xmm9, %xmm12", "66450f3a44e111"),
        ("pclmulqdq $17, (%r12), %xmm8", "66450f3a44042411"),
        ("blendps $15, %xmm1, %xmm2", "660f3a0cd10f"),
        ("blendps $15, %xmm9, %xmm12", "66450f3a0ce10f"),
        ("blendps $15, (%r12), %xmm8", "66450f3a0c04240f"),
        ("blendpd $3, %xmm1, %xmm2", "660f3a0dd103"),
        ("blendpd $3, %xmm9, %xmm12", "66450f3a0de103"),
        ("blendpd $3, (%r12), %xmm8", "66450f3a0d042403"),
        ("insertps $78, %xmm1, %xmm2", "660f3a21d14e"),
        ("insertps $78, %xmm9, %xmm12", "66450f3a21e14e"),
        ("insertps $78, (%r12), %xmm8", "66450f3a2104244e"),
        ("dpps $255, %xmm1, %xmm2", "660f3a40d1ff"),
        ("dpps $255, %xmm9, %xmm12", "66450f3a40e1ff"),
        ("dpps $255, (%r12), %xmm8", "66450f3a400424ff"),
        ("dppd $51, %xmm1, %xmm2", "660f3a41d133"),
        ("dppd $51, %xmm9, %xmm12", "66450f3a41e133"),
        ("dppd $51, (%r12), %xmm8", "66450f3a41042433"),
        ("mpsadbw $7, %xmm1, %xmm2", "660f3a42d107"),
        ("mpsadbw $7, %xmm9, %xmm12", "66450f3a42e107"),
        ("mpsadbw $7, (%r12), %xmm8", "66450f3a42042407"),
        ("roundps $11, %xmm1, %xmm2", "660f3a08d10b"),
        ("roundps $11, %xmm9, %xmm12", "66450f3a08e10b"),
        ("roundps $11, (%r12), %xmm8", "66450f3a0804240b"),
        ("roundpd $2, %xmm1, %xmm2", "660f3a09d102"),
        ("roundpd $2, %xmm9, %xmm12", "66450f3a09e102"),
        ("roundpd $2, (%r12), %xmm8", "66450f3a09042402"),
        ("roundss $3, %xmm1, %xmm2", "660f3a0ad103"),
        ("roundss $3, %xmm9, %xmm12", "66450f3a0ae103"),
        ("roundss $3, (%r12), %xmm8", "66450f3a0a042403"),
        ("roundsd $4, %xmm1, %xmm2", "660f3a0bd104"),
        ("roundsd $4, %xmm9, %xmm12", "66450f3a0be104"),
        ("roundsd $4, (%r12), %xmm8", "66450f3a0b042404"),
        ("cmpps $7, %xmm1, %xmm2", "0fc2d107"),
        ("cmpps $7, %xmm9, %xmm12", "450fc2e107"),
        ("cmpps $7, (%r12), %xmm8", "450fc2042407"),
        ("cmppd $7, %xmm1, %xmm2", "660fc2d107"),
        ("cmppd $7, %xmm9, %xmm12", "66450fc2e107"),
        ("cmppd $7, (%r12), %xmm8", "66450fc2042407"),
        ("aeskeygenassist $255, %xmm1, %xmm2", "660f3adfd1ff"),
        ("aeskeygenassist $255, %xmm9, %xmm12", "66450f3adfe1ff"),
        ("aeskeygenassist $255, (%r12), %xmm8", "66450f3adf0424ff"),
        ("pcmpistri $1, %xmm1, %xmm2", "660f3a63d101"),
        ("pcmpistri $1, %xmm9, %xmm12", "66450f3a63e101"),
        ("pcmpistri $1, (%r12), %xmm8", "66450f3a63042401"),
        ("pcmpestrm $1, %xmm1, %xmm2", "660f3a60d101"),
        ("pcmpestrm $1, %xmm9, %xmm12", "66450f3a60e101"),
        ("pcmpestrm $1, (%r12), %xmm8", "66450f3a60042401"),
        ("movntdq %xmm1, (%rax)", "660fe708"),
        ("movntdq %xmm9, 8(%r12)", "66450fe74c2408"),
        ("movntps %xmm1, (%rax)", "0f2b08"),
        ("movntps %xmm9, 8(%r12)", "450f2b4c2408"),
        ("movntpd %xmm1, (%rax)", "660f2b08"),
        ("movntpd %xmm9, 8(%r12)", "66450f2b4c2408"),
        ("lddqu (%rax), %xmm1", "f20ff008"),
        ("lddqu (%r13), %xmm9", "f2450ff04d00"),
        ("movntdqa (%rax), %xmm1", "660f382a08"),
        ("movntdqa (%r12,%rax,8), %xmm15", "66450f382a3cc4"),
        ("pmovmskb %xmm1, %eax", "660fd7c1"),
        ("pmovmskb %xmm15, %r9d", "66450fd7cf"),
        ("movmskps %xmm1, %eax", "0f50c1"),
        ("movmskps %xmm15, %r9d", "450f50cf"),
        ("movmskpd %xmm1, %eax", "660f50c1"),
        ("movmskpd %xmm15, %r9d", "66450f50cf"),
        ("pextrw $0, %xmm1, %eax", "660fc5c100"),
        ("pextrw $0, %xmm9, %r12d", "66450fc5e100"),
        ("pextrw $0, %xmm1, (%rax)", "660f3a150800"),
        ("pextrw $0, %xmm9, (%r13)", "66450f3a154d0000"),
        ("pextrw $0, %xmm15, 1024(%rbx)", "66440f3a15bb0004000000"),
        ("pextrw $0, %xmm8, (%r12,%r13,2)", "66470f3a15046c00"),
        ("pextrw $7, %xmm1, %eax", "660fc5c107"),
        ("pextrw $7, %xmm9, %r12d", "66450fc5e107"),
        ("pextrw $7, %xmm1, (%rax)", "660f3a150807"),
        ("pextrw $7, %xmm9, (%r13)", "66450f3a154d0007"),
        ("pextrw $7, %xmm15, 1024(%rbx)", "66440f3a15bb0004000007"),
        ("pextrw $7, %xmm8, (%r12,%r13,2)", "66470f3a15046c07"),
        ("extractps $1, %xmm1, %eax", "660f3a17c801"),
        ("extractps $3, %xmm9, (%r13)", "66450f3a174d0003"),
        ("crc32b %al, %eax", "f20f38f0c0"),
        ("crc32b %r9b, %r12d", "f2450f38f0e1"),
        ("crc32w %ax, %ecx", "66f20f38f1c8"),
        ("crc32l (%r13), %eax", "f2410f38f14500"),
        ("crc32q %rax, %rcx", "f2480f38f1c8"),
        ("crc32q (%r12), %r15", "f24d0f38f13c24"),
        ("movntil %eax, (%rax)", "0fc300"),
        ("movntiq %r9, 8(%r13)", "4d0fc34d08"),
        ("prefetcht0 (%rax)", "0f1808"),
        ("prefetchnta (%r12)", "410f180424"),
        ("prefetcht2 8(%r13)", "410f185d08"),
        ("ldmxcsr (%rax)", "0fae10"),
        ("stmxcsr (%r13)", "410fae5d00"),
        ("sfence", "0faef8"),
        ("lfence", "0faee8"),
        ("pause", "f390"),
        ("cvtss2si %xmm1, %eax", "f30f2dc1"),
        ("cvtsd2si %xmm9, %r12", "f24d0f2de1"),
        ("cvttss2si %xmm15, %rax", "f3490f2cc7"),
        ("rcpss %xmm1, %xmm2", "f30f53d1"),
        ("rsqrtss (%r13), %xmm9", "f3450f524d00"),
        ("movlps (%rax), %xmm1", "0f1208"),
        ("movlps %xmm9, (%r13)", "450f134d00"),
        ("movhpd (%rax), %xmm1", "660f1608"),
        ("movhpd %xmm9, (%r13)", "66450f174d00"),
        ("movupd (%rax), %xmm1", "660f1008"),
        ("movupd %xmm9, (%r13)", "66450f114d00"),
        ("movlhps %xmm1, %xmm2", "0f16d1"),
        ("movlhps %xmm15, %xmm8", "450f16c7"),
        ("vpsllvd %xmm1, %xmm2, %xmm3", "c4e26947d9"),
        ("vpsllvd %xmm9, %xmm12, %xmm15", "c4421947f9"),
        ("vpsllvd %ymm1, %ymm2, %ymm3", "c4e26d47d9"),
        ("vpsllvd %ymm9, %ymm12, %ymm15", "c4421d47f9"),
        ("vpsllvd (%r13), %ymm2, %ymm3", "c4c26d475d00"),
        ("vpsllvd (%r12,%rax,8), %xmm2, %xmm3", "c4c269471cc4"),
        ("vpsrlvd %xmm1, %xmm2, %xmm3", "c4e26945d9"),
        ("vpsrlvd %xmm9, %xmm12, %xmm15", "c4421945f9"),
        ("vpsrlvd %ymm1, %ymm2, %ymm3", "c4e26d45d9"),
        ("vpsrlvd %ymm9, %ymm12, %ymm15", "c4421d45f9"),
        ("vpsrlvd (%r13), %ymm2, %ymm3", "c4c26d455d00"),
        ("vpsrlvd (%r12,%rax,8), %xmm2, %xmm3", "c4c269451cc4"),
        ("vpsravd %xmm1, %xmm2, %xmm3", "c4e26946d9"),
        ("vpsravd %xmm9, %xmm12, %xmm15", "c4421946f9"),
        ("vpsravd %ymm1, %ymm2, %ymm3", "c4e26d46d9"),
        ("vpsravd %ymm9, %ymm12, %ymm15", "c4421d46f9"),
        ("vpsravd (%r13), %ymm2, %ymm3", "c4c26d465d00"),
        ("vpsravd (%r12,%rax,8), %xmm2, %xmm3", "c4c269461cc4"),
        ("vpsllvq %xmm1, %xmm2, %xmm3", "c4e2e947d9"),
        ("vpsllvq %xmm9, %xmm12, %xmm15", "c4429947f9"),
        ("vpsllvq %ymm1, %ymm2, %ymm3", "c4e2ed47d9"),
        ("vpsllvq %ymm9, %ymm12, %ymm15", "c4429d47f9"),
        ("vpsllvq (%r13), %ymm2, %ymm3", "c4c2ed475d00"),
        ("vpsllvq (%r12,%rax,8), %xmm2, %xmm3", "c4c2e9471cc4"),
        ("vpsrlvq %xmm1, %xmm2, %xmm3", "c4e2e945d9"),
        ("vpsrlvq %xmm9, %xmm12, %xmm15", "c4429945f9"),
        ("vpsrlvq %ymm1, %ymm2, %ymm3", "c4e2ed45d9"),
        ("vpsrlvq %ymm9, %ymm12, %ymm15", "c4429d45f9"),
        ("vpsrlvq (%r13), %ymm2, %ymm3", "c4c2ed455d00"),
        ("vpsrlvq (%r12,%rax,8), %xmm2, %xmm3", "c4c2e9451cc4"),
        ("vpaddb %xmm1, %xmm2, %xmm3", "c5e9fcd9"),
        ("vpaddb %ymm9, %ymm12, %ymm15", "c4411dfcf9"),
        ("vpaddb (%r13), %ymm2, %ymm3", "c4c16dfc5d00"),
        ("vpand %xmm1, %xmm2, %xmm3", "c5e9dbd9"),
        ("vpand %ymm9, %ymm12, %ymm15", "c4411ddbf9"),
        ("vpand (%r13), %ymm2, %ymm3", "c4c16ddb5d00"),
        ("vpshufb %xmm1, %xmm2, %xmm3", "c4e26900d9"),
        ("vpshufb %ymm9, %ymm12, %ymm15", "c4421d00f9"),
        ("vpshufb (%r13), %ymm2, %ymm3", "c4c26d005d00"),
        ("vpminsd %xmm1, %xmm2, %xmm3", "c4e26939d9"),
        ("vpminsd %ymm9, %ymm12, %ymm15", "c4421d39f9"),
        ("vpminsd (%r13), %ymm2, %ymm3", "c4c26d395d00"),
        ("vpmulld %xmm1, %xmm2, %xmm3", "c4e26940d9"),
        ("vpmulld %ymm9, %ymm12, %ymm15", "c4421d40f9"),
        ("vpmulld (%r13), %ymm2, %ymm3", "c4c26d405d00"),
        ("vaddps %xmm1, %xmm2, %xmm3", "c5e858d9"),
        ("vaddps %ymm9, %ymm12, %ymm15", "c4411c58f9"),
        ("vaddps (%r13), %ymm2, %ymm3", "c4c16c585d00"),
        ("vpcmpeqq %xmm1, %xmm2, %xmm3", "c4e26929d9"),
        ("vpcmpeqq %ymm9, %ymm12, %ymm15", "c4421d29f9"),
        ("vpcmpeqq (%r13), %ymm2, %ymm3", "c4c26d295d00"),
        ("vpsadbw %xmm1, %xmm2, %xmm3", "c5e9f6d9"),
        ("vpsadbw %ymm9, %ymm12, %ymm15", "c4411df6f9"),
        ("vpsadbw (%r13), %ymm2, %ymm3", "c4c16df65d00"),
        ("vpabsd %xmm1, %xmm2", "c4e2791ed1"),
        ("vpabsd (%r13), %ymm15", "c4427d1e7d00"),
        ("vptest %xmm1, %xmm2", "c4e27917d1"),
        ("vptest (%r13), %ymm15", "c4427d177d00"),
        ("vsqrtpd %xmm1, %xmm2", "c5f951d1"),
        ("vsqrtpd (%r13), %ymm15", "c4417d517d00"),
        ("vpmovzxbw %xmm1, %xmm2", "c4e27930d1"),
        ("vpmovzxbw %xmm9, %ymm15", "c4427d30f9"),
        ("vbroadcastss %xmm1, %xmm2", "c4e27918d1"),
        ("vbroadcastss %xmm9, %ymm15", "c4427d18f9"),
        ("vpbroadcastd %xmm1, %xmm2", "c4e27958d1"),
        ("vpbroadcastd %xmm9, %ymm15", "c4427d58f9"),
        ("vmovntdqa (%rax), %xmm2", "c4e2792a10"),
        ("vmovntdqa (%r13), %ymm15", "c4427d2a7d00"),
        ("vlddqu (%rax), %xmm2", "c5fbf010"),
        ("vlddqu (%r13), %ymm15", "c4417ff07d00"),
        ("vblendvps %xmm4, %xmm1, %xmm2, %xmm3", "c4e3694ad940"),
        ("vpblendvb %ymm12, %ymm9, %ymm15, %ymm1", "c4c3054cc9c0"),
        ("vpalignr $15, %ymm1, %ymm2, %ymm3", "c4e36d0fd90f"),
        ("vroundps $11, %ymm1, %ymm2", "c4e37d08d10b"),
        ("vinserti128 $1, %xmm1, %ymm2, %ymm3", "c4e36d38d901"),
        ("vextracti128 $1, %ymm9, %xmm12", "c4437d39cc01"),
        ("vextractf128 $1, %ymm1, (%r13)", "c4c37d194d0001"),
        ("vperm2i128 $1, %ymm1, %ymm2, %ymm3", "c4e36d46d901"),
        ("vpermq $27, %ymm9, %ymm15", "c443fd00f91b"),
        ("vpermd %ymm1, %ymm2, %ymm3", "c4e26d36d9"),
        ("vpermilps $27, %ymm1, %ymm2", "c4e37d04d11b"),
        ("vpermilps %ymm1, %ymm2, %ymm3", "c4e26d0cd9"),
        ("vpsllw $7, %ymm1, %ymm2", "c5ed71f107"),
        ("vpsrldq $4, %xmm9, %xmm12", "c4c11973d904"),
        ("vpextrw $3, %xmm1, %eax", "c5f9c5c103"),
        ("vpextrw $3, %xmm9, (%r13)", "c44379154d0003"),
        ("vpextrb $15, %xmm15, (%r12)", "c44379143c240f"),
        ("vpextrq $1, %xmm9, %r12", "c443f916cc01"),
        ("vpinsrb $1, %eax, %xmm1, %xmm2", "c4e37120d001"),
        ("vpinsrq $1, %r12, %xmm9, %xmm15", "c443b122fc01"),
        ("vpinsrw $7, (%r13), %xmm1, %xmm2", "c4c171c4550007"),
        ("vextractps $3, %xmm9, %r12d", "c4437917cc03"),
        ("vpmovmskb %ymm9, %r12d", "c4417dd7e1"),
        ("vmovmskpd %xmm15, %eax", "c4c17950c7"),
        ("vmovq %xmm8, %xmm0", "c579d6c0"),
        ("vmovq %rax, %xmm9", "c461f96ec8"),
        ("vmovq %xmm9, (%r13)", "c44179d64d00"),
        ("vmovq (%r12), %xmm15", "c4417a7e3c24"),
        ("vmovss %xmm15, %xmm7, %xmm0", "c54211f8"),
        ("vmovsd (%r13), %xmm9", "c4417b104d00"),
        ("vmovss %xmm9, (%r12)", "c4417a110c24"),
        ("vmovhps (%r13), %xmm1, %xmm2", "c4c170165500"),
        ("vmovlpd %xmm9, (%r12)", "c44179130c24"),
        ("vcvtsi2sdl %eax, %xmm1, %xmm2", "c5f32ad0"),
        ("vcvtsi2ssq %r12, %xmm9, %xmm15", "c441b22afc"),
        ("vcvttsd2si %xmm9, %r12", "c441fb2ce1"),
        ("vcvtss2si (%r13), %eax", "c4c17a2d4500"),
        ("vfmadd132ps %ymm1, %ymm2, %ymm3", "c4e26d98d9"),
        ("vfmadd213sd %xmm9, %xmm12, %xmm15", "c44299a9f9"),
        ("vfnmsub231pd (%r13), %ymm2, %ymm3", "c4c2edbe5d00"),
        ("vfmaddsub132ps %ymm9, %ymm12, %ymm15", "c4421d96f9"),
        ("vmaskmovps (%r13), %ymm1, %ymm2", "c4c2752c5500"),
        ("vpmaskmovq %xmm9, %xmm12, (%r12)", "c442998e0c24"),
        ("vzeroall", "c5fc77"),
        ("vzeroupper", "c5f877"),
        ("vmovaps %ymm9, (%r13)", "c4417c294d00"),
        ("vmovupd (%r12), %ymm15", "c4417d103c24"),
        ("vmovntdq %ymm9, (%r12)", "c4417de70c24"),
    ];
    for (asm, expected) in CASES {
        let tmp = TempDir::new("pinned_extended");
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

/// Forms GNU as refuses are not quietly encoded as something else.
#[test]
fn invalid_packed_forms_are_rejected() {
    for asm in [
        "movntdq %xmm1, %xmm2",
        "movntps %xmm1, %xmm2",
        "lddqu %xmm1, %xmm2",
        "movntdqa %xmm1, %xmm2",
        "vlddqu %xmm1, %xmm2",
        "pblendvb %xmm1, %xmm2, %xmm3",
        "blendvps %xmm4, %xmm2, %xmm3",
        "extractps $1, %eax, %xmm1",
        "vpsllw %ymm1, %ymm2, %ymm3",
        "vpsrad %ymm1, %ymm2, %ymm3",
        "pmovmskb %xmm1, %rax",
        "prefetcht0 %xmm1",
        "crc32b %eax, %eax",
        "crc32q %eax, %rax",
        "cvtss2si %xmm1, %ax",
    ] {
        let tmp = TempDir::new("invalid_packed");
        let path = tmp.path().join("invalid.o");
        assert!(
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(&format!(".text\n{asm}\n"), &path)
                .is_err(),
            "{asm}"
        );
    }
}
