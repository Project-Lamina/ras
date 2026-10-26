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

#[test]
fn move_widths_extensions_and_scalar_operations_match_gas() {
    let dir = std::env::temp_dir().join(format!("ras_x86_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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
        text(&std::fs::read(&output).unwrap()),
        text(&std::fs::read(&reference).unwrap())
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn memory_move_rejects_an_unrepresentable_immediate() {
    let path = std::env::temp_dir().join(format!("ras_invalid_immediate_{}.o", std::process::id()));
    assert!(
        Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .unwrap()
            .assemble("movq $4294967295, (%rax)\n", &path)
            .is_err()
    );
}

fn oracle(asm: &str, name: &str) -> (Vec<u8>, Vec<u8>) {
    let dir = std::env::temp_dir().join(format!("ras_x86_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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
    std::fs::remove_dir_all(dir).unwrap();
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
    assert_eq!(text(&output), text(&reference));
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
    let path = std::env::temp_dir().join(format!("ras_invalid_compiler_{}.o", std::process::id()));
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
