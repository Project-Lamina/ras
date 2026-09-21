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
    asm.push_str("movl %r9d, %r10d\nmovw %r9w, %r10w\nmovb %r9b, %r10b\nmovb %ah, %cl\nmovb %spl, %dil\nmovq $42, %rax\nmovq $4294967295, %rax\nmovq $-42, %r9\nmovzbl (%rsp), %ecx\nmovzbq (%rax), %r9\nmovzwq (%r13), %r10\nmovzwl 8(%r12), %ecx\ncmovel %eax, %ecx\ncmovel %r9d, %r10d\nmulss %xmm1, %xmm0\nmulsd %xmm10, %xmm9\nret\n");
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
