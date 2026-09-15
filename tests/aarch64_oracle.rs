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
fn integer_float_memory_and_branch_encodings_match_clang() {
    let dir = std::env::temp_dir().join(format!("ras_aarch64_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("input.s");
    let reference = dir.join("reference.o");
    let output = dir.join("ras.o");
    let asm = r#"
.section .rodata
message: .asciz "ok"
.text
.globl entry
entry:
stp x29, x30, [sp, #-16]!
mov x29, sp
sub sp, sp, #32
mov x9, #42
mov w10, #-1
movz x11, #123, lsl #32
movk x11, #456, lsl #48
add x12, x9, x11
add w12, w9, w11
sub w12, w9, w11
and x12, x9, x11
and w12, w9, w11
mul x12, x9, x11
mul w12, w9, w11
msub x12, x9, x11, x10
msub w12, w9, w11, w10
sdiv x12, x9, x11
udiv w12, w9, w11
cmp x12, #0
cmp w12, w11
csel x12, x9, x11, eq
cset x12, ne
uxtb x9, w9
fmov d0, x9
fmov x9, d0
fmov s0, w9
fmov w9, s0
fmul d0, d0, d1
fmul s0, s0, s1
str x9, [sp, #8]
ldr x9, [sp, #8]
str w9, [sp, #4]
ldr w9, [sp, #4]
strb w9, [sp]
ldrb w9, [sp]
strh w9, [sp, #2]
ldrh w9, [sp, #2]
stur x9, [x29, #-8]
ldur x9, [x29, #-8]
stp w9, w10, [sp, #8]
ldp w9, w10, [sp, #8]
stp w9, w10, [sp, #-8]!
ldp w9, w10, [sp], #8
adrp x0, message
add x0, x0, :lo12:message
b.ne .Ldone
bl external
b .Ldone
.Ldone:
dmb sy
add sp, sp, #32
ldp x29, x30, [sp], #16
ret
"#;
    std::fs::write(&source, asm).unwrap();
    let result = Command::new("clang")
        .args(["--target=aarch64-linux-gnu", "-c"])
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
    Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(asm, &output)
        .unwrap();
    assert_eq!(
        text(&std::fs::read(&output).unwrap()),
        text(&std::fs::read(&reference).unwrap())
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn duplicate_symbols_are_rejected() {
    let output = std::env::temp_dir().join("ras_duplicate_aarch64.o");
    assert!(
        Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
            .unwrap()
            .assemble("function: ret\nfunction: ret\n", &output)
            .is_err()
    );
}
