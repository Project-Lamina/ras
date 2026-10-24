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

#[test]
fn bl_to_same_section_local_label_resolves_in_place() {
    let output = std::env::temp_dir().join(format!("ras_bl_local_{}.o", std::process::id()));
    let asm = ".text\nentry:\n  bl local\n  ret\nlocal:\n  ret\n";
    Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(asm, &output)
        .unwrap();
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(text(&bytes)[..4], 0x9400_0002u32.to_le_bytes());
    std::fs::remove_file(output).unwrap();
}

#[test]
fn hints_system_calls_and_register_branches_encode_like_clang() {
    let output = std::env::temp_dir().join(format!("ras_hints_aarch64_{}.o", std::process::id()));
    let asm = ".text\nnop\nyield\nwfi\nsvc #0\nbrk #0x1234\nbr x1\nblr x30\n";
    Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(asm, &output)
        .unwrap();
    let words: Vec<u32> = text(&std::fs::read(&output).unwrap())
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    std::fs::remove_file(output).unwrap();
    assert_eq!(
        words,
        [
            0xd503_201f,
            0xd503_203f,
            0xd503_207f,
            0xd400_0001,
            0xd420_0000 | (0x1234 << 5),
            0xd61f_0020,
            0xd63f_03c0,
        ]
    );
}

#[test]
fn branches_to_global_same_section_labels_are_relocated() {
    let output = std::env::temp_dir().join(format!("ras_global_b_{}.o", std::process::id()));
    let asm = ".text\n.globl g\ng:\n  b g\n  b.eq g\n  ret\n";
    Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(asm, &output)
        .unwrap();
    let words: Vec<u32> = text(&std::fs::read(&output).unwrap())
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    std::fs::remove_file(output).unwrap();
    assert_eq!(words[..2], [0x1400_0000, 0x5400_0000]);
}

fn assemble_bytes(asm: &str) -> Result<Vec<u8>, String> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let output = std::env::temp_dir().join(format!(
        "ras_aarch64_words_{}_{}.o",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let result = Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(asm, &output)
        .map_err(|e| e.to_string())
        .and_then(|()| std::fs::read(&output).map_err(|e| e.to_string()));
    let _ = std::fs::remove_file(&output);
    result
}

fn assemble_words(asm: &str) -> Result<Vec<u32>, String> {
    assemble_bytes(asm).map(|bytes| {
        text(&bytes)
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    })
}

/// (offset, relocation type, symbol name, addend) for every RELA entry in an object file.
fn relocations(bytes: &[u8]) -> Vec<(u64, u32, String, i64)> {
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
            out.push((u64_at(entry), info as u32, name, u64_at(entry + 16) as i64));
        }
    }
    out
}

#[test]
fn add_sub_and_cmp_immediates_encode_the_lsl_12_form() {
    assert_eq!(
        assemble_words(".text\nadd x0, x1, #4, lsl #12\n").unwrap(),
        [0x9140_1020]
    );
    assert_eq!(
        assemble_words(".text\nadd x0, x1, #4096\n").unwrap(),
        [0x9140_0420]
    );
    assert_eq!(
        assemble_words(".text\nsub w0, w1, #1, lsl #12\n").unwrap(),
        [0x5140_0420]
    );
    assert_eq!(
        assemble_words(".text\ncmp x0, #1, lsl #12\n").unwrap(),
        [0xf140_041f]
    );
}

#[test]
fn add_sub_and_cmp_immediates_reject_unrepresentable_shifts() {
    for asm in [
        ".text\nadd x0, x1, #1, lsl #24\n",
        ".text\nadd x0, x1, #1, lsr #12\n",
        ".text\nadd x0, x1, #4097\n",
        ".text\nadd x0, x1, #16781312\n",
        ".text\nadd x0, x1, #4096, lsl #12\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn add_sub_and_cmp_shifted_registers_encode_their_shift() {
    assert_eq!(
        assemble_words(".text\nadd x0, x1, x2, lsl #3\n").unwrap(),
        [0x8b02_0c20]
    );
    assert_eq!(
        assemble_words(".text\nsub x0, x1, x2, lsr #5\n").unwrap(),
        [0xcb42_1420]
    );
    assert_eq!(
        assemble_words(".text\ncmp x0, x1, lsl #2\n").unwrap(),
        [0xeb01_081f]
    );
}

#[test]
fn add_sub_and_cmp_reject_out_of_range_or_unknown_shifts() {
    for asm in [
        ".text\nadd x0, x1, x2, lsl #64\n",
        ".text\nadd w0, w1, w2, lsl #32\n",
        ".text\nadd x0, x1, x2, ror #1\n",
        ".text\nadd x0, x1, x2, lsl\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn add_sub_and_cmp_with_stack_pointer_use_the_extended_form() {
    assert_eq!(
        assemble_words(".text\nadd sp, sp, x1\n").unwrap(),
        [0x8b21_63ff]
    );
    assert_eq!(
        assemble_words(".text\nadd x0, sp, x1\n").unwrap(),
        [0x8b21_63e0]
    );
    assert_eq!(
        assemble_words(".text\nsub sp, x1, x2\n").unwrap(),
        [0xcb22_603f]
    );
    assert_eq!(
        assemble_words(".text\ncmp sp, x0\n").unwrap(),
        [0xeb20_63ff]
    );
    assert_eq!(
        assemble_words(".text\nadd wsp, w1, w2\n").unwrap(),
        [0x0b22_403f]
    );
    assert_eq!(
        assemble_words(".text\nadd x0, sp, x1, lsl #4\n").unwrap(),
        [0x8b21_73e0]
    );
    for asm in [
        ".text\nadd x0, sp, x1, lsr #2\n",
        ".text\nadd x0, sp, x1, lsl #5\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn add_sub_and_cmp_reject_mixed_register_widths_and_fp_operands() {
    for asm in [
        ".text\nadd w0, x1, #0\n",
        ".text\nadd x0, w1, #1\n",
        ".text\nadd w0, w1, x2\n",
        ".text\nadd x0, x1, w2\n",
        ".text\nsub w0, x1, x2\n",
        ".text\ncmp w0, x1\n",
        ".text\nadd x0, x1, sp\n",
        ".text\ncmp x0, sp\n",
        ".text\nadd d0, x1, x2\n",
        ".text\nadd x0, s1, x2\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn extra_operands_on_fixed_arity_instructions_are_rejected() {
    for asm in [
        ".text\nret x0, x1\n",
        ".text\nbr x0, x1\n",
        ".text\nsvc #0, #1\n",
        ".text\ndmb sy, sy\n",
        ".text\nmul x0, x1, x2, x3\n",
        ".text\nmsub x0, x1, x2, x3, x4\n",
        ".text\ncsel x0, x1, x2, eq, junk\n",
        ".text\ncset x0, eq, x1\n",
        ".text\nuxtb w0, w1, w2\n",
        ".text\nmov x0, x1, lsl #3\n",
        ".text\nmov x0, #1, lsl #16\n",
        ".text\nmovz x0, #1, lsl #16, extra\n",
        ".text\nmovk x0, #1, lsl #16, extra\n",
        ".text\nfmul d0, d1, d2, d3\n",
        ".text\nldr x9, [sp, #8, lsl #3]\n",
        ".text\nldr w9, [sp, #8, lsl #3]\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn w_register_mov_immediates_must_fit_in_32_bits() {
    for asm in [
        ".text\nmov w0, #0x1ffff0000\n",
        ".text\nmov w0, #8589869056\n",
        ".text\nmov w0, #4294967296\n",
        ".text\nmov w0, #-2147483649\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
    assert_eq!(
        assemble_words(".text\nmov w0, #-2147483648\n").unwrap(),
        [0x52b0_0000]
    );
}

#[test]
fn data_processing_registers_must_share_one_width_and_avoid_sp() {
    for asm in [
        ".text\nmul x0, w1, x2\n",
        ".text\nmul w0, x1, x2\n",
        ".text\nsdiv x0, x1, w2\n",
        ".text\nudiv w0, x1, x2\n",
        ".text\nmsub w0, w1, w2, x3\n",
        ".text\ncsel w0, x1, x2, eq\n",
        ".text\nand w0, x1, x2\n",
        ".text\nmul sp, x1, x2\n",
        ".text\ncsel sp, x1, x2, eq\n",
        ".text\ncset w0, al\n",
        ".text\ncset w0, nv\n",
        ".text\nuxtb w0, x1\n",
        ".text\nret w0\n",
        ".text\nret wzr\n",
        ".text\nbr w0\n",
        ".text\nblr w0\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
    assert_eq!(assemble_words(".text\nret\n").unwrap(), [0xd65f_03c0]);
    assert_eq!(assemble_words(".text\nret xzr\n").unwrap(), [0xd65f_03e0]);
    assert_eq!(
        assemble_words(".text\ncset w0, hs\n").unwrap(),
        [0x1a9f_37e0]
    );
}

#[test]
fn fp_scalar_and_pair_loads_and_stores_encode_their_register_size() {
    assert_eq!(
        assemble_words(".text\nldr d0, [x1]\n").unwrap(),
        [0xfd40_0020]
    );
    assert_eq!(
        assemble_words(".text\nstr s0, [sp, #4]\n").unwrap(),
        [0xbd00_07e0]
    );
    assert_eq!(
        assemble_words(".text\nldp d0, d1, [sp]\n").unwrap(),
        [0x6d40_07e0]
    );
    assert_eq!(
        assemble_words(".text\nldp d0, d1, [sp, #8]\n").unwrap(),
        [0x6d40_87e0]
    );
    assert_eq!(
        assemble_words(".text\nldp d0, d1, [x0], #32\n").unwrap(),
        [0x6cc2_0400]
    );
}

#[test]
fn loads_and_stores_reject_bad_base_and_data_registers() {
    for asm in [
        ".text\nldr x0, [w1]\n",
        ".text\nldr x0, [xzr, #8]\n",
        ".text\nldr x0, [wsp]\n",
        ".text\nldrb x9, [sp]\n",
        ".text\nstrh x9, [sp]\n",
        ".text\nldr sp, [x0]\n",
        ".text\nldr wsp, [x0]\n",
        ".text\nldr w9, [x0, #8, lsl #3]\n",
        ".text\nldp x0, w1, [sp]\n",
        ".text\nldp d0, s1, [sp]\n",
        ".text\nldp d0, d1, [sp, #4]\n",
        ".text\nldp x0, x1, [x0], #32\n",
        ".text\nstp x0, x1, [x0, #-16]!\n",
        ".text\nstp x0\n",
        ".text\nldr\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn symbol_operands_keep_their_addend_in_relocations() {
    let rel = relocations(
        &assemble_bytes(".text\nbl sym+4\nadrp x0, sym+8\nadd x0, x0, :lo12:sym-4\n").unwrap(),
    );
    assert!(rel.contains(&(0, 283, "sym".into(), 4)), "{rel:?}");
    assert!(rel.contains(&(4, 275, "sym".into(), 8)), "{rel:?}");
    assert!(rel.contains(&(8, 277, "sym".into(), -4)), "{rel:?}");
}

#[test]
fn local_branch_targets_honor_their_addend() {
    assert_eq!(
        assemble_words(".text\nb tgt+4\ntgt:\nret\n").unwrap(),
        [0x1400_0002, 0xd65f_03c0]
    );
}
