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

#[test]
fn got_and_lo12_relocations_use_their_own_kinds() {
    let rel = relocations(
        &assemble_bytes(
            ".text\nadrp x0, :got:sym\nldr x0, [x0, :got_lo12:sym]\nldr x1, [x2, #:lo12:sym]\n",
        )
        .unwrap(),
    );
    assert_eq!(
        rel,
        [
            (0, 311, "sym".to_string(), 0),
            (4, 312, "sym".to_string(), 0),
            (8, 286, "sym".to_string(), 0),
        ]
    );
    assert_eq!(
        assemble_words(
            ".text\nadrp x0, :got:sym\nldr x0, [x0, :got_lo12:sym]\nldr x1, [x2, :lo12:sym]\n"
        )
        .unwrap(),
        [0x9000_0000, 0xf940_0000, 0xf940_0041]
    );
}

#[test]
fn unsupported_relocation_modifiers_are_rejected() {
    for asm in [
        ".text\nadrp x0, :gottprel:sym\n",
        ".text\nadrp x0, :tlsdesc:sym\n",
        ".text\nldr w0, [x1, :lo12:sym]\n",
        ".text\nldr x0, [x0, :tprel_lo12_nc:sym]\n",
        ".text\nmovz x0, :abs_g0:sym\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn compare_and_test_branches_encode_their_label_offsets() {
    assert_eq!(
        assemble_words(".text\ncbz x0, tgt\ntgt:\nret\n").unwrap(),
        [0xb400_0020, 0xd65f_03c0]
    );
    assert_eq!(
        assemble_words(".text\ncbnz w1, tgt\ntgt:\nret\n").unwrap(),
        [0x3500_0021, 0xd65f_03c0]
    );
    assert_eq!(
        assemble_words(".text\ntbz x0, #3, tgt\ntgt:\nret\n").unwrap(),
        [0x3618_0020, 0xd65f_03c0]
    );
    assert_eq!(
        assemble_words(".text\ntbnz w0, #31, tgt\ntgt:\nret\n").unwrap(),
        [0x37f8_0020, 0xd65f_03c0]
    );
    assert_eq!(
        assemble_words(".text\ntbnz x0, #40, tgt\ntgt:\nret\n").unwrap(),
        [0xb740_0020, 0xd65f_03c0]
    );
}

#[test]
fn compare_and_test_branches_reject_out_of_range_labels() {
    for asm in [
        ".text\ncbz x0, tgt\n.space 1048572\ntgt:\nret\n",
        ".text\ntbz x0, #3, tgt\n.space 32764\ntgt:\nret\n",
        ".text\ntgt:\n.space 32772\ntbz x0, #3, tgt\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
    assert!(assemble_words(".text\ncbz x0, tgt\n.space 1048568\ntgt:\nret\n").is_ok());
    assert!(assemble_words(".text\ntbz x0, #3, tgt\n.space 32760\ntgt:\nret\n").is_ok());
}

#[test]
fn compare_and_test_branches_reject_bad_operands() {
    for asm in [
        ".text\ncbz x0\n",
        ".text\ncbz sp, tgt\n",
        ".text\ntbz w0, #32, tgt\n",
        ".text\ntbz x0, #64, tgt\n",
        ".text\ntbz x0, tgt\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn logical_immediates_encode_as_bitmask_fields() {
    assert_eq!(
        assemble_words(".text\nand x0, x1, #0xff\n").unwrap(),
        [0x9240_1c20]
    );
    assert_eq!(
        assemble_words(".text\norr w0, w1, #0x80000000\n").unwrap(),
        [0x3201_0020]
    );
    assert_eq!(
        assemble_words(".text\neor x0, x1, #0x5555555555555555\n").unwrap(),
        [0xd200_f020]
    );
    assert_eq!(
        assemble_words(".text\nands w0, w1, #0x1\n").unwrap(),
        [0x7200_0020]
    );
    assert_eq!(
        assemble_words(".text\ntst x0, #0xff\n").unwrap(),
        [0xf240_1c1f]
    );
}

#[test]
fn logical_register_forms_encode_orr_eor_ands_and_tst() {
    assert_eq!(
        assemble_words(".text\norr x0, x1, x2\n").unwrap(),
        [0xaa02_0020]
    );
    assert_eq!(
        assemble_words(".text\neor w0, w1, w2\n").unwrap(),
        [0x4a02_0020]
    );
    assert_eq!(
        assemble_words(".text\nands x0, x1, x2\n").unwrap(),
        [0xea02_0020]
    );
    assert_eq!(
        assemble_words(".text\ntst w0, w1\n").unwrap(),
        [0x6a01_001f]
    );
}

#[test]
fn logical_immediates_reject_values_that_cannot_be_encoded() {
    for asm in [
        ".text\nand x0, x1, #0\n",
        ".text\nand x0, x1, #-1\n",
        ".text\nand w0, w1, #0x100000000\n",
        ".text\nand w0, w1, #0x101\n",
        ".text\norr x0, x1, #0x12345\n",
        ".text\nand x0, sp, #1\n",
        ".text\nand x0, x1, w2\n",
        ".text\ntst x0, #0\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn wide_mov_immediates_lower_to_move_wide_chains() {
    assert_eq!(
        assemble_words(".text\nmov x0, #0x12345678\n").unwrap(),
        [0xd28a_cf00, 0xf2a2_4680]
    );
    assert_eq!(
        assemble_words(".text\nmov x0, #0x123456789abcdef0\n").unwrap(),
        [0xd29b_de00, 0xf2b3_5780, 0xf2ca_cf00, 0xf2e2_4680]
    );
    assert_eq!(
        assemble_words(".text\nmov x0, #0xffffffffffff1234\n").unwrap(),
        [0x929d_b960]
    );
    assert_eq!(
        assemble_words(".text\nmov w0, #-1\n").unwrap(),
        [0x1280_0000]
    );
    assert_eq!(
        assemble_words(".text\nmov w0, #0xffff0000\n").unwrap(),
        [0x52bf_ffe0]
    );
    assert_eq!(
        assemble_words(".text\nmov x0, #0x0000ffff0000ffff\n").unwrap(),
        [0xb200_3fe0]
    );
    assert_eq!(
        assemble_words(".text\nmov x0, #0xffff1234ffff5678\n").unwrap(),
        [0x9295_30e0, 0xf2c2_4680]
    );
    assert_eq!(
        assemble_words(".text\nmov w0, #74565\n").unwrap(),
        [0x5284_68a0, 0x72a0_0020]
    );
    assert_eq!(
        assemble_words(".text\nmovn x0, #1\n").unwrap(),
        [0x9280_0020]
    );
    assert_eq!(
        assemble_words(".text\nmovn w0, #0xffff, lsl #16\n").unwrap(),
        [0x12bf_ffe0]
    );
    for asm in [
        ".text\nmovn w0, #0x10000\n",
        ".text\nmovn x0, #1, lsl #3\n",
        ".text\nmov d0, #1\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn pre_and_post_indexed_loads_and_stores_encode_their_writeback() {
    assert_eq!(
        assemble_words(".text\nldr x9, [x0, #8]!\n").unwrap(),
        [0xf840_8c09]
    );
    assert_eq!(
        assemble_words(".text\nstr w9, [x0], #8\n").unwrap(),
        [0xb800_8409]
    );
    assert_eq!(
        assemble_words(".text\nldrb w9, [x0], #-1\n").unwrap(),
        [0x385f_f409]
    );
    assert_eq!(
        assemble_words(".text\nldr d0, [x1], #16\n").unwrap(),
        [0xfc41_0420]
    );
    assert_eq!(
        assemble_words(".text\nstr x9, [sp, #-16]!\n")
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn writeback_forms_reject_unencodable_offsets_and_registers() {
    for asm in [
        ".text\nldr x9, [x0, #256]!\n",
        ".text\nldr x9, [x0], #256\n",
        ".text\nldr x9, [x0], x1\n",
        ".text\nldr x0, [x0, #8]!\n",
        ".text\nstr x0, [x0], #8\n",
        ".text\nldr x9, [x0], #8, #8\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn barrier_options_and_carry_condition_aliases_encode() {
    assert_eq!(
        assemble_words(".text\ndmb ish\ndmb ishld\ndsb sy\nisb\n").unwrap(),
        [0xd503_3bbf, 0xd503_39bf, 0xd503_3f9f, 0xd503_3fdf]
    );
    assert_eq!(
        assemble_words(".text\ncset w0, cc\ncset w0, cs\ncsel w0, w1, w2, cs\n").unwrap(),
        [0x1a9f_27e0, 0x1a9f_37e0, 0x1a82_2020]
    );
    assert!(assemble_words(".text\ndmb foo\n").is_err());
}

#[test]
fn fmul_takes_scalar_registers_of_one_size() {
    assert_eq!(
        assemble_words(".text\nfmul d0, d1, d2\nfmul s0, s1, s2\n").unwrap(),
        [0x1e62_0820, 0x1e22_0820]
    );
    for asm in [
        ".text\nfmul x0, x1, x2\n",
        ".text\nfmul d0, s1, d2\n",
        ".text\nfmul s0, d1, d2\n",
        ".text\nfmul d0, sp, d2\n",
        ".text\nfmul d0, d1\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn scalar_fp_register_31_is_valid_but_general_register_31_is_not() {
    assert_eq!(
        assemble_words(".text\nfmul d31, d1, d2\n").unwrap(),
        [0x1e62_083f]
    );
    assert!(assemble_words(".text\nadd x0, x31, x1\n").is_err());
    assert!(assemble_words(".text\nmov w31, w0\n").is_err());
}

#[test]
fn unscaled_loads_and_stores_have_no_writeback_form() {
    for asm in [
        ".text\nldur x9, [x0, #8]!\n",
        ".text\nstur w9, [x0], #8\n",
        ".text\nldurb w9, [x0], #8\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn post_indexed_loads_and_stores_reject_an_inner_offset() {
    for asm in [
        ".text\nldr x9, [sp, #8], #8\n",
        ".text\nstr w9, [sp, #8], #8\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
    assert!(assemble_words(".text\nldr x9, [sp], #8\n").is_ok());
}

#[test]
fn register_mov_requires_matching_widths_and_general_registers() {
    assert_eq!(
        assemble_words(".text\nmov x0, sp\n").unwrap(),
        [0x9100_03e0]
    );
    assert_eq!(
        assemble_words(".text\nmov x0, x1\n").unwrap(),
        [0xaa01_03e0]
    );
    assert_eq!(
        assemble_words(".text\nmov wsp, w1\n").unwrap(),
        [0x1100_003f]
    );
    for asm in [
        ".text\nmov sp, w0\n",
        ".text\nmov w0, sp\n",
        ".text\nmov x0, wsp\n",
        ".text\nmov x0, wzr\n",
        ".text\nmov w0, xzr\n",
        ".text\nmov x0, d1\n",
        ".text\nmov w0, s1\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn symbol_operands_must_name_a_symbol() {
    for asm in [
        ".text\nadd x0, x1, :lo12:\n",
        ".text\nadrp x0, :got:\n",
        ".text\nadrp x0, \n",
        ".text\nbl +4\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn udf_encodes_bare_immediate() {
    assert_eq!(
        assemble_words(".text\nudf #0\nudf #1\nudf #0xffff\nudf #0x1234\n").unwrap(),
        [0x0000_0000, 0x0000_0001, 0x0000_ffff, 0x0000_1234]
    );
    assert!(assemble_words(".text\nudf #0x10000\n").is_err());
    assert!(assemble_words(".text\nudf\n").is_err());
    assert!(assemble_words(".text\nudf #0, #1\n").is_err());
}

#[test]
fn compare_negative_encodes_as_flag_setting_add() {
    assert_eq!(
        assemble_words(".text\ncmn x0, x1\ncmn w2, w3\ncmn x4, #16\ncmn x7, x8, lsl #3\n").unwrap(),
        [0xab01_001f, 0x2b03_005f, 0xb100_409f, 0xab08_0cff]
    );
    for asm in [".text\ncmn x0, w1\n", ".text\ncmn x0, #4097\n"] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn bitfield_extracts_cover_lsb_and_width() {
    assert_eq!(
        assemble_words(".text\nsbfx x0, x1, #0, #8\nubfx w2, w3, #16, #16\nsxtw x0, w1\n").unwrap(),
        [0x9340_1c20, 0x5310_7c62, 0x9340_7c20]
    );
    for asm in [
        ".text\nsbfx x0, x1, #0, #0\n",
        ".text\nsbfx x0, x1, #60, #8\n",
        ".text\nubfx x0, x1, #-1, #8\n",
        ".text\nsxtw w0, w1\n",
    ] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn shifts_take_both_immediate_and_register_amounts() {
    assert_eq!(
        assemble_words(".text\nlsl x9, x11, #56\nasr x12, x10, #56\nlsr w2, w3, #31\n").unwrap(),
        [0xd348_1d69, 0x9378_fd4c, 0x531f_7c62]
    );
    assert_eq!(
        assemble_words(".text\nlsl x0, x1, x2\nasrv w3, w4, w5\nrorv x12, x13, x14\n").unwrap(),
        [0x9ac2_2020, 0x1ac5_2883, 0x9ace_2dac]
    );
    for asm in [".text\nlsl x0, x1, #64\n", ".text\nasr w0, w1, #32\n"] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn signed_loads_pick_the_destination_width() {
    assert_eq!(
        assemble_words(".text\nldrsb w0, [x1]\nldrsb x0, [x1]\nldrsw x0, [x1]\n").unwrap(),
        [0x39c0_0020, 0x3980_0020, 0xb980_0020]
    );
    assert_eq!(
        assemble_words(".text\nldursb x2, [x3, #-256]\nldursw x6, [x7, #-4]\n").unwrap(),
        [0x3890_0062, 0xb89f_c0e6]
    );
    for asm in [".text\nldrsw w0, [x1]\n", ".text\nldrsb s0, [x1]\n"] {
        assert!(assemble_words(asm).is_err(), "accepted {asm:?}");
    }
}

#[test]
fn scalar_float_arithmetic_covers_all_four_operations() {
    assert_eq!(
        assemble_words(".text\nfadd s0, s0, s1\nfsub d8, d9, d10\nfdiv s11, s12, s13\n").unwrap(),
        [0x1e21_2800, 0x1e6a_3928, 0x1e2d_198b]
    );
    assert!(assemble_words(".text\nfadd s0, s1, d2\n").is_err());
}
