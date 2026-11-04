//! The ARMv7-A (A32) encoder and the ELF32 ARM objects it writes.
#![cfg(any(target_os = "linux", target_os = "macos"))]

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;
use std::process::Command;

mod common;
use common::TempDir;

const PREAMBLE: &str = ".syntax unified\n.arch armv7-a\n.arch_extension idiv\n\
                        .fpu vfpv3-d16\n.eabi_attribute 28, 1\n.arm\n";

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

struct Object {
    bytes: Vec<u8>,
}

impl Object {
    fn assemble(label: &str, asm: &str) -> Self {
        let tmp = TempDir::new(label);
        let path = tmp.path().join("out.o");
        Ras::new(TargetArchitecture::Armv7, TargetOperatingSystem::Linux)
            .unwrap()
            .assemble(asm, &path)
            .unwrap();
        Self {
            bytes: std::fs::read(&path).unwrap(),
        }
    }

    fn header_count(&self) -> usize {
        u16_at(&self.bytes, 48) as usize
    }

    fn header(&self, index: usize) -> usize {
        u32_at(&self.bytes, 32) as usize + index * 40
    }

    fn names(&self) -> &[u8] {
        let strings = u16_at(&self.bytes, 50) as usize;
        &self.bytes[u32_at(&self.bytes, self.header(strings) + 16) as usize..]
    }

    fn find(&self, name: &str) -> Option<usize> {
        (1..self.header_count()).find(|index| {
            let at = u32_at(&self.bytes, self.header(*index)) as usize;
            self.names()[at..].split(|b| *b == 0).next().unwrap() == name.as_bytes()
        })
    }

    fn section(&self, name: &str) -> &[u8] {
        let header = self.header(self.find(name).unwrap_or_else(|| panic!("no {name}")));
        let start = u32_at(&self.bytes, header + 16) as usize;
        &self.bytes[start..start + u32_at(&self.bytes, header + 20) as usize]
    }

    fn field(&self, name: &str, index: usize) -> u32 {
        u32_at(
            &self.bytes,
            self.header(self.find(name).unwrap()) + index * 4,
        )
    }

    fn words(&self, name: &str) -> Vec<u32> {
        self.section(name)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32_at(word, 0))
            .collect()
    }

    /// Every relocation of `name` as (offset, symbol, type).
    fn relocations(&self, name: &str) -> Vec<(u32, String, u32)> {
        let strings = self.section(".strtab").to_vec();
        let symtab = self.section(".symtab").to_vec();
        self.section(name)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|entry| {
                let info = u32_at(entry, 4);
                let at = u32_at(&symtab[(info >> 8) as usize * 16..], 0) as usize;
                let symbol = strings[at..].split(|b| *b == 0).next().unwrap();
                (
                    u32_at(entry, 0),
                    String::from_utf8_lossy(symbol).into_owned(),
                    info & 0xff,
                )
            })
            .collect()
    }
}

fn encode(line: &str) -> u32 {
    let object = Object::assemble(
        "armv7-encode",
        &format!("{PREAMBLE}.text\n.globl m\nm:\n    {line}\n"),
    );
    object.words(".text")[0]
}

fn error(line: &str) -> String {
    let tmp = TempDir::new("armv7-refuse");
    Ras::new(TargetArchitecture::Armv7, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(
            &format!("{PREAMBLE}.text\n.globl m\nm:\n    {line}\n"),
            &tmp.path().join("out.o"),
        )
        .unwrap_err()
        .to_string()
}

/// Every mnemonic lamina's armv7 backend emits, with the A32 word it encodes
/// to. The condition occupies bits 31:28 of all of them.
const ENCODINGS: &[(&str, u32)] = &[
    ("add r4, r4, #1", 0xe2844001),
    ("add r4, r4, r5", 0xe0844005),
    ("add r4, r4, r4, lsr #5", 0xe08442a4),
    ("and r5, r5, #31", 0xe205501f),
    ("and r5, r6, r4, lsr #3", 0xe00651a4),
    ("asr r6, r4, r5", 0xe1a06554),
    ("bic sp, sp, #15", 0xe3cdd00f),
    ("bx lr", 0xe12fff1e),
    ("blx r4", 0xe12fff34),
    ("clz r4, r5", 0xe16f4f15),
    ("cmn r5, #1", 0xe3750001),
    ("cmp r4, #0", 0xe3540000),
    ("eor r6, r4, r5", 0xe0246005),
    ("ldr r4, [r11, #-8]", 0xe51b4008),
    ("ldr r5, [r4]", 0xe5945000),
    ("ldrb r4, [sp]", 0xe5dd4000),
    ("ldrsb r5, [r4]", 0xe1d450d0),
    ("ldrsh r5, [r4]", 0xe1d450f0),
    ("lsl r6, r4, r5", 0xe1a06514),
    ("lsr r4, r4, #5", 0xe1a042a4),
    ("lsr r6, r4, r5", 0xe1a06534),
    ("mov r0, r1", 0xe1a00001),
    ("mov r4, #7", 0xe3a04007),
    ("moveq r4, #1", 0x03a04001),
    ("movge r4, #1", 0xa3a04001),
    ("movgt r4, #1", 0xc3a04001),
    ("movhi r4, #1", 0x83a04001),
    ("movlo r4, #1", 0x33a04001),
    ("movls r4, #1", 0x93a04001),
    ("movlt r4, #1", 0xb3a04001),
    ("movmi r4, #1", 0x43a04001),
    ("movne r4, #1", 0x13a04001),
    ("movt r5, #65535", 0xe34f5fff),
    ("movw r5, #65535", 0xe30f5fff),
    ("mul r4, r4, r5", 0xe0040594),
    ("mvnlt r0, #0", 0xb3e00000),
    ("orr r4, r4, r5", 0xe1844005),
    (
        "pop {r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}",
        0xe8bd5ff0,
    ),
    (
        "push {r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}",
        0xe92d5ff0,
    ),
    ("rbit r4, r5", 0xe6ff4f35),
    ("rev r4, r5", 0xe6bf4f35),
    ("rev16 r4, r5", 0xe6bf4fb5),
    ("ror r4, r4, r5", 0xe1a04574),
    ("rsb r6, r5, #0", 0xe2656000),
    ("sdiv r6, r4, r5", 0xe716f514),
    ("str r4, [r11, #-8]", 0xe50b4008),
    ("str r5, [r4]", 0xe5845000),
    ("strb r5, [r4]", 0xe5c45000),
    ("strh r5, [r4]", 0xe1c450b0),
    ("sub r4, r11, #16", 0xe24b4010),
    ("subs r0, r0, #1", 0xe2500001),
    ("sxtb r0, r4", 0xe6af0074),
    ("sxth r0, r4", 0xe6bf0074),
    ("udf #0", 0xe7f000f0),
    ("udiv r6, r4, r5", 0xe736f514),
    ("uxtb r4, r5", 0xe6ef4075),
    ("uxth r4, r5", 0xe6ff4075),
    ("vadd.f32 s16, s16, s17", 0xee388a28),
    ("vadd.f64 d12, d12, d13", 0xee3ccb0d),
    ("vcmp.f64 d12, d13", 0xeeb4cb4d),
    ("vcvt.f32.f64 s16, d12", 0xeeb78bcc),
    ("vcvt.f32.s32 s16, s17", 0xeeb88ae8),
    ("vcvt.f32.u32 s16, s17", 0xeeb88a68),
    ("vcvt.f64.f32 d12, s16", 0xeeb7cac8),
    ("vcvt.f64.s32 d12, s16", 0xeeb8cbc8),
    ("vcvt.f64.u32 d12, s16", 0xeeb8cb48),
    ("vcvt.s32.f32 s16, s17", 0xeebd8ae8),
    ("vcvt.s32.f64 s16, d12", 0xeebd8bcc),
    ("vcvt.u32.f32 s16, s17", 0xeebc8ae8),
    ("vcvt.u32.f64 s16, d12", 0xeebc8bcc),
    ("vdiv.f32 s16, s16, s17", 0xee888a28),
    ("vdiv.f64 d12, d12, d13", 0xee8ccb0d),
    ("vldr d12, [r11, #-8]", 0xed1bcb02),
    ("vldr s16, [r12]", 0xed9c8a00),
    ("vmov s16, r4", 0xee084a10),
    ("vmov r4, s16", 0xee184a10),
    ("vmov d12, r12, r4", 0xec44cb1c),
    ("vmov r0, r1, d0", 0xec510b10),
    ("vmoveq.f64 d12, d13", 0x0eb0cb4d),
    ("vmrs APSR_nzcv, fpscr", 0xeef1fa10),
    ("vmul.f32 s16, s16, s17", 0xee288a28),
    ("vmul.f64 d12, d12, d13", 0xee2ccb0d),
    ("vneg.f64 d12, d13", 0xeeb1cb4d),
    ("vpop {d8, d9, d10, d11, d12, d13, d14, d15}", 0xecbd8b10),
    ("vpush {d8, d9, d10, d11, d12, d13, d14, d15}", 0xed2d8b10),
    ("vsqrt.f32 s16, s17", 0xeeb18ae8),
    ("vsqrt.f64 d12, d13", 0xeeb1cbcd),
    ("vstr d12, [r11, #-8]", 0xed0bcb02),
    ("vstr s16, [r12]", 0xed8c8a00),
    ("vsub.f64 d12, d12, d13", 0xee3ccb4d),
];

#[test]
fn encodes_the_instruction_set_lamina_emits() {
    for (line, expected) in ENCODINGS {
        assert_eq!(encode(line), *expected, "{line}");
    }
}

/// A conditional mnemonic is the same instruction with a different top nibble,
/// not an opcode of its own.
#[test]
fn the_condition_is_the_top_nibble() {
    let unconditional = encode("mov r4, #1");
    assert_eq!(unconditional >> 28, 0xe);
    for (line, condition) in [
        ("moveq r4, #1", 0x0),
        ("movne r4, #1", 0x1),
        ("movlo r4, #1", 0x3),
        ("movmi r4, #1", 0x4),
        ("movhi r4, #1", 0x8),
        ("movls r4, #1", 0x9),
        ("movge r4, #1", 0xa),
        ("movlt r4, #1", 0xb),
        ("movgt r4, #1", 0xc),
    ] {
        let word = encode(line);
        assert_eq!(word >> 28, condition, "{line}");
        assert_eq!(word & 0x0fff_ffff, unconditional & 0x0fff_ffff, "{line}");
    }
}

/// movw and movt carry their sixteen-bit immediate split across imm4 at bits
/// 19:16 and imm12 at bits 11:0.
#[test]
fn movw_and_movt_split_the_immediate() {
    // 0xbeef splits into imm4 0xb at bits 19:16 and imm12 0xeef.
    assert_eq!(encode("movw r0, #48879"), 0xe30b_0eef);
    assert_eq!(encode("movt r0, #48879"), 0xe34b_0eef);
    assert_eq!(encode("movw r5, #0"), 0xe3005000);
    assert_eq!(encode("movw r5, #65535"), 0xe30f5fff);
    for line in ["movw r0, #65536", "movt r0, #-1"] {
        assert!(error(line).contains("16 bits"), "{line}");
    }
}

/// An encoding ras cannot produce correctly names itself rather than guessing
/// a bit pattern.
#[test]
fn refuses_what_it_does_not_encode_by_name() {
    for (line, fragment) in [
        ("nop", "Unsupported ARM instruction: nop"),
        ("ldm r0, {r1, r2}", "Unsupported ARM instruction: ldm"),
        ("smull r0, r1, r2, r3", "Unsupported ARM instruction: smull"),
        ("bkpt #0", "Unsupported ARM instruction: bkpt"),
        (
            "mcr p15, 0, r0, c1, c0, 0",
            "Unsupported ARM instruction: mcr",
        ),
        (
            "vmla.f64 d0, d1, d2",
            "Unsupported ARM instruction: vmla.f64",
        ),
        (
            "vcvt.f16.f32 s0, s1",
            "Unsupported ARM instruction: vcvt.f16.f32",
        ),
        ("add r0, r0, #257", "is not an 8-bit value rotated"),
        (
            "ldr r0, [r1, r2]",
            "register-offset addressing is not encoded",
        ),
        ("ldr r0, [r1], #4", "post-indexed addressing is not encoded"),
        ("ldr r0, [r1, #4]!", "write-back addressing is not encoded"),
        ("ldr r0, [r1, #4096]", "does not fit in 12 bits"),
        ("ldrsb r0, [r1, #256]", "does not fit in 8 bits"),
        ("vldr s0, [r1, #3]", "is not a multiple of four within 1020"),
        (
            "vmov.f32 s0, #1.0",
            "VFP immediate operands are not encoded",
        ),
        ("vmrs r0, fpscr", "Only `vmrs APSR_nzcv, fpscr` is encoded"),
        ("vpush {d8, d10}", "VFP register list is not consecutive"),
        ("ror r0, r1, #0", "ARM shift amount 0 out of range"),
        ("mov r16, #1", "Unknown ARM register: r16"),
        ("vadd.f64 d12, d12, s3", "Unknown ARM d-register: s3"),
    ] {
        let message = error(line);
        assert!(message.contains(fragment), "{line}: {message}");
    }
}

/// ras encodes A32 only, and says so rather than assembling Thumb as ARM.
#[test]
fn refuses_directives_it_cannot_honour() {
    for (source, fragment) in [
        (".thumb\n", "selects Thumb"),
        (".syntax divided\n", "unified ARM syntax only"),
        (".arch armv8-a\n", "no build attributes for `.arch armv8-a`"),
        (".fpu neon\n", "no build attributes for `.fpu neon`"),
    ] {
        let tmp = TempDir::new("armv7-directive");
        let message = Ras::new(TargetArchitecture::Armv7, TargetOperatingSystem::Linux)
            .unwrap()
            .assemble(source, &tmp.path().join("out.o"))
            .unwrap_err()
            .to_string();
        assert!(message.contains(fragment), "{source}: {message}");
    }
}

/// A branch inside the object is resolved here, counting from the address the
/// processor computes, which is the branch plus eight. One that leaves the
/// object becomes a relocation whose addend is the displacement field itself,
/// because ARM objects carry Elf32_Rel.
#[test]
fn branches_resolve_locally_and_relocate_outward() {
    let object = Object::assemble(
        "armv7-branch",
        &format!("{PREAMBLE}.text\n.globl m\nm:\n  b 1f\n  beq 1f\n1:\n  bl other\n  bx lr\n"),
    );
    assert_eq!(
        object.words(".text"),
        [
            // b   .+8  -> (8 - (0 + 8)) / 4 == 0
            0xea00_0000,
            // beq .+4  -> (8 - (4 + 8)) / 4 == -1
            0x0aff_ffff,
            // bl  other, with the -8 bias held in the displacement
            0xebff_fffe,
            0xe12f_ff1e,
        ]
    );
    assert_eq!(
        object.relocations(".rel.text"),
        [(8, "other".to_string(), 28)]
    );
}

/// An address is built by a movw/movt pair, which the linker completes. Both
/// halves leave their immediate at zero, which is the Elf32_Rel addend.
#[test]
fn symbol_halves_become_movw_and_movt_relocations() {
    let object = Object::assemble(
        "armv7-halves",
        &format!(
            "{PREAMBLE}.data\nvalue: .long 7\n             .text\n.globl m\nm:\n  movw r4, #:lower16:value\n  movt r4, #:upper16:value\n"
        ),
    );
    assert_eq!(object.words(".text"), [0xe300_4000, 0xe340_4000]);
    assert_eq!(
        object.relocations(".rel.text"),
        [(0, "value".to_string(), 43), (4, "value".to_string(), 44),]
    );
}

/// A data word pointing at a symbol is R_ARM_ABS32, and its addend is written
/// into the word itself.
#[test]
fn data_pointers_keep_their_addend_in_the_field() {
    let object = Object::assemble(
        "armv7-data",
        &format!("{PREAMBLE}.data\ntext: .byte 1, 2, 3, 4\nptr: .long text+3\n"),
    );
    assert_eq!(object.section(".data"), [1, 2, 3, 4, 3, 0, 0, 0]);
    assert_eq!(
        object.relocations(".rel.data"),
        [(4, "text".to_string(), 2)]
    );
}

#[test]
fn writes_an_elf32_arm_object() {
    let object = Object::assemble(
        "armv7-object",
        &format!("{PREAMBLE}.text\n.globl m\nm:\n  bl other\n  bx lr\n"),
    );
    // ELFCLASS32, ELFDATA2LSB, EM_ARM, EF_ARM_EABI_VER5.
    assert_eq!(object.bytes[4], 1);
    assert_eq!(object.bytes[5], 1);
    assert_eq!(u16_at(&object.bytes, 18), 40);
    assert_eq!(u32_at(&object.bytes, 36), 0x0500_0000);
    assert_eq!(u16_at(&object.bytes, 46), 40);
    // SHT_REL with eight-byte entries, named .rel<section>, not .rela.
    assert!(object.find(".rela.text").is_none());
    assert_eq!(object.field(".rel.text", 1), 9);
    assert_eq!(object.field(".rel.text", 9), 8);
    // SHT_ARM_ATTRIBUTES, holding the aeabi subsection the directives describe.
    assert_eq!(object.field(".ARM.attributes", 1), 0x7000_0003);
    let attributes = object.section(".ARM.attributes");
    assert_eq!(attributes[0], b'A');
    assert_eq!(u32_at(attributes, 1) as usize, attributes.len() - 1);
    assert_eq!(&attributes[5..11], b"aeabi\0");
    assert_eq!(attributes[11], 1);
    assert_eq!(u32_at(attributes, 12) as usize, attributes.len() - 11);
    assert_eq!(
        &attributes[16..],
        // Tag_CPU_name "7-A", Tag_CPU_arch v7, profile A, ARM and Thumb-2 ISA,
        // Tag_FP_arch VFPv3-D16, Tag_ABI_VFP_args VFP registers, Tag_DIV_use.
        &[
            5, b'7', b'-', b'A', 0, 6, 10, 7, 0x41, 8, 1, 9, 2, 10, 4, 28, 1, 44, 2
        ]
    );
}

/// ras refuses armv7 for the object formats it has no ARM writer for.
#[test]
fn only_elf_targets_accept_armv7() {
    for os in [TargetOperatingSystem::MacOS, TargetOperatingSystem::Windows] {
        assert!(
            !ras::target::is_object_file_supported(TargetArchitecture::Armv7, os),
            "{os:?}"
        );
    }
    assert!(ras::target::is_object_file_supported(
        TargetArchitecture::Armv7,
        TargetOperatingSystem::Linux
    ));
}

/// The decisive check: an object ras assembled, linked and run.
#[test]
fn a_linked_program_runs_and_prints() {
    let have = |tool: &str| {
        Command::new(tool)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    };
    if !have("arm-linux-gnueabihf-gcc") || !have("qemu-arm-static") {
        return;
    }

    let tmp = TempDir::new("armv7-run");
    let object = tmp.path().join("program.o");
    Ras::new(TargetArchitecture::Armv7, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(
            &format!(
                "{PREAMBLE}.section .note.GNU-stack,\"\",%progbits\n                 .data\nmessage: .asciz \"armv7 %d\\n\"\n                 .text\n.globl main\nmain:\n                 \x20 push {{r4, lr}}\n                 \x20 movw r0, #:lower16:message\n  movt r0, #:upper16:message\n                 \x20 movw r1, #20\n  movw r2, #2\n  mul r1, r1, r2\n  add r1, r1, #2\n                 \x20 bl printf\n  movw r0, #0\n  pop {{r4, pc}}\n"
            ),
            &object,
        )
        .unwrap();

    let program = tmp.path().join("program");
    let link = Command::new("arm-linux-gnueabihf-gcc")
        .args(["-static", "-no-pie"])
        .arg(&object)
        .arg("-o")
        .arg(&program)
        .output()
        .unwrap();
    assert!(
        link.status.success(),
        "{}",
        String::from_utf8_lossy(&link.stderr)
    );
    let run = Command::new("qemu-arm-static")
        .arg(&program)
        .output()
        .unwrap();
    assert!(run.status.success(), "program exited {:?}", run.status);
    assert_eq!(String::from_utf8_lossy(&run.stdout), "armv7 42\n");
}
