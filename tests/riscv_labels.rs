use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;

mod common;
use common::TempDir;

fn section(bytes: &[u8], wanted: &str) -> Vec<u8> {
    let u64_at = |at| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
    let headers = u64_at(40);
    let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
    let strings = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;
    let names = &bytes[u64_at(headers + strings * 64 + 24)..];
    let wanted = format!("{wanted}\0");
    for i in 1..count {
        let h = headers + i * 64;
        let name = u32::from_le_bytes(bytes[h..h + 4].try_into().unwrap()) as usize;
        if names[name..].starts_with(wanted.as_bytes()) {
            return bytes[u64_at(h + 24)..u64_at(h + 24) + u64_at(h + 32)].to_vec();
        }
    }
    panic!("Missing {wanted}")
}

fn section32(bytes: &[u8], wanted: &str) -> Vec<u8> {
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let headers = u32_at(32);
    let count = u16::from_le_bytes(bytes[48..50].try_into().unwrap()) as usize;
    let strings = u16::from_le_bytes(bytes[50..52].try_into().unwrap()) as usize;
    let names = &bytes[u32_at(headers + strings * 40 + 16)..];
    let wanted = format!("{wanted}\0");
    for i in 1..count {
        let h = headers + i * 40;
        let name = u32_at(h) as usize;
        if names[name..].starts_with(wanted.as_bytes()) {
            return bytes[u32_at(h + 16)..u32_at(h + 16) + u32_at(h + 20)].to_vec();
        }
    }
    panic!("Missing {wanted}")
}

fn entsize32(bytes: &[u8], wanted: &str) -> u32 {
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let headers = u32_at(32) as usize;
    let count = u16::from_le_bytes(bytes[48..50].try_into().unwrap()) as usize;
    let strings = u16::from_le_bytes(bytes[50..52].try_into().unwrap()) as usize;
    let names = &bytes[u32_at(headers + strings * 40 + 16) as usize..];
    let wanted = format!("{wanted}\0");
    for i in 1..count {
        let h = headers + i * 40;
        if names[u32_at(h) as usize..].starts_with(wanted.as_bytes()) {
            return u32_at(h + 36);
        }
    }
    panic!("Missing {wanted}")
}

fn assemble(name: &str, source: &str) -> Result<Vec<u8>, ras::error::RasError> {
    let tmp = TempDir::new("riscv_labels");
    let dir = tmp.path();
    let output = dir.join(format!("{name}.o"));
    Ras::new(TargetArchitecture::Riscv64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(source, &output)?;
    Ok(std::fs::read(&output).unwrap())
}

fn words(text: &[u8]) -> Vec<u32> {
    text.chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn backward_and_forward_labels_resolve_to_clang_encodings() {
    let source = "\
.text
f:
  addi a0, zero, 1
loop:
  addi a0, a0, -1
  bne a0, zero, loop
  beq a0, zero, done
  jal zero, f
  bgt a1, a2, loop
done:
  ret
";
    let object = assemble("resolved", source).unwrap();
    assert_eq!(
        words(&section(&object, ".text")),
        [
            0x0010_0513,
            0xfff5_0513,
            0xfe05_1ee3,
            0x0005_0663,
            0xff1f_f06f,
            0xfeb6_48e3,
            0x0000_8067,
        ]
    );
}

#[test]
fn branch_offsets_reach_exactly_four_kib() {
    let reach =
        |skip: usize| format!(".text\nstart:\n  beq a0, a1, end\n  .skip {skip}\nend:\n  ret\n");
    assert!(assemble("branch_limit", &reach(4090)).is_ok());
    let err = assemble("branch_past", &reach(4092)).unwrap_err();
    assert!(
        err.to_string().contains("RISC-V branch out of range"),
        "{err}"
    );
}

#[test]
fn jump_offsets_reach_exactly_one_mib() {
    let reach = |skip: usize| format!(".text\nstart:\n  j end\n  .skip {skip}\nend:\n  ret\n");
    assert!(assemble("jump_limit", &reach(1_048_570)).is_ok());
    let err = assemble("jump_past", &reach(1_048_572)).unwrap_err();
    assert!(
        err.to_string().contains("RISC-V jump out of range"),
        "{err}"
    );
}

#[test]
fn global_and_undefined_jump_targets_stay_relocations() {
    let source = ".text\n.globl f\nf:\n  j f\n  jal ra, external\n";
    let object = assemble("relocated", source).unwrap();
    assert_eq!(
        words(&section(&object, ".text")),
        [0x0000_006f, 0x0000_00ef]
    );
}

#[test]
fn align_counts_powers_of_two_and_pads_text_with_nops() {
    let object = assemble("align", ".text\n  addi a0, a0, 1\n  .align 4\n  ret\n").unwrap();
    assert_eq!(
        words(&section(&object, ".text")),
        [
            0x0015_0513,
            0x0000_0013,
            0x0000_0013,
            0x0000_0013,
            0x0000_8067
        ]
    );
}

#[test]
fn riscv32_object_output_is_elf32() {
    let tmp = TempDir::new("riscv32");
    let output = tmp.path().join("riscv32.o");
    Ras::new(TargetArchitecture::Riscv32, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(
            ".text\n.globl f\nf:\n  addi a0, a0, 1\n  call h\n  ret\n",
            &output,
        )
        .expect("assemble");
    let object = std::fs::read(&output).expect("read");
    assert_eq!(&object[0..4], b"\x7fELF");
    assert_eq!(object[4], 1, "ELFCLASS32");
    assert_eq!(
        u16::from_le_bytes([object[18], object[19]]),
        243,
        "EM_RISCV"
    );
    assert_eq!(u16::from_le_bytes([object[40], object[41]]), 52, "e_ehsize");
    assert_eq!(
        u16::from_le_bytes([object[46], object[47]]),
        40,
        "e_shentsize"
    );
    assert_eq!(
        section32(&object, ".text"),
        [
            0x13, 0x05, 0x15, 0x00, 0x97, 0x00, 0x00, 0x00, 0xe7, 0x80, 0x00, 0x00, 0x67, 0x80,
            0x00, 0x00,
        ]
    );
    let relocations = section32(&object, ".rela.text");
    assert_eq!(relocations.len(), 12, "one Elf32_Rela");
    assert_eq!(entsize32(&object, ".rela.text"), 12, "sh_entsize");
    assert_eq!(entsize32(&object, ".symtab"), 16, "sh_entsize");
    assert_eq!(
        u32::from_le_bytes([
            relocations[0],
            relocations[1],
            relocations[2],
            relocations[3]
        ]),
        4,
        "r_offset"
    );
    let info = u32::from_le_bytes([
        relocations[4],
        relocations[5],
        relocations[6],
        relocations[7],
    ]);
    assert_eq!(info & 0xff, 19, "R_RISCV_CALL_PLT");
    let symbol = info >> 8;
    assert_ne!(symbol, 0, "symbol index");
    let symbols = section32(&object, ".symtab");
    assert_eq!(symbols.len() % 16, 0, "Elf32_Sym entries");
    let strings = section32(&object, ".strtab");
    let entry = &symbols[symbol as usize * 16..][..16];
    let name = u32::from_le_bytes(entry[0..4].try_into().unwrap()) as usize;
    let end = strings[name..].iter().position(|b| *b == 0).unwrap() + name;
    assert_eq!(&strings[name..end], b"h");
    assert_eq!(entry[12], 1 << 4, "STB_GLOBAL, STT_NOTYPE");
    assert_eq!(
        u16::from_le_bytes([entry[14], entry[15]]),
        0,
        "undefined section"
    );
    let f = (1..symbols.len() / 16)
        .find(|i| {
            let at = u32::from_le_bytes(symbols[i * 16..i * 16 + 4].try_into().unwrap()) as usize;
            strings[at..].starts_with(b"f\0")
        })
        .expect("f");
    assert_eq!(
        u16::from_le_bytes([symbols[f * 16 + 14], symbols[f * 16 + 15]]),
        1,
        "f lives in .text"
    );
}

#[test]
fn gas_data_directives_encode_like_clang() {
    let object = assemble(
        "data",
        ".data\n.word 1, -1\n.half 0x1234\n.dword 0x0102030405060708\n.2byte 7\n.4byte 9\n.8byte 10\n.float 1.5\n.double -2.25\n.single 0.5\n",
    )
    .unwrap();
    assert_eq!(
        section(&object, ".data"),
        [
            0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x34, 0x12, 0x08, 0x07, 0x06, 0x05,
            0x04, 0x03, 0x02, 0x01, 0x07, 0x00, 0x09, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x02, 0xc0, 0x00, 0x00, 0x00, 0x3f,
        ]
    );
}
