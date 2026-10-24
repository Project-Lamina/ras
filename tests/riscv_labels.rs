use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;

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

fn assemble(name: &str, source: &str) -> Result<Vec<u8>, ras::error::RasError> {
    let dir = std::env::temp_dir().join(format!("ras_riscv_labels_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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
fn riscv32_object_output_is_refused_rather_than_written_as_elf64() {
    let output = std::env::temp_dir().join(format!("ras_riscv32_{}.o", std::process::id()));
    let result = Ras::new(TargetArchitecture::Riscv32, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(".text\n  ret\n", &output);
    assert!(result.is_err());
    assert!(!output.exists());
}
