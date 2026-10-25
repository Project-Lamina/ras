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
fn li_boundary_immediates_match_clang_rv64g() {
    let dir = std::env::temp_dir().join(format!("ras_riscv_li_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("input.s");
    let reference = dir.join("reference.o");
    let output = dir.join("ras.o");
    let values = [
        "0",
        "1",
        "-1",
        "2047",
        "2048",
        "-2048",
        "-2049",
        "4095",
        "4096",
        "-4096",
        "0x7ff",
        "0x800",
        "0xfff",
        "0x1000",
        "0x1800",
        "0x7fffffff",
        "-0x80000000",
        "0x80000000",
        "0xffffffff",
        "0x100000000",
        "0x123456789abcdef0",
        "-9223372036854775808",
        "9223372036854775807",
    ];
    let asm: String = values
        .iter()
        .map(|value| format!("li a0, {value}\n"))
        .fold(".text\n".to_string(), |acc, line| acc + &line);
    std::fs::write(&source, &asm).unwrap();
    let result = Command::new("clang")
        .args(["--target=riscv64-linux-gnu", "-march=rv64g", "-c"])
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
    Ras::new(TargetArchitecture::Riscv64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(&asm, &output)
        .unwrap();
    assert_eq!(
        text(&std::fs::read(&output).unwrap()),
        text(&std::fs::read(&reference).unwrap())
    );
    std::fs::remove_dir_all(dir).unwrap();
}
