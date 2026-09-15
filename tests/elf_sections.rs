use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[test]
fn sections_keep_contents_symbols_and_relocations() {
    let path = std::env::temp_dir().join(format!("ras_sections_{}.o", std::process::id()));
    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(
            ".section .rodata\n.globl message\nmessage: .asciz \"hi\"\n.data\nvalue: .ascii \"ab\"\n.text\n.globl entry\nentry: leaq message(%rip), %rax\nret\n.section .rodata\n.ascii \"!\"\n",
            &path,
        )
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    let count = u16::from_le_bytes(bytes[60..62].try_into().unwrap()) as usize;
    let strings = u16::from_le_bytes(bytes[62..64].try_into().unwrap()) as usize;
    let headers = u64_at(&bytes, 40) as usize;
    let string_header = headers + strings * 64;
    let names = &bytes[u64_at(&bytes, string_header + 24) as usize..];
    let section = |name: &str| {
        (1..count)
            .find_map(|i| {
                let h = headers + i * 64;
                let n = u32::from_le_bytes(bytes[h..h + 4].try_into().unwrap()) as usize;
                (names[n..].split(|b| *b == 0).next().unwrap() == name.as_bytes()).then_some((i, h))
            })
            .unwrap()
    };
    let contents = |h: usize| {
        let start = u64_at(&bytes, h + 24) as usize;
        &bytes[start..start + u64_at(&bytes, h + 32) as usize]
    };
    let (rodata_index, rodata) = section(".rodata");
    assert_eq!(u64_at(&bytes, rodata + 8), 2);
    assert_eq!(contents(rodata), b"hi\0!");
    assert_eq!(contents(section(".data").1), b"ab");
    assert_eq!(
        contents(section(".text").1),
        &[0x48, 0x8d, 5, 0, 0, 0, 0, 0xc3]
    );
    let symbols = contents(section(".symtab").1);
    assert!(symbols.chunks_exact(24).any(|s| {
        u16::from_le_bytes(s[6..8].try_into().unwrap()) as usize == rodata_index
            && s[4] >> 4 == 1
            && u64_at(s, 8) == 0
    }));
    let reloc = contents(section(".rela.text").1);
    assert_eq!(u64_at(reloc, 0), 3);
    assert_eq!(u64_at(reloc, 8) as u32, 2);
    assert_eq!(u64_at(reloc, 16) as i64, -4);
}
