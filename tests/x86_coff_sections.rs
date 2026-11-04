use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;

const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_MEM_DISCARDABLE: u32 = 0x0200_0000;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

fn u16_at(bytes: &[u8], at: usize) -> usize {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()) as usize
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// Characteristics of the named COFF section. Names longer than eight bytes are
/// held in the string table and the header keeps `/offset`.
fn characteristics(bytes: &[u8], name: &str) -> u32 {
    let sections = u16_at(bytes, 2);
    let strtab = u32_at(bytes, 8) as usize + u32_at(bytes, 12) as usize * 18;
    let header = (0..sections)
        .map(|i| 20 + 40 * i)
        .find(|&h| {
            let raw = bytes[h..h + 8].split(|b| *b == 0).next().unwrap();
            let raw = match raw.strip_prefix(b"/") {
                Some(offset) => {
                    let at = strtab
                        + std::str::from_utf8(offset)
                            .unwrap()
                            .parse::<usize>()
                            .unwrap();
                    bytes[at..].split(|b| *b == 0).next().unwrap()
                }
                None => raw,
            };
            raw == name.as_bytes()
        })
        .expect("section present");
    u32_at(bytes, header + 36)
}

fn assemble_windows(source: &str) -> Vec<u8> {
    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Windows)
        .unwrap()
        .assemble_to_bytes(source)
        .unwrap()
}

#[test]
fn custom_code_section_is_kept_code() {
    let obj = assemble_windows(
        ".text\n\
         .globl main\n\
         main:\n\
             call custom\n\
             ret\n\
         .section .text.custom,\"xr\"\n\
         .globl custom\n\
         custom:\n\
             ret\n",
    );
    let flags = characteristics(&obj, ".text.custom");
    assert_eq!(
        flags & (IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE),
        IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE
    );
    assert_eq!(flags & (IMAGE_SCN_MEM_WRITE | IMAGE_SCN_MEM_DISCARDABLE), 0);
}

#[test]
fn data_section_flags_set_write_and_discard() {
    let obj = assemble_windows(
        ".section .tls$,\"dw\"\n\
             .quad 0\n\
         .section .rdata$x,\"dr\"\n\
             .quad 0\n\
         .section .drop,\"drD\"\n\
             .quad 0\n\
         .section .debug$x,\"dr\"\n\
             .quad 0\n\
         .text\n\
         .globl main\n\
         main:\n\
             ret\n",
    );
    let tls = characteristics(&obj, ".tls$");
    assert_ne!(tls & IMAGE_SCN_CNT_INITIALIZED_DATA, 0);
    assert_ne!(tls & IMAGE_SCN_MEM_WRITE, 0);
    assert_eq!(tls & IMAGE_SCN_MEM_DISCARDABLE, 0);
    let rdata = characteristics(&obj, ".rdata$x");
    assert_eq!(rdata & (IMAGE_SCN_MEM_WRITE | IMAGE_SCN_MEM_DISCARDABLE), 0);
    for discarded in [".drop", ".debug$x"] {
        let flags = characteristics(&obj, discarded);
        assert_ne!(flags & IMAGE_SCN_MEM_DISCARDABLE, 0, "{discarded}");
        assert_eq!(flags & IMAGE_SCN_MEM_WRITE, 0, "{discarded}");
    }
}
