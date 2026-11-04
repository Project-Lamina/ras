use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;

const IMAGE_REL_AMD64_ADDR32NB: u16 = 0x0003;
const IMAGE_REL_AMD64_SECREL: u16 = 0x000b;
const IMAGE_REL_AMD64_SREL32: u16 = 0x000e;

fn u16_at(bytes: &[u8], at: usize) -> usize {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()) as usize
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// `(offset, type, field)` for each relocation of the named COFF section, where
/// `field` is the 32-bit value stored at the relocated position.
fn section_relocations(bytes: &[u8], name: &str) -> Vec<(u32, u16, u32)> {
    let sections = u16_at(bytes, 2);
    let header = (0..sections)
        .map(|i| 20 + 40 * i)
        .find(|&h| bytes[h..h + 8].split(|b| *b == 0).next() == Some(name.as_bytes()))
        .expect("section present");
    let raw = u32_at(bytes, header + 20) as usize;
    let relocs = u32_at(bytes, header + 24) as usize;
    let count = u16_at(bytes, header + 32);
    (0..count)
        .map(|i| {
            let at = relocs + 10 * i;
            let offset = u32_at(bytes, at);
            let ty = u16_at(bytes, at + 8) as u16;
            let field = u32_at(bytes, raw + offset as usize);
            (offset, ty, field)
        })
        .collect()
}

fn assemble_windows(source: &str) -> Vec<u8> {
    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Windows)
        .unwrap()
        .assemble_to_bytes(source)
        .unwrap()
}

#[test]
fn secrel32_emits_section_relative_type_not_srel32() {
    let obj = assemble_windows(
        ".text\n\
         .globl f\n\
         f:\n\
             leaq _x@SECREL32(%rax), %rax\n\
             movl _x@SECREL32+8(%rsp), %ecx\n\
             ret\n\
         .section .tls$,\"aw\"\n\
         .p2align 3\n\
         _x:\n\
             .zero 16\n",
    );
    let relocs = section_relocations(&obj, ".text");
    assert_eq!(relocs.len(), 2, "{relocs:?}");
    for (_, ty, _) in &relocs {
        assert_eq!(*ty, IMAGE_REL_AMD64_SECREL);
        assert_ne!(*ty, IMAGE_REL_AMD64_SREL32);
    }
    assert_eq!(relocs[0].0, 3);
    assert_eq!(relocs[1].0, 10);
    assert_eq!(relocs[1].2, 8);
}

#[test]
fn rva_and_imgrel_emit_image_relative_addr32nb() {
    let obj = assemble_windows(
        ".text\n\
         .globl f\n\
         f:\n\
             ret\n\
         .Lend:\n\
         .section .xdata,\"dr\"\n\
         .Linfo:\n\
             .long 0\n\
         .section .pdata,\"dr\"\n\
             .rva f, .Lend\n\
             .long .Linfo@IMGREL\n",
    );
    let relocs = section_relocations(&obj, ".pdata");
    assert_eq!(relocs.len(), 3, "{relocs:?}");
    for (index, (offset, ty, _)) in relocs.iter().enumerate() {
        assert_eq!(*ty, IMAGE_REL_AMD64_ADDR32NB, "{relocs:?}");
        assert_eq!(*offset, 4 * index as u32, "{relocs:?}");
    }
}

#[test]
fn rva_is_refused_outside_coff() {
    let error = Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble_to_bytes(".text\n.globl f\nf:\n    ret\n.data\n    .rva f\n")
        .expect_err("ELF has no image base");
    assert!(
        error.to_string().contains("only supported in COFF objects"),
        "{error}"
    );
}
