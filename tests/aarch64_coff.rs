//! The ARM64 COFF writer: which relocations it maps, and what it refuses.

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;

const IMAGE_FILE_MACHINE_ARM64: u16 = 0xAA64;
const SECTION_HEADER_SIZE: usize = 40;
const SYMBOL_SIZE: usize = 18;
const RELOC_SIZE: usize = 10;

fn assemble(asm: &str) -> Result<Vec<u8>, String> {
    Ras::new(TargetArchitecture::Aarch64, TargetOperatingSystem::Windows)
        .map_err(|e| e.to_string())?
        .assemble_to_bytes(asm)
        .map_err(|e| e.to_string())
}

fn object(asm: &str) -> Vec<u8> {
    assemble(asm).expect("aarch64 COFF object")
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn symbol_names(bytes: &[u8]) -> Vec<String> {
    let sym_ptr = u32_at(bytes, 8) as usize;
    let nsyms = u32_at(bytes, 12) as usize;
    let strtab = sym_ptr + nsyms * SYMBOL_SIZE;
    let mut names = Vec::new();
    let mut i = 0;
    while i < nsyms {
        let e = sym_ptr + i * SYMBOL_SIZE;
        let raw = &bytes[e..e + 8];
        let name = if raw[..4] == [0, 0, 0, 0] {
            let off = strtab + u32_at(raw, 4) as usize;
            let end = bytes[off..].iter().position(|&b| b == 0).unwrap();
            String::from_utf8(bytes[off..off + end].to_vec()).unwrap()
        } else {
            String::from_utf8(raw.iter().copied().take_while(|&b| b != 0).collect()).unwrap()
        };
        names.push(name);
        i += 1 + bytes[e + 17] as usize;
    }
    names
}

/// `(section name, offset, relocation type, target symbol name)` for every relocation.
fn relocations(bytes: &[u8]) -> Vec<(String, u32, u16, String)> {
    let names = symbol_names(bytes);
    let nsections = u16_at(bytes, 2) as usize;
    let sym_ptr = u32_at(bytes, 8) as usize;
    let nsyms = u32_at(bytes, 12) as usize;
    let strtab = sym_ptr + nsyms * SYMBOL_SIZE;
    // Symbol indices count auxiliary records, so build an index-to-name map.
    let mut by_index = vec![String::new(); nsyms];
    let mut i = 0;
    let mut n = 0;
    while i < nsyms {
        by_index[i] = names[n].clone();
        i += 1 + bytes[sym_ptr + i * SYMBOL_SIZE + 17] as usize;
        n += 1;
    }
    let mut out = Vec::new();
    for s in 0..nsections {
        let h = 20 + s * SECTION_HEADER_SIZE;
        let raw = &bytes[h..h + 8];
        let section = if raw[0] == b'/' {
            let off: usize = std::str::from_utf8(&raw[1..])
                .unwrap()
                .trim_end_matches('\0')
                .parse()
                .unwrap();
            let at = strtab + off;
            let end = bytes[at..].iter().position(|&b| b == 0).unwrap();
            String::from_utf8(bytes[at..at + end].to_vec()).unwrap()
        } else {
            String::from_utf8(raw.iter().copied().take_while(|&b| b != 0).collect()).unwrap()
        };
        let reloc_ptr = u32_at(bytes, h + 24) as usize;
        for r in 0..u16_at(bytes, h + 32) as usize {
            let e = reloc_ptr + r * RELOC_SIZE;
            out.push((
                section.clone(),
                u32_at(bytes, e),
                u16_at(bytes, e + 8),
                by_index[u32_at(bytes, e + 4) as usize].clone(),
            ));
        }
    }
    out
}

#[test]
fn object_header_names_the_arm64_machine() {
    let obj = object(".text\n.globl main\nmain:\n    ret\n");
    assert_eq!(u16_at(&obj, 0), IMAGE_FILE_MACHINE_ARM64);
}

#[test]
fn page_and_page_offset_pairs_become_pagebase_rel21_and_pageoffset_12a() {
    let obj = object(
        ".data\nmsg:\n    .byte 0x68, 0x00\n\
         .text\n.globl main\nmain:\n    adrp x0, msg\n    add x0, x0, :lo12:msg\n    ret\n",
    );
    assert_eq!(
        relocations(&obj),
        vec![
            (".text".into(), 0, 0x0004, "msg".into()),
            (".text".into(), 4, 0x0006, "msg".into()),
        ]
    );
}

#[test]
fn a_relocated_load_becomes_pageoffset_12l() {
    let obj = object(
        ".data\n.p2align 3\nslot:\n    .quad 0\n\
         .text\n.globl main\nmain:\n    adrp x0, slot\n    ldr x1, [x0, #:lo12:slot]\n    ret\n",
    );
    let kinds: Vec<u16> = relocations(&obj).iter().map(|r| r.2).collect();
    assert_eq!(kinds, vec![0x0004, 0x0007]);
}

#[test]
fn a_call_to_an_external_function_becomes_branch26() {
    let obj = object(".text\n.globl main\nmain:\n    bl other\n    ret\n");
    assert_eq!(
        relocations(&obj),
        vec![(".text".into(), 0, 0x0003, "other".into())]
    );
}

#[test]
fn branches_to_a_global_keep_their_field_width() {
    let obj = object(
        ".text\n.globl main\n.globl far\nmain:\n\
             b.ne far\n    tbz x0, #3, far\n    b far\n    ret\nfar:\n    ret\n",
    );
    let kinds: Vec<u16> = relocations(&obj).iter().map(|r| r.2).collect();
    assert_eq!(kinds, vec![0x000F, 0x0010, 0x0003]);
}

#[test]
fn an_absolute_pointer_becomes_addr64_with_the_addend_in_the_field() {
    let obj = object(".data\nslot:\n    .quad target + 24\n.text\n.globl main\nmain:\n    ret\n");
    let relocs = relocations(&obj);
    assert_eq!(relocs.len(), 1);
    assert_eq!((relocs[0].2, relocs[0].3.as_str()), (0x000E, "target"));

    let data_header = (0..u16_at(&obj, 2) as usize)
        .map(|s| 20 + s * SECTION_HEADER_SIZE)
        .find(|&h| obj[h..h + 5] == *b".data")
        .expect(".data header");
    let at = u32_at(&obj, data_header + 20) as usize + relocs[0].1 as usize;
    assert_eq!(u64::from_le_bytes(obj[at..at + 8].try_into().unwrap()), 24);
}

#[test]
fn an_image_relative_address_becomes_addr32nb() {
    let obj = object(".text\n.globl main\nmain:\n    ret\n.section .pdata,\"dr\"\n    .rva main\n");
    let relocs = relocations(&obj);
    assert_eq!(relocs.len(), 1);
    assert_eq!(
        (relocs[0].0.as_str(), relocs[0].2, relocs[0].3.as_str()),
        (".pdata", 0x0002, "main")
    );
}

#[test]
fn instruction_relocations_leave_the_instruction_word_untouched() {
    let obj = object(".data\nmsg:\n    .byte 0\n.text\n.globl main\nmain:\n    adrp x0, msg\n");
    let text_header = (0..u16_at(&obj, 2) as usize)
        .map(|s| 20 + s * SECTION_HEADER_SIZE)
        .find(|&h| obj[h..h + 5] == *b".text")
        .expect(".text header");
    let at = u32_at(&obj, text_header + 20) as usize;
    assert_eq!(u32_at(&obj, at), 0x9000_0000);
}

#[test]
fn elf_mapping_symbols_stay_out_of_coff_objects() {
    let obj = object(".text\n.globl main\nmain:\n    ret\n.data\nd:\n    .byte 1\n");
    let names = symbol_names(&obj);
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("$x") || n.starts_with("$d")),
        "{names:?}"
    );
}

#[test]
fn a_got_page_relocation_is_refused_by_name() {
    let error = assemble(".text\n.globl main\nmain:\n    adrp x0, :got:sym\n    ret\n")
        .expect_err("refused");
    assert!(error.contains("R_AARCH64_ADR_GOT_PAGE"), "{error}");
    assert!(error.contains("sym"), "{error}");
}

#[test]
fn a_got_low12_relocation_is_refused_by_name() {
    let error = assemble(".text\n.globl main\nmain:\n    ldr x0, [x0, #:got_lo12:sym]\n    ret\n")
        .expect_err("refused");
    assert!(error.contains("R_AARCH64_LD64_GOT_LO12_NC"), "{error}");
}

#[test]
fn an_addend_on_an_instruction_relocation_is_refused() {
    let error = assemble(
        ".data\nmsg:\n    .byte 0\n.text\n.globl main\nmain:\n    adrp x0, msg+8\n    ret\n",
    )
    .expect_err("refused");
    assert!(error.contains("R_AARCH64_ADR_PREL_PG_HI21"), "{error}");
    assert!(error.contains("addend 8"), "{error}");
}

#[test]
fn a_pc_relative_data_difference_is_refused_by_name() {
    let error =
        assemble(".text\n.globl main\nmain:\n    ret\n.data\nhere:\n    .word there - here\n")
            .expect_err("refused");
    assert!(error.contains("R_AARCH64_PREL32"), "{error}");
}
