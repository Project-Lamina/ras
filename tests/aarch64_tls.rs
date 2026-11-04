#![cfg(any(target_os = "linux", target_os = "macos"))]

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;

mod common;
use common::TempDir;

fn u16_at(bytes: &[u8], offset: usize) -> usize {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> usize {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize
}

fn cstr(bytes: &[u8], offset: usize) -> &str {
    let end = bytes[offset..].iter().position(|b| *b == 0).unwrap();
    std::str::from_utf8(&bytes[offset..offset + end]).unwrap()
}

struct Header {
    name: String,
    offset: usize,
    size: usize,
    link: usize,
    entsize: usize,
}

fn headers(bytes: &[u8]) -> Vec<Header> {
    let table = u64_at(bytes, 40);
    let count = u16_at(bytes, 60);
    let names = u16_at(bytes, 62);
    let names_at = u64_at(bytes, table + names * 64 + 24);
    (0..count)
        .map(|i| {
            let h = table + i * 64;
            Header {
                name: cstr(bytes, names_at + u32_at(bytes, h) as usize).to_string(),
                offset: u64_at(bytes, h + 24),
                size: u64_at(bytes, h + 32),
                link: u32_at(bytes, h + 40) as usize,
                entsize: u64_at(bytes, h + 56),
            }
        })
        .collect()
}

fn text_words(bytes: &[u8]) -> Vec<u32> {
    let text = headers(bytes)
        .into_iter()
        .find(|h| h.name == ".text")
        .unwrap();
    (0..text.size / 4)
        .map(|i| u32_at(bytes, text.offset + i * 4))
        .collect()
}

/// `(offset, kind, symbol, addend)` for every relocation in `.rela.text`.
fn text_relocations(bytes: &[u8]) -> Vec<(u64, u32, String, i64)> {
    let all = headers(bytes);
    let rela = all.iter().find(|h| h.name == ".rela.text").unwrap();
    let symtab = &all[rela.link];
    let strtab = &all[symtab.link];
    (0..rela.size / rela.entsize)
        .map(|i| {
            let at = rela.offset + i * 24;
            let offset = u64_at(bytes, at) as u64;
            let info = u64_at(bytes, at + 8);
            let addend = u64_at(bytes, at + 16) as i64;
            let sym = symtab.offset + (info >> 32) * 24;
            let name = cstr(bytes, strtab.offset + u32_at(bytes, sym) as usize).to_string();
            (offset, (info & 0xffff_ffff) as u32, name, addend)
        })
        .collect()
}

const COUNTER: &str = ".section .tbss,\"awT\",@nobits\n\
                       .globl counter\n\
                       .type counter,@tls_object\n\
                       counter: .zero 8\n";

fn try_assemble(name: &str, os: TargetOperatingSystem, source: &str) -> Result<Vec<u8>, String> {
    let tmp = TempDir::new(&format!("aarch64_tls_{name}"));
    let path = tmp.path().join(format!("aarch64_tls_{name}.o"));
    Ras::new(TargetArchitecture::Aarch64, os)
        .unwrap()
        .assemble(source, &path)
        .map_err(|e| format!("{e:?}"))?;
    Ok(std::fs::read(&path).unwrap())
}

fn assemble(name: &str, source: &str) -> Vec<u8> {
    try_assemble(name, TargetOperatingSystem::Linux, source).unwrap()
}

fn refusal(name: &str, source: &str) -> String {
    try_assemble(name, TargetOperatingSystem::Linux, source).unwrap_err()
}

#[test]
fn local_exec_sequence_emits_tprel_relocations() {
    let bytes = assemble(
        "local_exec",
        &format!(
            ".text\n\
             .globl f\n\
             f:\n\
             mrs x9, tpidr_el0\n\
             add x9, x9, :tprel_hi12:counter, lsl #12\n\
             add x9, x9, :tprel_lo12_nc:counter\n\
             ret\n{COUNTER}"
        ),
    );
    assert_eq!(
        text_words(&bytes),
        vec![0xd53b_d049, 0x9140_0129, 0x9100_0129, 0xd65f_03c0]
    );
    assert_eq!(
        text_relocations(&bytes),
        vec![
            (4, 549, "counter".to_string(), 0),
            (8, 551, "counter".to_string(), 0),
        ]
    );
}

#[test]
fn initial_exec_sequence_emits_gottprel_relocations() {
    let bytes = assemble(
        "initial_exec",
        &format!(
            ".text\n\
             .globl f\n\
             f:\n\
             adrp x11, :gottprel:counter\n\
             ldr x11, [x11, :gottprel_lo12:counter]\n\
             mrs x12, tpidr_el0\n\
             add x11, x11, x12\n\
             ret\n{COUNTER}"
        ),
    );
    assert_eq!(
        text_words(&bytes)[..2],
        [0x9000_000b, 0xf940_016b],
        "the page and page-offset fields must be left at zero for the linker"
    );
    assert_eq!(
        text_relocations(&bytes),
        vec![
            (0, 541, "counter".to_string(), 0),
            (4, 542, "counter".to_string(), 0),
        ]
    );
}

#[test]
fn tprel_lo12_without_nc_is_a_distinct_relocation() {
    let bytes = assemble(
        "tprel_lo12",
        &format!(
            ".text\n\
             f:\n\
             add x9, x9, :tprel_lo12:counter\n{COUNTER}"
        ),
    );
    assert_eq!(text_words(&bytes), vec![0x9100_0129]);
    assert_eq!(
        text_relocations(&bytes),
        vec![(0, 550, "counter".to_string(), 0)]
    );
}

#[test]
fn tprel_add_carries_an_addend() {
    let bytes = assemble(
        "tprel_addend",
        &format!(
            ".text\n\
             f:\n\
             add x9, x9, :tprel_lo12_nc:counter+8\n{COUNTER}"
        ),
    );
    assert_eq!(text_words(&bytes), vec![0x9100_0129]);
    assert_eq!(
        text_relocations(&bytes),
        vec![(0, 551, "counter".to_string(), 8)]
    );
}

#[test]
fn tprel_add_refuses_32_bit_registers() {
    let message = refusal(
        "tprel_w",
        &format!(
            ".text\n\
             f:\n\
             add w9, w9, :tprel_lo12_nc:counter\n{COUNTER}"
        ),
    );
    assert!(
        message.contains(":tprel_lo12_nc:counter takes 64-bit registers"),
        "{message}"
    );
}

#[test]
fn tprel_hi12_refuses_a_shift_other_than_12() {
    let message = refusal(
        "tprel_shift",
        &format!(
            ".text\n\
             f:\n\
             add x9, x9, :tprel_hi12:counter, lsl #0\n{COUNTER}"
        ),
    );
    assert!(
        message.contains("Unsupported shift for :tprel_hi12:counter"),
        "{message}"
    );
}

#[test]
fn unknown_tprel_modifier_is_refused() {
    let message = refusal(
        "tprel_unknown",
        &format!(
            ".text\n\
             f:\n\
             add x9, x9, :tprel_g0:counter\n{COUNTER}"
        ),
    );
    assert!(
        message.contains("Unsupported relocation modifier: :tprel_g0:counter"),
        "{message}"
    );
}

#[test]
fn unknown_gottprel_modifier_is_refused() {
    let message = refusal(
        "gottprel_unknown",
        &format!(
            ".text\n\
             f:\n\
             ldr x9, [x9, :gottprel_g1:counter]\n{COUNTER}"
        ),
    );
    assert!(
        message.contains("Unsupported relocation modifier"),
        "{message}"
    );
}

#[test]
fn gottprel_load_refuses_32_bit_registers() {
    let message = refusal(
        "gottprel_w",
        &format!(
            ".text\n\
             f:\n\
             ldr w9, [x9, :gottprel_lo12:counter]\n{COUNTER}"
        ),
    );
    assert!(
        message.contains("Relocated ldr takes 64-bit registers"),
        "{message}"
    );
}

#[test]
fn elf_tls_relocations_are_refused_for_macho() {
    let message = try_assemble(
        "macho",
        TargetOperatingSystem::MacOS,
        ".text\n\
         _f:\n\
         add x9, x9, :tprel_hi12:_counter, lsl #12\n",
    )
    .unwrap_err();
    assert!(
        message.contains("R_AARCH64_TLSLE_ADD_TPREL_HI12")
            && message.contains("has no Mach-O equivalent"),
        "{message}"
    );
}

#[test]
fn elf_tls_relocations_are_refused_for_coff() {
    let message = try_assemble(
        "coff",
        TargetOperatingSystem::Windows,
        ".text\n\
         f:\n\
         adrp x9, :gottprel:counter\n",
    )
    .unwrap_err();
    assert!(
        message.contains("R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21")
            && message.contains("no ARM64 COFF relocation"),
        "{message}"
    );
}
