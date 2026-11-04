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
    kind: u32,
    flags: u64,
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
                kind: u32_at(bytes, h + 4),
                flags: u64_at(bytes, h + 8) as u64,
                offset: u64_at(bytes, h + 24),
                size: u64_at(bytes, h + 32),
                link: u32_at(bytes, h + 40) as usize,
                entsize: u64_at(bytes, h + 56),
            }
        })
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

fn assemble(name: &str, source: &str) -> Vec<u8> {
    let tmp = TempDir::new(&format!("x86_tls_{name}"));
    let path = tmp.path().join(format!("x86_tls_{name}.o"));
    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(source, &path)
        .unwrap();
    std::fs::read(&path).unwrap()
}

#[test]
fn initial_and_local_exec_sequences_emit_tls_relocations() {
    let bytes = assemble(
        "initial_local_exec",
        "movq tvar@gottpoff(%rip), %rax\n\
         movq %fs:(%rax), %rdx\n\
         movq %fs:0, %rax\n\
         leaq tvar@tpoff(%rax), %rdx\n\
         movq %fs:tzero@tpoff, %rcx\n",
    );
    let text = headers(&bytes)
        .into_iter()
        .find(|h| h.name == ".text")
        .unwrap();
    assert_eq!(
        &bytes[text.offset..text.offset + text.size],
        &[
            0x48, 0x8b, 0x05, 0, 0, 0, 0, // movq tvar@gottpoff(%rip), %rax
            0x64, 0x48, 0x8b, 0x10, // movq %fs:(%rax), %rdx
            0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0, // movq %fs:0, %rax
            0x48, 0x8d, 0x90, 0, 0, 0, 0, // leaq tvar@tpoff(%rax), %rdx
            0x64, 0x48, 0x8b, 0x0c, 0x25, 0, 0, 0, 0, // movq %fs:tzero@tpoff, %rcx
        ][..]
    );
    assert_eq!(
        text_relocations(&bytes),
        vec![
            (3, 22, "tvar".to_string(), -4),
            (23, 23, "tvar".to_string(), 0),
            (32, 23, "tzero".to_string(), 0),
        ]
    );
}

#[test]
fn unsupported_tls_model_is_an_error() {
    let tmp = TempDir::new("x86_tls_bad");
    let path = tmp.path().join("x86_tls_bad.o");
    let result = Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble("leaq tvar@tlsgd(%rip), %rdi\n", &path);
    let message = format!("{:?}", result.unwrap_err());
    assert!(
        message.contains("unsupported TLS model @tlsgd"),
        "{message}"
    );
}

#[test]
fn tls_section_and_symbol_flags() {
    let bytes = assemble(
        "section_flags",
        ".section .tdata,\"awT\",@progbits\n\
         .globl tvar\n\
         .type tvar,@tls_object\n\
         tvar: .quad 0x1122334455667788\n\
         .section .tbss,\"awT\",@nobits\n\
         .globl tzero\n\
         .type tzero,@tls_object\n\
         tzero: .zero 16\n\
         .text\n\
         .globl _start\n\
         _start: ret\n\
         .section .data\n\
         ptr: .quad tvar@tpoff\n",
    );
    let all = headers(&bytes);
    let tdata = all.iter().find(|h| h.name == ".tdata").unwrap();
    assert_eq!(tdata.flags & 0x400, 0x400);
    assert_eq!(tdata.kind, 1);
    assert_eq!(tdata.size, 8);
    let tbss = all.iter().find(|h| h.name == ".tbss").unwrap();
    assert_eq!(tbss.flags & 0x400, 0x400);
    assert_eq!(tbss.kind, 8);
    assert_eq!(tbss.size, 16);
    let data_rela = all.iter().find(|h| h.name == ".rela.data").unwrap();
    assert_eq!(u64_at(&bytes, data_rela.offset + 8) as u32, 18);
}
