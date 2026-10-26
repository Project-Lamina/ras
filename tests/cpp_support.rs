//! These compare ras against GNU binutils and clang on ELF objects, so they
//! only run where both the reference assembler and the format are available.
#![cfg(target_os = "linux")]

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;
use std::path::{Path, PathBuf};
use std::process::Command;

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn string(bytes: &[u8], offset: usize) -> &str {
    std::str::from_utf8(bytes[offset..].split(|b| *b == 0).next().unwrap()).unwrap()
}

struct Object(Vec<u8>);

impl Object {
    fn header(&self, index: usize) -> &[u8] {
        let start = u64_at(&self.0, 40) as usize + index * 64;
        &self.0[start..start + 64]
    }

    fn contents(&self, index: usize) -> &[u8] {
        let header = self.header(index);
        let start = u64_at(header, 24) as usize;
        &self.0[start..start + u64_at(header, 32) as usize]
    }

    fn section_name(&self, index: usize) -> &str {
        let names = u16::from_le_bytes(self.0[62..64].try_into().unwrap()) as usize;
        string(self.contents(names), u32_at(self.header(index), 0) as usize)
    }

    fn section(&self, name: &str) -> usize {
        let count = u16::from_le_bytes(self.0[60..62].try_into().unwrap()) as usize;
        (1..count).find(|i| self.section_name(*i) == name).unwrap()
    }

    fn symbols(&self) -> Vec<(&str, &[u8])> {
        let symbols = self.section(".symtab");
        let strings = self.contents(u32_at(self.header(symbols), 40) as usize);
        self.contents(symbols)
            .chunks_exact(24)
            .map(|s| (string(strings, u32_at(s, 0) as usize), s))
            .collect()
    }

    fn symbol(&self, name: &str) -> &[u8] {
        self.symbols()
            .into_iter()
            .find(|(n, _)| *n == name)
            .unwrap()
            .1
    }

    fn relocations(&self, section: &str) -> Vec<(u64, u32, String, i64)> {
        let symbols = self.symbols();
        self.contents(self.section(section))
            .chunks_exact(24)
            .map(|r| {
                let info = u64_at(r, 8);
                (
                    u64_at(r, 0),
                    info as u32,
                    symbols[(info >> 32) as usize].0.into(),
                    u64_at(r, 16) as i64,
                )
            })
            .collect()
    }

    fn group(&self) -> (&str, Vec<&str>) {
        let group = self.section(".group");
        let header = self.header(group);
        assert_eq!(u32_at(header, 4), 17);
        assert_eq!(u32_at(header, 40) as usize, self.section(".symtab"));
        assert_eq!(u64_at(header, 48), 4);
        assert_eq!(u64_at(header, 56), 4);
        let words: Vec<_> = self
            .contents(group)
            .chunks_exact(4)
            .map(|w| u32_at(w, 0))
            .collect();
        assert_eq!(words[0], 1);
        let mut members = Vec::new();
        for index in &words[1..] {
            assert_ne!(u64_at(self.header(*index as usize), 8) & 0x200, 0);
            members.push(self.section_name(*index as usize));
        }
        members.sort();
        (self.symbols()[u32_at(header, 44) as usize].0, members)
    }
}

struct WorkDir(PathBuf);

impl WorkDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("ras_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(command: &mut Command) -> Vec<u8> {
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{command:?}: {:?}\n{}\n{}",
        result.status,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}

fn assemble_pair(dir: &Path, source: &str) -> (Object, Object) {
    let input = dir.join("input.s");
    let reference = dir.join("gas.o");
    let output = dir.join("ras.o");
    std::fs::write(&input, source).unwrap();
    run(Command::new("as")
        .arg("--64")
        .arg(&input)
        .arg("-o")
        .arg(&reference));
    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(source, &output)
        .unwrap();
    (
        Object(std::fs::read(output).unwrap()),
        Object(std::fs::read(reference).unwrap()),
    )
}

#[test]
fn weak_hidden_comdat_and_relative_relocations_match_gas() {
    let dir = WorkDir::new("cpp_metadata");
    let (ras, gas) = assemble_pair(
        &dir.0,
        ".section .gcc_except_table,\"a\"\n.long DW.ref.type-.\n.byte 7\nbase: .long DW.ref.type-base\n.long DW.ref.type-base\n.hidden DW.ref.type\n.weak DW.ref.type\n.section .data.ref,\"awG\",@progbits,DW.ref.type,comdat\n.balign 8\n.type DW.ref.type,@object\n.size DW.ref.type,8\nDW.ref.type: .quad external\n.weak external\n.hidden external\n",
    );
    for name in ["DW.ref.type", "external"] {
        let a = ras.symbol(name);
        let b = gas.symbol(name);
        assert_eq!(&a[4..6], &b[4..6]);
        assert_eq!(u64_at(a, 8), u64_at(b, 8));
        assert_eq!(u64_at(a, 16), u64_at(b, 16));
    }
    assert_eq!(ras.group(), gas.group());
    for name in [".rela.gcc_except_table", ".rela.data.ref"] {
        assert_eq!(ras.relocations(name), gas.relocations(name));
    }
    assert_eq!(
        ras.relocations(".rela.gcc_except_table")
            .iter()
            .map(|r| r.3)
            .collect::<Vec<_>>(),
        [0, 0, 4]
    );
}

#[test]
fn implicit_group_signature_is_defined_locally() {
    let dir = WorkDir::new("group_signature");
    let (ras, gas) = assemble_pair(
        &dir.0,
        ".section .text.inline,\"axG\",@progbits,signature,comdat\n.weak function\nfunction: ret\n",
    );
    assert_eq!(ras.group(), gas.group());
    for object in [&ras, &gas] {
        let signature = object.symbol("signature");
        assert_eq!(signature[4] >> 4, 0);
        let section = u16::from_le_bytes(signature[6..8].try_into().unwrap()) as usize;
        assert_eq!(object.section_name(section), ".group");
    }
}

#[test]
fn symbolic_leb128_and_aliases_match_gas() {
    let dir = WorkDir::new("leb128");
    let (ras, gas) = assemble_pair(
        &dir.0,
        ".text\nstart: .space 130\nend: ret\n.weak alias\n.set alias,start\n.data\n.uleb128 end-start,0,127,128,16384\n.sleb128 -65,-64,63,64,start-end\n.uleb128 tail-head\nhead: .space 128\ntail:\n",
    );
    assert_eq!(
        ras.contents(ras.section(".data")),
        gas.contents(gas.section(".data"))
    );
    assert_eq!(ras.symbol("alias")[4], gas.symbol("alias")[4]);
    assert_eq!(
        u64_at(ras.symbol("alias"), 8),
        u64_at(gas.symbol("alias"), 8)
    );
}

#[test]
fn relaxed_branches_with_symbolic_leb128_match_gas() {
    let dir = WorkDir::new("relaxed_leb128");
    let (ras, gas) = assemble_pair(
        &dir.0,
        ".text\nstart: jmp end\n.space 128,0x90\nend: ret\n.data\n.uleb128 end-start\n.sleb128 start-end\n.balign 16\n.byte 7\n",
    );
    for section in [".text", ".data"] {
        assert_eq!(
            ras.contents(ras.section(section)),
            gas.contents(gas.section(section))
        );
    }
}

#[test]
fn chained_layout_dependencies_match_gas() {
    let dir = WorkDir::new("layout_dependencies");
    let (ras, gas) = assemble_pair(
        &dir.0,
        ".text\nstart: .space 16384\nend: ret\n.data\n.uleb128 tail-head\nhead: .uleb128 end-start\ntail:\n.section .dependent,\"a\"\n.sleb128 tail-head\n",
    );
    for section in [".text", ".data", ".dependent"] {
        assert_eq!(
            ras.contents(ras.section(section)),
            gas.contents(gas.section(section))
        );
    }
    assert_eq!(ras.contents(ras.section(".data")), &[3, 0x80, 0x80, 1]);
}

#[test]
fn cascading_forward_branches_match_gas() {
    let dir = WorkDir::new("branch_cascade");
    let mut source = String::from(".text\n.globl cascade\ncascade:\n");
    let count = 16;
    for index in 0..count {
        source.push_str(&format!("jmp target{index}\n"));
        if index > 0 {
            source.push_str(&format!("target{}:\n", index - 1));
        }
        let padding = if index + 1 == count { 128 } else { 123 };
        source.push_str(&format!(".space {padding},0x90\n"));
    }
    source.push_str(&format!("target{}: ret\n", count - 1));
    let (ras, gas) = assemble_pair(&dir.0, &source);
    let text = ras.contents(ras.section(".text"));
    assert_eq!(text, gas.contents(gas.section(".text")));
    for index in 0..count {
        assert_eq!(&text[index * 128..index * 128 + 5], &[0xe9, 128, 0, 0, 0]);
    }
}

#[test]
fn local_and_common_storage_match_gas() {
    let dir = WorkDir::new("common");
    let (ras, gas) = assemble_pair(
        &dir.0,
        ".local private\n.comm private,17,16\n.comm shared,24,8\n.local second\n.comm second,8,8\n",
    );
    for name in ["private", "shared", "second"] {
        let a = ras.symbol(name);
        let b = gas.symbol(name);
        assert_eq!(a[4], b[4]);
        assert_eq!(u64_at(a, 8), u64_at(b, 8));
        assert_eq!(u64_at(a, 16), u64_at(b, 16));
        if name == "shared" {
            assert_eq!(&a[6..8], &b[6..8]);
        }
    }
    assert_eq!(u64_at(ras.header(ras.section(".bss")), 32), 32);
    assert_eq!(u64_at(ras.header(ras.section(".bss")), 48), 16);
}

#[test]
fn unsupported_directives_and_unresolved_leb128_fail() {
    let dir = WorkDir::new("unsupported");
    for source in [
        ".unknown 1",
        ".cfi_startproc\n.cfi_unknown\n.cfi_endproc",
        ".uleb128 missing",
        ".uleb128 -1",
        ".set alias,missing",
    ] {
        let path = dir.0.join("bad.o");
        assert!(
            Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
                .unwrap()
                .assemble(source, &path)
                .is_err(),
            "{source}"
        );
        assert!(!path.exists());
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn gcc_cpp_exceptions_unwind_with_pie_and_fixed_addresses() {
    let dir = WorkDir::new("cpp_runtime");
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_ras"), dir.0.join("as")).unwrap();
    if let Some(weld) = std::env::var_os("RAS_TEST_WELD") {
        std::os::unix::fs::symlink(std::fs::canonicalize(weld).unwrap(), dir.0.join("ld")).unwrap();
    }
    let input = dir.0.join("input.cpp");
    let duplicate = dir.0.join("duplicate.cpp");
    std::fs::write(&input, include_str!("fixtures/exceptions.cpp")).unwrap();
    std::fs::write(
        &duplicate,
        "#include <vector>\nint other() { std::vector<int> v; v.push_back(3); return v[0]; }\n",
    )
    .unwrap();
    for mode in ["-pie", "-no-pie"] {
        let output = dir.0.join("program");
        run(Command::new("g++")
            .arg(mode)
            .arg(format!("-B{}/", dir.0.display()))
            .arg(&input)
            .arg(&duplicate)
            .arg("-o")
            .arg(&output));
        assert_eq!(
            run(&mut Command::new(&output)),
            b"ctor\nsum=285 caught:boom\ndtor\n"
        );
    }
    std::fs::write(&input, "#include <cstdio>\n#include <stdexcept>\nstruct Guard { ~Guard() { puts(\"cleanup\"); } };\n__attribute__((noinline)) void raise() { Guard g; throw std::runtime_error(\"boom\"); }\nint main() { try { raise(); } catch (const std::exception&) { puts(\"caught\"); } }\n").unwrap();
    for optimization in ["-O0", "-O2"] {
        let output = dir.0.join("unwind");
        run(Command::new("g++")
            .arg(optimization)
            .arg(format!("-B{}/", dir.0.display()))
            .arg(&input)
            .arg("-o")
            .arg(&output));
        assert_eq!(run(&mut Command::new(&output)), b"cleanup\ncaught\n");
    }
}

#[test]
fn weak_calls_and_global_aliases_match_gas() {
    let dir = WorkDir::new("weak_calls");
    let (ras, gas) = assemble_pair(
        &dir.0,
        ".text\n.weak target\n.type target,@function\n.size target,11\ntarget: call target\ncall late\nlate: ret\n.weak late\n.globl alias\n.set alias,target\n",
    );
    assert_eq!(ras.relocations(".rela.text"), gas.relocations(".rela.text"));
    for name in ["target", "late", "alias"] {
        assert_eq!(ras.symbol(name)[4], gas.symbol(name)[4]);
        assert_eq!(u64_at(ras.symbol(name), 8), u64_at(gas.symbol(name), 8));
    }
}

#[test]
fn increment_and_decrement_encodings_match_gas() {
    let dir = WorkDir::new("inc_dec");
    let (ras, gas) = assemble_pair(
        &dir.0,
        ".text\nincb %al\ndecb %r8b\nincl %ecx\ndecl %r12d\nincq %r15\ndecq %rdx\nincb (%rax)\ndecq 8(%rbp)\n",
    );
    assert_eq!(
        ras.contents(ras.section(".text")),
        gas.contents(gas.section(".text"))
    );
}

#[test]
fn temporary_symbols_resolve_without_leaking_internal_labels() {
    let dir = WorkDir::new("temporary_symbols");
    let source = ".text\n.Lras_cfi: nop\n.Lras_size: nop\n.globl entry\n.type entry,@function\nentry:\n.cfi_startproc\n1: nop\n.Ltarget: ret\n.cfi_endproc\n.size entry,.-entry\n.type .Ltyped,@function\n.Ltyped: ret\n.globl .Lglobal\n.Lglobal: ret\n.set .Lalias,.Ltarget+1\n.data\n.quad .Ltarget+3\n.quad 1b\n.quad .Lalias\n";
    let input = dir.0.join("input.s");
    std::fs::write(&input, source).unwrap();
    let (default, gas) = assemble_pair(&dir.0, source);
    let targets = |object: &Object| {
        let symbols = object.symbols();
        object
            .contents(object.section(".rela.data"))
            .chunks_exact(24)
            .map(|r| {
                let info = u64_at(r, 8);
                let symbol = symbols[(info >> 32) as usize].1;
                let section = u16::from_le_bytes(symbol[6..8].try_into().unwrap()) as usize;
                (
                    object.section_name(section).to_string(),
                    info as u32,
                    (u64_at(symbol, 8) as i64).wrapping_add(u64_at(r, 16) as i64),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(targets(&default), targets(&gas));
    assert!(
        default
            .symbols()
            .iter()
            .all(|(name, _)| !name.starts_with(".L") || *name == ".Lglobal")
    );
    let driver = dir.0.join("as");
    #[cfg(unix)]
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_ras"), &driver).unwrap();
    #[cfg(not(unix))]
    std::fs::copy(env!("CARGO_BIN_EXE_ras"), &driver).unwrap();
    for program in [std::path::Path::new(env!("CARGO_BIN_EXE_ras")), &driver] {
        for flag in ["-L", "--keep-locals"] {
            let output = dir.0.join("kept.o");
            run(Command::new(program)
                .arg(flag)
                .arg(&input)
                .arg("-o")
                .arg(&output));
            let kept = Object(std::fs::read(&output).unwrap());
            for name in [".Ltarget", ".Ltyped", ".Lalias", ".Lras_cfi", ".Lras_size"] {
                kept.symbol(name);
            }
            assert!(kept.symbols().iter().all(|(name, _)| {
                !name.starts_with(".Lras_") || [".Lras_cfi", ".Lras_size"].contains(name)
            }));
            assert_eq!(targets(&kept), targets(&gas));
            assert_eq!(
                kept.contents(kept.section(".eh_frame")),
                default.contents(default.section(".eh_frame"))
            );
        }
    }
}

#[test]
fn relocations_retain_merge_and_got_local_symbols() {
    let dir = WorkDir::new("retained_locals");
    let (ras, gas) = assemble_pair(
        &dir.0,
        ".text\nmovq .Lgot@GOTPCREL(%rip),%rax\n.Lgot: ret\nleaq .Lmerged(%rip),%rax\n.section .rodata.str1.1,\"aMS\",@progbits,1\n.Lmerged: .asciz \"kept\"\n.Lunused: .asciz \"discarded\"\n",
    );
    for name in [".Lgot", ".Lmerged"] {
        assert_eq!(u64_at(ras.symbol(name), 8), u64_at(gas.symbol(name), 8));
    }
    assert!(ras.symbols().iter().all(|(name, _)| *name != ".Lunused"));
    assert_eq!(ras.relocations(".rela.text"), gas.relocations(".rela.text"));
}

#[test]
fn compact_cfi_preserves_decoded_unwind_rows() {
    let dir = WorkDir::new("compact_cfi");
    let mut source = String::from(".text\n.cfi_startproc\n");
    for (index, advance) in [1, 63, 64, 255, 256, 65535, 65536].iter().enumerate() {
        source.push_str(&format!(
            ".space {advance}\n.cfi_def_cfa_offset {}\n",
            (index + 2) * 128
        ));
    }
    source.push_str(".cfi_offset 3,-16\n.cfi_offset 64,-24\n.cfi_offset 4,8\n.cfi_val_offset 5,-16\n.cfi_val_offset 6,8\n.cfi_restore 3\n.cfi_restore 64\n.cfi_def_cfa 7,-8\nnop\n.cfi_endproc\n");
    let (ras, gas) = assemble_pair(&dir.0, &source);
    assert!(
        ras.contents(ras.section(".eh_frame")).len()
            <= gas.contents(gas.section(".eh_frame")).len()
    );
    let rows = |path: &Path| {
        let output = run(Command::new("readelf")
            .arg("--debug-dump=frames-interp")
            .arg(path));
        let output = String::from_utf8(output).unwrap();
        let mut in_fde = false;
        output
            .lines()
            .filter_map(|line| {
                if line.contains(" FDE ") {
                    in_fde = true;
                    return Some(line.split_once("pc=").unwrap().1.to_string());
                }
                if line.contains(" CIE ") {
                    in_fde = false;
                }
                (in_fde && !line.trim().is_empty())
                    .then(|| line.split_whitespace().collect::<Vec<_>>().join(" "))
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(rows(&dir.0.join("ras.o")), rows(&dir.0.join("gas.o")));
}

#[test]
fn cies_share_only_compatible_unwind_metadata() {
    let dir = WorkDir::new("shared_cies");
    let mut source = String::from(".text\n");
    for (simple, directives) in [
        ("", ""),
        ("", ""),
        ("simple", ""),
        ("", ".cfi_signal_frame\n"),
        ("", ".cfi_return_column 15\n"),
        ("", ".cfi_personality 0x1b,personality\n"),
        ("", ".cfi_personality 0x1b,other\n"),
        (
            "",
            ".cfi_personality 0x1b,personality\n.cfi_lsda 0x1b,lsda1\n",
        ),
        (
            "",
            ".cfi_personality 0x1b,personality\n.cfi_lsda 0x1b,lsda2\n",
        ),
        ("", ".cfi_personality 0x1b,personality\n.cfi_lsda 0,lsda1\n"),
    ] {
        source.push_str(&format!(
            ".cfi_startproc {simple}\n{directives}.space 4\n.cfi_endproc\n"
        ));
    }
    for architecture in [TargetArchitecture::X86_64, TargetArchitecture::Aarch64] {
        let output = dir.0.join("frames.o");
        Ras::new(architecture, TargetOperatingSystem::Linux)
            .unwrap()
            .assemble(&source, &output)
            .unwrap();
        let object = Object(std::fs::read(&output).unwrap());
        let frames = object.contents(object.section(".eh_frame"));
        let mut cies = std::collections::HashSet::new();
        let mut fdes = 0;
        let mut offset = 0;
        while offset < frames.len() {
            let length = u32_at(frames, offset) as usize;
            assert!(length > 0);
            let pointer = u32_at(frames, offset + 4) as usize;
            if pointer == 0 {
                cies.insert(offset);
            } else {
                assert!(cies.contains(&(offset + 4 - pointer)));
                fdes += 1;
            }
            offset += 4 + length;
        }
        assert_eq!(cies.len(), 8);
        assert_eq!(fdes, 10);
        let relocations = object.relocations(".rela.eh_frame");
        assert_eq!(
            relocations.iter().filter(|r| r.2 == "personality").count(),
            3
        );
        assert_eq!(relocations.iter().filter(|r| r.2 == "lsda1").count(), 2);
        assert_eq!(relocations.iter().filter(|r| r.2 == "lsda2").count(), 1);
    }
}
