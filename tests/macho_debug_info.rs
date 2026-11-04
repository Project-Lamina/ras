//! The same DWARF, carried in a Mach-O object.
//!
//! Mach-O spells a debug section `__DWARF,__debug_*` and marks it `S_ATTR_DEBUG`, and
//! its linker drops those sections rather than relocating them, so the only relocation
//! left is the one that names code. Every assertion below is a structural one, and
//! nothing in this file claims a debugger ran.
#![cfg(any(target_os = "linux", target_os = "macos"))]

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::{
    BaseType, DebugInfoRequest, DebugType, DeclSite, FormalParameter, ObjectWriteOptions, Ras,
    Subprogram,
};
use std::path::Path;
use std::process::Command;

mod common;
use common::TempDir;

const RELOC_UNSIGNED: u32 = 0;

const SOURCE: &str = ".file 1 \"shapes.lamina\"\n\
.text\n\
.globl _twice\n\
_twice:\n\
.loc 1 1 1\n\
addl %edi, %edi\n\
movl %edi, %eax\n\
ret\n\
_half:\n\
.loc 1 6 1\n\
movl %edi, %eax\n\
ret\n";

/// The DWARF name is the source name and the object label carries Mach-O's underscore,
/// so the two differ for every function here.
fn request() -> DebugInfoRequest {
    let describe = |name: &str, line: u32| Subprogram {
        name: name.into(),
        symbol: Some(format!("_{name}")),
        external: name == "twice",
        decl: Some(DeclSite { file: 1, line }),
        ret: Some(DebugType::Base(BaseType::I32)),
        params: vec![FormalParameter {
            name: Some("n".into()),
            ty: Some(DebugType::Base(BaseType::I32)),
        }],
        variadic: false,
    };
    DebugInfoRequest {
        producer: Some("ras test".into()),
        subprograms: vec![describe("twice", 1), describe("half", 6)],
    }
}

fn assemble(arch: TargetArchitecture, path: &Path) {
    let options = ObjectWriteOptions {
        function_symbols: vec!["_twice".into(), "_half".into()],
        debug_info: Some(request()),
        ..ObjectWriteOptions::default()
    };
    let source = match arch {
        TargetArchitecture::Aarch64 => SOURCE
            .replace("addl %edi, %edi", "add w0, w0, w0")
            .replace("movl %edi, %eax", "mov w0, w0"),
        _ => SOURCE.to_string(),
    };
    Ras::with_object_write_options(arch, TargetOperatingSystem::MacOS, options)
        .expect("assembler")
        .assemble(&source, path)
        .expect("assemble");
}

fn which(tool: &str) -> Option<String> {
    for name in [tool.to_string(), format!("{tool}-18"), format!("{tool}-19")] {
        let found = Command::new(&name)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success());
        if found {
            return Some(name);
        }
    }
    // Xcode keeps its LLVM tools out of PATH.
    let output = Command::new("xcrun")
        .args(["--find", tool])
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[test]
fn the_debug_sections_are_in_the_dwarf_segment_and_marked_debug() {
    if which("llvm-readobj").is_none() {
        return;
    }
    let dir = TempDir::new("macho_debug_sections");
    let path = dir.path().join("out.o");
    assemble(TargetArchitecture::X86_64, &path);
    let output = Command::new(which("llvm-readobj").expect("llvm-readobj"))
        .arg("--sections")
        .arg(&path)
        .output()
        .expect("llvm-readobj");
    let dump = String::from_utf8_lossy(&output.stdout).into_owned();

    for name in ["__debug_line", "__debug_abbrev", "__debug_info"] {
        let section = dump
            .split("    Index:")
            .find(|part| part.contains(&format!("Name: {name} ")))
            .unwrap_or_else(|| panic!("no section {name} in {dump}"));
        assert!(section.contains("Segment: __DWARF "), "{section}");
        assert!(section.contains("Debug ("), "{section}");
    }
}

/// `ld` discards the `__DWARF` sections instead of rebasing them, so an offset from one
/// into another is already final. Only the addresses that name code may be relocated.
#[test]
fn the_only_relocations_in_the_debug_sections_name_the_text_section() {
    let dir = TempDir::new("macho_debug_relocs");
    let path = dir.path().join("out.o");
    assemble(TargetArchitecture::X86_64, &path);
    let object = std::fs::read(&path).expect("read object");
    let macho = Macho::parse(&object);

    let text = macho
        .sections
        .iter()
        .position(|s| s.name == "__text")
        .expect("a text section");
    let mut relocated = 0;
    for section in macho.sections.iter().filter(|s| s.segment == "__DWARF") {
        for &(_, info) in &section.relocs {
            assert_eq!(info >> 28, RELOC_UNSIGNED, "{} type", section.name);
            assert_eq!((info >> 25) & 3, 3, "{} width", section.name);
            assert_eq!((info >> 24) & 1, 0, "{} pcrel", section.name);
            assert_eq!((info >> 27) & 1, 0, "{} extern", section.name);
            assert_eq!(info & 0x00ff_ffff, text as u32 + 1, "{}", section.name);
            relocated += 1;
        }
    }
    assert_eq!(
        relocated, 4,
        "the unit's and both subprograms' low_pc, and the line table's first address"
    );
    assert!(
        macho
            .sections
            .iter()
            .any(|s| s.name == "__debug_abbrev" && s.relocs.is_empty())
    );
}

#[test]
fn a_subprogram_is_named_without_the_object_underscore() {
    let Some(dwarfdump) = which("llvm-dwarfdump") else {
        return;
    };
    let dir = TempDir::new("macho_debug_names");
    let path = dir.path().join("out.o");
    assemble(TargetArchitecture::X86_64, &path);
    let output = Command::new(dwarfdump)
        .arg("--debug-info")
        .arg(&path)
        .output()
        .expect("llvm-dwarfdump");
    let dump = String::from_utf8_lossy(&output.stdout).into_owned();

    assert!(dump.contains("DW_AT_name\t(\"twice\")"), "{dump}");
    assert!(dump.contains("DW_AT_name\t(\"half\")"), "{dump}");
    assert!(!dump.contains("\"_twice\""), "{dump}");
    assert_eq!(dump.matches("DW_TAG_subprogram").count(), 2, "{dump}");
}

/// The property the ELF path holds, held here too: nothing knows where the allocator
/// put a parameter, so the answer must be silence rather than a guess.
#[test]
fn a_parameter_without_a_known_location_is_given_none() {
    let Some(dwarfdump) = which("llvm-dwarfdump") else {
        return;
    };
    let dir = TempDir::new("macho_debug_no_location");
    let path = dir.path().join("out.o");
    assemble(TargetArchitecture::X86_64, &path);
    let output = Command::new(dwarfdump)
        .arg("--debug-info")
        .arg(&path)
        .output()
        .expect("llvm-dwarfdump");
    let dump = String::from_utf8_lossy(&output.stdout).into_owned();

    assert!(!dump.contains("DW_AT_location"), "{dump}");
}

#[test]
fn both_mach_o_architectures_pass_the_dwarf_verifier() {
    let Some(dwarfdump) = which("llvm-dwarfdump") else {
        return;
    };
    for arch in [TargetArchitecture::X86_64, TargetArchitecture::Aarch64] {
        let dir = TempDir::new("macho_debug_verify");
        let path = dir.path().join("out.o");
        assemble(arch, &path);
        let output = Command::new(&dwarfdump)
            .arg("--verify")
            .arg(&path)
            .output()
            .expect("llvm-dwarfdump");
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{arch:?}: {text}");
        assert!(text.contains("No errors."), "{arch:?}: {text}");
    }
}

/// `dsymutil` reads the objects themselves, so feeding it a debug map that names this
/// object is the whole of the dSYM path short of a linker that writes one. It relocates
/// `low_pc` and the line table to the addresses the map gives and verifies its output.
#[test]
fn dsymutil_links_the_debug_info_to_the_addresses_a_debug_map_gives() {
    let Some(dsymutil) = which("dsymutil") else {
        return;
    };
    let dir = TempDir::new("macho_debug_dsym");
    let path = dir.path().join("out.o");
    assemble(TargetArchitecture::X86_64, &path);
    let base = 0x1_0000_1000u64;
    let map = dir.path().join("map.yaml");
    let entries: String = [("_twice", 0u64, 5u64), ("_half", 5, 3)]
        .iter()
        .map(|(name, at, size)| {
            format!(
                "      - {{ sym: {name}, objAddr: 0x{at:X}, binAddr: 0x{:X}, size: 0x{size:X} }}\n",
                base + at
            )
        })
        .collect();
    std::fs::write(
        &map,
        format!(
            "---\ntriple:          'x86_64-apple-darwin'\nbinary-path:     '{0}'\nobjects:\n  - filename:        '{0}'\n    timestamp:       0\n    symbols:\n{entries}...\n",
            path.display()
        ),
    )
    .expect("write debug map");

    let out = dir.path().join("out.dwarf");
    let run = Command::new(dsymutil)
        .args(["-y", "-f", "--verify-dwarf=all", "-o"])
        .arg(&out)
        .arg(&map)
        .output()
        .expect("dsymutil");
    assert!(
        run.status.success(),
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let dwarfdump = which("llvm-dwarfdump").expect("llvm-dwarfdump beside dsymutil");
    let dump = Command::new(dwarfdump)
        .arg("--debug-info")
        .arg(&out)
        .output()
        .expect("llvm-dwarfdump");
    let dump = String::from_utf8_lossy(&dump.stdout).into_owned();
    assert!(
        dump.contains(&format!("DW_AT_low_pc\t(0x{base:016x})")),
        "{dump}"
    );
    assert!(
        dump.contains(&format!("DW_AT_low_pc\t(0x{:016x})", base + 5)),
        "half moved with the debug map: {dump}"
    );
}

struct MachoSection {
    segment: String,
    name: String,
    relocs: Vec<(u32, u32)>,
}

struct Macho {
    sections: Vec<MachoSection>,
}

impl Macho {
    fn parse(object: &[u8]) -> Self {
        let u32_at = |at: usize| u32::from_le_bytes(object[at..at + 4].try_into().expect("four"));
        let name16 = |at: usize| {
            let field = &object[at..at + 16];
            let end = field.iter().position(|&b| b == 0).unwrap_or(16);
            String::from_utf8_lossy(&field[..end]).into_owned()
        };
        assert_eq!(u32_at(0), 0xfeed_facf, "Mach-O 64 magic");
        let mut sections = Vec::new();
        let mut command = 32;
        for _ in 0..u32_at(16) {
            if u32_at(command) == 0x19 {
                let count = u32_at(command + 64) as usize;
                for index in 0..count {
                    let header = command + 72 + 80 * index;
                    let relocation_offset = u32_at(header + 56) as usize;
                    let relocation_count = u32_at(header + 60) as usize;
                    sections.push(MachoSection {
                        name: name16(header),
                        segment: name16(header + 16),
                        relocs: (0..relocation_count)
                            .map(|entry| {
                                let at = relocation_offset + 8 * entry;
                                (u32_at(at), u32_at(at + 4))
                            })
                            .collect(),
                    });
                }
            }
            command += u32_at(command + 4) as usize;
        }
        Self { sections }
    }
}
