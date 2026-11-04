//! A `.loc` row must follow its instruction when branch relaxation moves code.
//! The expected rows were checked against GNU as.
#![cfg(target_os = "linux")]

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;
use std::process::Command;

mod common;
use common::TempDir;

fn source() -> String {
    let mut text = String::from(
        ".file 1 \"relax.c\"\n.text\n.globl relaxed\nrelaxed:\n.loc 1 2 0\nxorl %eax, %eax\n.loc 1 3 0\njne .Lfar\n.loc 1 4 0\nmovl $1, %eax\n",
    );
    for _ in 0..40 {
        text.push_str("addl $1000, %eax\n");
    }
    text.push_str(".loc 1 5 0\nmovl $5, %ecx\n.Lfar:\n.loc 1 6 2\nmovl $7, %edx\n.p2align 4\n.loc 1 7 0\njmp .Lend\n");
    for _ in 0..30 {
        text.push_str("addl $2000, %ebx\n");
    }
    text.push_str(".Lend:\n.loc 1 9 0 is_stmt 0\naddl %edx, %eax\n.loc 1 10 0\nret\n");
    text
}

fn rows(object: &std::path::Path) -> Option<Vec<(u32, u64, bool)>> {
    let output = Command::new("readelf")
        .arg("--debug-dump=decodedline")
        .arg(object)
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 || !fields[0].starts_with("relax") {
            continue;
        }
        let (Ok(number), Some(address)) = (
            fields[1].parse::<u32>(),
            fields[2].strip_prefix("0x").map_or_else(
                || fields[2].parse::<u64>().ok(),
                |hex| u64::from_str_radix(hex, 16).ok(),
            ),
        ) else {
            continue;
        };
        rows.push((number, address, fields.get(3) == Some(&"x")));
    }
    Some(rows)
}

#[test]
fn loc_rows_follow_relaxed_branches() {
    let dir = TempDir::new("debug_line");
    let path = dir.path().join("out.o");
    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(&source(), &path)
        .unwrap();
    let Some(rows) = rows(&path) else {
        return;
    };
    assert_eq!(
        rows,
        [
            (2, 0x0, true),
            (3, 0x2, true),
            (4, 0x8, true),
            (5, 0xd5, true),
            (6, 0xda, true),
            (7, 0xe0, true),
            (9, 0x199, false),
            (10, 0x19b, false),
        ]
    );
}

fn dwarfdump(object: &std::path::Path) -> Option<String> {
    for tool in ["llvm-dwarfdump-18", "llvm-dwarfdump"] {
        if let Ok(output) = Command::new(tool).arg("--debug-line").arg(object).output()
            && output.status.success()
        {
            return String::from_utf8(output.stdout).ok();
        }
    }
    None
}

#[test]
fn loc_flags_reach_the_line_program() {
    let dir = TempDir::new("debug_line_flags");
    let path = dir.path().join("flags.o");
    Ras::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(
            ".file 1 \"flags.c\"\n.text\n.globl f\nf:\n.loc 1 1 0 prologue_end\nnop\n.loc 1 2 0 basic_block discriminator 3\nnop\n.loc 1 3 0 epilogue_begin isa 1\nret\n",
            &path,
        )
        .unwrap();
    let Some(text) = dwarfdump(&path) else {
        return;
    };
    let rows: Vec<Vec<String>> = text
        .lines()
        .filter(|line| line.starts_with("0x"))
        .map(|line| {
            line.split_whitespace()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(rows.len(), 4, "{text}");
    let flags = |row: &Vec<String>| row[7..].join(" ");
    let isa = |row: &Vec<String>| row[4].clone();
    let discriminator = |row: &Vec<String>| row[5].clone();
    assert_eq!(flags(&rows[0]), "is_stmt prologue_end");
    assert_eq!(flags(&rows[1]), "is_stmt basic_block");
    assert_eq!(flags(&rows[2]), "is_stmt epilogue_begin");
    assert_eq!(
        [
            discriminator(&rows[0]),
            discriminator(&rows[1]),
            discriminator(&rows[2])
        ],
        ["0", "3", "0"]
    );
    assert_eq!(
        [isa(&rows[0]), isa(&rows[1]), isa(&rows[2])],
        ["0", "0", "1"]
    );
}
