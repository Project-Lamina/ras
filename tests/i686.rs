//! The 32-bit x86 encoder and the ELF32 objects it writes.
#![cfg(target_os = "linux")]

use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use ras::Ras;
use std::process::Command;

mod common;
use common::TempDir;

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

struct Object {
    bytes: Vec<u8>,
}

impl Object {
    fn assemble(label: &str, asm: &str) -> Self {
        let tmp = TempDir::new(label);
        let path = tmp.path().join("out.o");
        Ras::new(TargetArchitecture::I686, TargetOperatingSystem::Linux)
            .unwrap()
            .assemble(asm, &path)
            .unwrap();
        Self {
            bytes: std::fs::read(&path).unwrap(),
        }
    }

    fn header_count(&self) -> usize {
        u16_at(&self.bytes, 48) as usize
    }

    fn header(&self, index: usize) -> usize {
        u32_at(&self.bytes, 32) as usize + index * 40
    }

    fn names(&self) -> &[u8] {
        let strings = u16_at(&self.bytes, 50) as usize;
        &self.bytes[u32_at(&self.bytes, self.header(strings) + 16) as usize..]
    }

    fn find(&self, name: &str) -> Option<usize> {
        (1..self.header_count()).find(|index| {
            let at = u32_at(&self.bytes, self.header(*index)) as usize;
            self.names()[at..].split(|b| *b == 0).next().unwrap() == name.as_bytes()
        })
    }

    fn section(&self, name: &str) -> &[u8] {
        let header = self.header(self.find(name).unwrap_or_else(|| panic!("no {name}")));
        let start = u32_at(&self.bytes, header + 16) as usize;
        &self.bytes[start..start + u32_at(&self.bytes, header + 20) as usize]
    }

    fn field(&self, name: &str, index: usize) -> u32 {
        u32_at(
            &self.bytes,
            self.header(self.find(name).unwrap()) + index * 4,
        )
    }

    /// Every relocation of `name` as (offset, symbol index, type, field value).
    fn relocations(&self, name: &str) -> Vec<(u32, u32, u32, i64)> {
        let target = &name[".rel".len()..];
        let applies_to = self.section(target).to_vec();
        self.section(name)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|entry| {
                let offset = u32_at(entry, 0);
                let info = u32_at(entry, 4);
                let kind = info & 0xff;
                let width = match kind {
                    1 | 2 | 4 => 4,
                    20 => 2,
                    _ => 1,
                };
                let at = offset as usize;
                let mut value = [0u8; 8];
                value[..width].copy_from_slice(&applies_to[at..at + width]);
                let mut stored = i64::from_le_bytes(value);
                let bits = width * 8;
                if stored >> (bits - 1) & 1 == 1 {
                    stored -= 1i64 << bits;
                }
                (offset, info >> 8, kind, stored)
            })
            .collect()
    }

    fn symbol_names(&self) -> Vec<(String, u32, u16)> {
        let strings = self.section(".strtab").to_vec();
        self.section(".symtab")
            .as_chunks::<16>()
            .0
            .iter()
            .map(|entry| {
                let at = u32_at(entry, 0) as usize;
                let name = strings[at..].split(|b| *b == 0).next().unwrap();
                (
                    String::from_utf8_lossy(name).into_owned(),
                    u32_at(entry, 4),
                    u16_at(entry, 14),
                )
            })
            .collect()
    }
}

fn encode(line: &str) -> Vec<u8> {
    Object::assemble("i686-encode", &format!(".text\n.globl m\nm:\n    {line}\n"))
        .section(".text")
        .to_vec()
}

fn error(line: &str) -> String {
    let tmp = TempDir::new("i686-refuse");
    Ras::new(TargetArchitecture::I686, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(
            &format!(".text\n.globl m\nm:\n    {line}\n"),
            &tmp.path().join("out.o"),
        )
        .unwrap_err()
        .to_string()
}

/// Every mnemonic and operand shape lamina's i686 backend emits.
const ENCODINGS: &[(&str, &[u8])] = &[
    ("pushl %ebp", &[0x55]),
    ("pushl %ebx", &[0x53]),
    ("pushl $7", &[0x6a, 0x07]),
    ("popl %ebx", &[0x5b]),
    ("movl %esp, %ebp", &[0x89, 0xe5]),
    ("subl $648, %esp", &[0x81, 0xec, 0x88, 0x02, 0x00, 0x00]),
    ("addl $8, %esp", &[0x83, 0xc4, 0x08]),
    ("movl $30, %eax", &[0xb8, 0x1e, 0x00, 0x00, 0x00]),
    ("movl %eax, -8(%ebp)", &[0x89, 0x45, 0xf8]),
    ("movl -8(%ebp), %ecx", &[0x8b, 0x4d, 0xf8]),
    ("movb %al, -9(%ebp)", &[0x88, 0x45, 0xf7]),
    ("movb -9(%ebp), %al", &[0x8a, 0x45, 0xf7]),
    ("movw %ax, -12(%ebp)", &[0x66, 0x89, 0x45, 0xf4]),
    ("xorl %eax, %eax", &[0x31, 0xc0]),
    ("addl %ecx, %eax", &[0x01, 0xc8]),
    ("andl %ecx, %eax", &[0x21, 0xc8]),
    ("orl %ecx, %eax", &[0x09, 0xc8]),
    ("subl %ecx, %eax", &[0x29, 0xc8]),
    ("cmpl %ecx, %eax", &[0x39, 0xc8]),
    ("testl %eax, %eax", &[0x85, 0xc0]),
    ("andb %cl, %al", &[0x20, 0xc8]),
    ("orb %cl, %al", &[0x08, 0xc8]),
    ("andl $65535, %eax", &[0x25, 0xff, 0xff, 0x00, 0x00]),
    ("cmpl $-1, %ecx", &[0x83, 0xf9, 0xff]),
    ("imull %ecx, %eax", &[0x0f, 0xaf, 0xc1]),
    ("imull $3, %eax, %eax", &[0x6b, 0xc0, 0x03]),
    ("idivl %ecx", &[0xf7, 0xf9]),
    ("divl %ecx", &[0xf7, 0xf1]),
    ("cltd", &[0x99]),
    ("shrl %cl, %eax", &[0xd3, 0xe8]),
    ("shll %cl, %eax", &[0xd3, 0xe0]),
    ("sarl %cl, %eax", &[0xd3, 0xf8]),
    ("rorl %cl, %eax", &[0xd3, 0xc8]),
    ("rolb %cl, %al", &[0xd2, 0xc0]),
    ("rolw $3, %ax", &[0x66, 0xc1, 0xc0, 0x03]),
    ("shrl $16, %eax", &[0xc1, 0xe8, 0x10]),
    ("cmovel %ecx, %eax", &[0x0f, 0x44, 0xc1]),
    ("cmovsl %ecx, %eax", &[0x0f, 0x48, 0xc1]),
    ("cmoval %edx, %ecx", &[0x0f, 0x47, 0xca]),
    ("sete %al", &[0x0f, 0x94, 0xc0]),
    ("setne %al", &[0x0f, 0x95, 0xc0]),
    ("seta %al", &[0x0f, 0x97, 0xc0]),
    ("setae %al", &[0x0f, 0x93, 0xc0]),
    ("setb %al", &[0x0f, 0x92, 0xc0]),
    ("setbe %al", &[0x0f, 0x96, 0xc0]),
    ("setg %al", &[0x0f, 0x9f, 0xc0]),
    ("setl %al", &[0x0f, 0x9c, 0xc0]),
    ("setp %cl", &[0x0f, 0x9a, 0xc1]),
    ("setnp %cl", &[0x0f, 0x9b, 0xc1]),
    ("movsbl %al, %eax", &[0x0f, 0xbe, 0xc0]),
    ("movsbl -9(%ebp), %eax", &[0x0f, 0xbe, 0x45, 0xf7]),
    ("movswl %ax, %eax", &[0x0f, 0xbf, 0xc0]),
    ("movswl -12(%ebp), %eax", &[0x0f, 0xbf, 0x45, 0xf4]),
    ("movzbl %al, %eax", &[0x0f, 0xb6, 0xc0]),
    ("movzbl -9(%ebp), %ecx", &[0x0f, 0xb6, 0x4d, 0xf7]),
    ("movzwl %ax, %eax", &[0x0f, 0xb7, 0xc0]),
    ("bsfl %eax, %edx", &[0x0f, 0xbc, 0xd0]),
    ("bsrl %eax, %edx", &[0x0f, 0xbd, 0xd0]),
    ("bswapl %eax", &[0x0f, 0xc8]),
    ("leal -8(%ebp), %eax", &[0x8d, 0x45, 0xf8]),
    ("leal 16(%esp,%ecx,4), %eax", &[0x8d, 0x44, 0x8c, 0x10]),
    ("call *32(%esp)", &[0xff, 0x54, 0x24, 0x20]),
    ("flds -8(%ebp)", &[0xd9, 0x45, 0xf8]),
    ("fldl -16(%ebp)", &[0xdd, 0x45, 0xf0]),
    ("fildl -8(%ebp)", &[0xdb, 0x45, 0xf8]),
    ("fildll -16(%ebp)", &[0xdf, 0x6d, 0xf0]),
    ("fstps -8(%ebp)", &[0xd9, 0x5d, 0xf8]),
    ("fstpl -16(%ebp)", &[0xdd, 0x5d, 0xf0]),
    ("fadds -8(%ebp)", &[0xd8, 0x45, 0xf8]),
    ("faddl -16(%ebp)", &[0xdc, 0x45, 0xf0]),
    ("fmuls -8(%ebp)", &[0xd8, 0x4d, 0xf8]),
    ("fmull -16(%ebp)", &[0xdc, 0x4d, 0xf0]),
    ("fsubl -16(%ebp)", &[0xdc, 0x65, 0xf0]),
    ("fdivl -16(%ebp)", &[0xdc, 0x75, 0xf0]),
    ("fchs", &[0xd9, 0xe0]),
    ("fsqrt", &[0xd9, 0xfa]),
    ("fstp %st(0)", &[0xdd, 0xd8]),
    ("fucomip %st(1), %st", &[0xdf, 0xe9]),
    ("int $0x80", &[0xcd, 0x80]),
    ("ud2", &[0x0f, 0x0b]),
    ("leave", &[0xc9]),
    ("ret", &[0xc3]),
    ("ret $8", &[0xc2, 0x08, 0x00]),
];

#[test]
fn encodes_the_instruction_set_lamina_emits() {
    for (line, expected) in ENCODINGS {
        assert_eq!(&encode(line), expected, "{line}");
    }
}

/// Nothing in 32-bit mode may carry a REX prefix, and no encoding may reach
/// for a register or an addressing mode that only long mode has.
#[test]
fn refuses_long_mode_forms_by_name() {
    for (line, fragment) in [
        ("addq %rax, %rbx", "addq is not implemented"),
        ("pushq %rax", "pushq is not implemented"),
        ("cltq", "cltq is not implemented"),
        ("syscall", "syscall is not implemented"),
        ("movslq %eax, %ebx", "movslq is not implemented"),
        ("movabsq $1, %rax", "movabsq is not implemented"),
        ("movsd %xmm0, %xmm1", "movsd is not implemented"),
        ("vaddps %ymm0, %ymm1, %ymm2", "vaddps is not implemented"),
        ("leaq sym(%rip), %eax", "leaq is not implemented"),
        ("movl %r8d, %eax", "%r8d does not exist in 32-bit mode"),
        ("addl %rax, %ebx", "%rax does not exist in 32-bit mode"),
        ("movl 4(%rax), %eax", "Invalid memory operand"),
    ] {
        let message = error(line);
        assert!(message.contains(fragment), "{line}: {message}");
    }
}

#[test]
fn branches_resolve_locally_and_relocate_across_symbols() {
    let object = Object::assemble(
        "i686-branch",
        ".text\n.globl m\nm:\n  je 1f\n  jne other\n  jmp 1f\n1:\n  call other\n  ret\n",
    );
    assert_eq!(
        object.section(".text"),
        [
            // je .+2 and jmp .+0 reach a local label as short jumps.
            0x74, 0x08, 0x0f, 0x85, 0xfc, 0xff, 0xff, 0xff, 0xeb, 0x00, 0xe8, 0xfc, 0xff, 0xff,
            0xff, 0xc3
        ]
    );
    let relocations = object.relocations(".rel.text");
    let symbols = object.symbol_names();
    assert_eq!(
        relocations
            .iter()
            .map(|(offset, _, kind, field)| (*offset, *kind, *field))
            .collect::<Vec<_>>(),
        [(4, 2, -4), (11, 2, -4)],
        "both references to `other` are R_386_PC32 with the addend in the field"
    );
    for (_, index, _, _) in &relocations {
        assert_eq!(symbols[*index as usize].0, "other");
    }
}

#[test]
fn symbol_immediates_and_data_pointers_become_absolute_relocations() {
    let object = Object::assemble(
        "i686-abs",
        ".data\ntext: .ascii \"abcde\"\nptr: .long text+2\n.text\n.globl m\nm:\n  movl $text, %eax\n  ret\n",
    );
    assert_eq!(object.section(".text"), [0xb8, 0, 0, 0, 0, 0xc3]);
    let text = object.relocations(".rel.text");
    assert_eq!(text.len(), 1);
    assert_eq!((text[0].0, text[0].2, text[0].3), (1, 1, 0));
    let data = object.relocations(".rel.data");
    assert_eq!(data.len(), 1);
    assert_eq!(
        (data[0].0, data[0].2, data[0].3),
        (5, 1, 2),
        "`.long text+2` keeps its addend in the field an Elf32_Rel patches"
    );
}

#[test]
fn writes_an_elf32_object() {
    let object = Object::assemble(
        "i686-elf",
        ".data\nv: .long 1\n.text\n.globl m\nm:\n  call other\n  ret\n",
    );
    assert_eq!(&object.bytes[..4], b"\x7fELF");
    assert_eq!(object.bytes[4], 1, "ELFCLASS32");
    assert_eq!(u16_at(&object.bytes, 18), 3, "EM_386");
    assert_eq!(u16_at(&object.bytes, 40), 52, "e_ehsize");
    assert_eq!(u16_at(&object.bytes, 46), 40, "e_shentsize");

    assert!(object.find(".rela.text").is_none(), "i386 uses Elf32_Rel");
    assert_eq!(object.field(".rel.text", 1), 9, "SHT_REL");
    assert_eq!(object.field(".rel.text", 9), 8, "sizeof(Elf32_Rel)");
    assert_eq!(object.field(".symtab", 9), 16, "sizeof(Elf32_Sym)");

    // Elf32_Sym orders its fields name, value, size, info, other, shndx; the
    // defined symbol must land in .text with its value at offset 4.
    let text = object.find(".text").unwrap() as u16;
    let symbols = object.symbol_names();
    assert!(symbols.contains(&("m".to_string(), 0, text)), "{symbols:?}");
    assert!(
        symbols
            .iter()
            .any(|(name, value, index)| name == "other" && *value == 0 && *index == 0),
        "{symbols:?}"
    );
}

/// The decisive check: an object ras assembled, linked and run.
#[cfg(target_arch = "x86_64")]
#[test]
fn a_linked_program_runs_and_prints() {
    let tmp = TempDir::new("i686-run");
    let probe = Command::new("cc")
        .args(["-m32", "-xc", "-", "-o"])
        .arg(tmp.path().join("probe"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(b"int main(void){return 0;}")?;
            child.wait()
        });
    if !probe.is_ok_and(|status| status.success()) {
        return;
    }

    let object = tmp.path().join("program.o");
    Ras::new(TargetArchitecture::I686, TargetOperatingSystem::Linux)
        .unwrap()
        .assemble(
            ".data\nmessage: .asciz \"i686 %d\\n\"\n\
             .text\n.globl main\nmain:\n\
             \x20 pushl %ebp\n  movl %esp, %ebp\n  subl $8, %esp\n\
             \x20 movl $20, %eax\n  movl $2, %ecx\n  imull %ecx, %eax\n  addl $2, %eax\n\
             \x20 movl %eax, 4(%esp)\n  movl $message, %eax\n  movl %eax, (%esp)\n\
             \x20 call printf\n  xorl %eax, %eax\n  leave\n  ret\n",
            &object,
        )
        .unwrap();

    let program = tmp.path().join("program");
    let link = Command::new("cc")
        .args(["-m32", "-no-pie"])
        .arg(&object)
        .arg("-o")
        .arg(&program)
        .output()
        .unwrap();
    assert!(
        link.status.success(),
        "{}",
        String::from_utf8_lossy(&link.stderr)
    );
    let run = Command::new(&program).output().unwrap();
    assert!(run.status.success(), "program exited {:?}", run.status);
    assert_eq!(String::from_utf8_lossy(&run.stdout), "i686 42\n");
}
