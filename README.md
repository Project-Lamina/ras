# ras

as/gas alternative Assembler by Lamina Chain

ras assembles assembly source to relocatable object files. It is a binary and a library
(`lamina-ras`, `[lib] name = "ras"`). `lamina` uses the library in-process.

## Usage

```
ras [options] [input.s]
ras [options] -o <output.o> [input.s]
```

Reads stdin when no input is given or the input is `-`. Writes `a.out` when `-o` is
omitted.

```
-o, --output <file>     output object file
--target <arch_os>      target, e.g. x86_64_linux, aarch64_macos
-L, --keep-locals       keep temporary local symbols
-v, --verbose           verbose output
--version
-h, --help
```

Invoked as `as` (a symlink or a copy named `as`, or a name ending in `-as`), ras accepts
the flags a GCC driver passes, such as `-I`, `-march=`, `--gdwarf-*` and `--64`. The
`-g` flag is not accepted by the `ras` binary itself.

## Architectures

One encoder module per architecture in `src/encoder/`:

- `x86_64`, including 32-bit i686 mode (`movl` assembles for `i686_linux`)
- `aarch64`
- `armv7`
- `riscv`, for 32-bit and 64-bit targets (`riscv64gc_linux` is accepted)
- `powerpc`, for little-endian PowerPC64 (`ppc64le_linux`). The big-endian `ppc64_linux`
  target is rejected with `has no binary encoder`.
- `arx64`, only with the `nightly` feature

An instruction the target's encoder does not know is refused with `Encoding error`.
`movq` on `aarch64_linux` is one example.

## Object formats

- ELF for Linux, the BSDs, Redox, Orbis and Prospero
- Mach-O for macOS
- COFF for Windows. The COFF writer covers x86_64 and aarch64 only.

## Debug information

When the input has `.file` and `.loc` rows, ELF and Mach-O objects get DWARF:
`.debug_line`, `.debug_abbrev` and `.debug_info` on ELF, and the matching `__debug_*`
sections on Mach-O (`src/dwarf.rs`).

COFF objects get CodeView `.debug$S` and `.debug$T` (`src/codeview.rs`) only when the
library caller sets `ObjectWriteOptions::debug_info`. The `ras` binary does not set it, so
a COFF object from the command line has no debug sections. No PDB is written. CodeView
records do not describe where values live at run time: no `S_LOCAL` and no
`S_DEFRANGE_*` records are emitted.

## Library

```rust
let mut ras = ras::Ras::new(arch, os)?;
ras.assemble_file(input_path, output_path)?;
```

`Ras::assemble`, `Ras::assemble_to_bytes` and `Ras::assemble_file` take the assembly
text, bytes or file. `ObjectWriteOptions` controls `keep_locals`, the minimal DWARF
sections and the debug request.

## Building

```
cargo build --release
cargo test
```
