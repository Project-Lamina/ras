//! What a caller wants a debugger told about a module, in terms no debug format owns.
//!
//! `dwarf` encodes this for ELF and Mach-O and `codeview` for COFF. Both read the same
//! request: the facts a compiler has about a function — its name, where it was
//! declared, its parameters and their types, and where its code ended up — are the
//! same facts whichever format carries them.

use crate::parser::{DebugFile, Section};
use crate::{object::ExternalReloc, target::ObjectFormat};
use std::collections::{BTreeMap, HashMap};

/// What the caller wants described, named in target-neutral terms.
///
/// A variable's location is deliberately absent: a backend that cannot say where a
/// value lives after register allocation must leave it unsaid, and a debugger reads
/// the silence as "optimized out". When backends can say, the location belongs on
/// [`FormalParameter`] and on a `locals` list beside it.
#[derive(Debug, Clone, Default)]
pub struct DebugInfoRequest {
    /// Name and version of the producing compiler.
    pub producer: Option<String>,
    pub subprograms: Vec<Subprogram>,
}

/// One function. `name` is what a debugger shows; `symbol` is the label the object
/// gives its code, for the platforms that spell the two differently — Mach-O's leading
/// underscore. `None` means they agree. A function this object does not define is
/// skipped rather than described without an address.
#[derive(Debug, Clone)]
pub struct Subprogram {
    pub name: String,
    pub symbol: Option<String>,
    pub external: bool,
    pub decl: Option<DeclSite>,
    /// `None` for a function that returns nothing, or whose return type has no
    /// description the writer can give honestly.
    pub ret: Option<DebugType>,
    pub params: Vec<FormalParameter>,
    /// A trailing `...`.
    pub variadic: bool,
}

impl Subprogram {
    /// The label the object gives this function's code.
    pub(crate) fn object_symbol(&self) -> &str {
        self.symbol.as_deref().unwrap_or(self.name.as_str())
    }
}

/// `file` is the number the source gave a `.file` directive, not the line table index.
#[derive(Debug, Clone, Copy)]
pub struct DeclSite {
    pub file: u32,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct FormalParameter {
    pub name: Option<String>,
    pub ty: Option<DebugType>,
}

/// A type a debugger can be told about without guessing. An aggregate whose members
/// the caller cannot enumerate has no variant here on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebugType {
    Base(BaseType),
    /// A pointer to an unspecified type, which is all an untyped machine pointer is.
    Pointer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
}

impl BaseType {
    pub(crate) fn name(self) -> &'static str {
        match self {
            BaseType::Bool => "bool",
            BaseType::I8 => "i8",
            BaseType::I16 => "i16",
            BaseType::I32 => "i32",
            BaseType::I64 => "i64",
            BaseType::U8 => "u8",
            BaseType::U16 => "u16",
            BaseType::U32 => "u32",
            BaseType::U64 => "u64",
            BaseType::F32 => "f32",
            BaseType::F64 => "f64",
        }
    }

    pub(crate) fn byte_size(self) -> u8 {
        match self {
            BaseType::Bool | BaseType::I8 | BaseType::U8 => 1,
            BaseType::I16 | BaseType::U16 => 2,
            BaseType::I32 | BaseType::U32 | BaseType::F32 => 4,
            BaseType::I64 | BaseType::U64 | BaseType::F64 => 8,
        }
    }
}

/// Where a function's code sits, resolved by the assembler after layout.
#[derive(Debug, Clone)]
pub(crate) struct FunctionExtent {
    pub section: String,
    pub start: u64,
    pub length: u64,
}

/// One `.loc` row, resolved to a byte offset in its section.
pub(crate) struct LocRow {
    pub section: String,
    pub offset: u64,
    pub file: u32,
    pub line: u32,
    pub column: u32,
    pub is_stmt: bool,
    pub prologue_end: bool,
    pub epilogue_begin: bool,
    pub basic_block: bool,
    pub discriminator: u64,
    pub isa: u64,
}

/// Everything the assembler resolved that a debug writer describes.
pub(crate) struct DebugSource<'a> {
    pub format: ObjectFormat,
    pub sections: &'a [Section],
    pub rows: &'a [LocRow],
    pub files: &'a BTreeMap<u32, DebugFile>,
    pub extents: &'a HashMap<String, FunctionExtent>,
    pub request: Option<&'a DebugInfoRequest>,
}

/// The sections a debug writer produced, and the fixups they need.
pub(crate) struct DebugSections {
    pub sections: Vec<Section>,
    pub relocs: Vec<ExternalReloc>,
}
