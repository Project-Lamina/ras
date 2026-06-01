//! Core assembler implementation
//!
//! This module contains the main RasAssembler struct and basic operations
//! that are architecture-independent. Uses two-pass assembly for jmp/call
//! with label resolution.

use crate::encoder::traits::InstructionEncoder;
use crate::error::RasError;
use crate::object::{ExternalReloc, ObjectSymbol, ObjectWriteOptions, ObjectWriteRequest, ObjectWriter};
use crate::parser::{AssemblyParser, Line};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::collections::HashMap;

/// Return type of the two-pass encoding pass.
type EncodeResult = Result<(Vec<u8>, Vec<ObjectSymbol>, Vec<ExternalReloc>), RasError>;

#[cfg(windows)]
mod windows_loader {
    use std::ffi::{c_char, c_void};

    unsafe extern "system" {
        pub fn GetModuleHandleA(module_name: *const c_char) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, proc_name: *const c_char) -> *mut c_void;
    }
}

/// Assembler: converts assembly text to object files
pub struct RasAssembler {
    pub(crate) target_arch: TargetArchitecture,
    pub(crate) target_os: TargetOperatingSystem,
    encoder: Box<dyn InstructionEncoder>,
    object_writer: Box<dyn ObjectWriter>,
    object_write_options: ObjectWriteOptions,
    pub(crate) function_pointers: std::collections::HashMap<String, u64>, // Function name -> address
    #[cfg(feature = "encoder")]
    pub(crate) current_module: Option<*const lamina_mir::Module>, // Current module being compiled (for internal call detection)
}

enum PatchPoint {
    X86Rel32 {
        offset: usize,
        target: String,
    },
    X86RipRel32 {
        offset: usize,
        target: String,
    },
    Arx64Jal {
        offset: usize,
        target: String,
        rd: u8,
    },
    Arx64Branch {
        offset: usize,
        target: String,
        rs1: u8,
        rs2: u8,
        funct3: u8,
    },
}

impl RasAssembler {
    /// Create a new assembler for the given target architecture and OS.
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Result<Self, RasError> {
        Self::with_object_write_options(target_arch, target_os, ObjectWriteOptions::default())
    }

    /// Create a new assembler with custom object-file write options.
    pub fn with_object_write_options(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
        object_write_options: ObjectWriteOptions,
    ) -> Result<Self, RasError> {
        // Create encoder based on target architecture
        let encoder: Box<dyn InstructionEncoder> = match target_arch {
            TargetArchitecture::X86_64 => Box::new(crate::encoder::x86_64::X86_64Encoder::new()),
            TargetArchitecture::Aarch64 => Box::new(crate::encoder::aarch64::AArch64Encoder::new()),
            TargetArchitecture::Arx64 => Box::new(crate::encoder::arx64::Arx64Encoder::new()),
            TargetArchitecture::Riscv32 => {
                Box::new(crate::encoder::riscv::RiscVEncoder::new(false))
            }
            TargetArchitecture::Riscv64 => Box::new(crate::encoder::riscv::RiscVEncoder::new(true)),
            _ => {
                return Err(RasError::UnsupportedTarget(
                    crate::target::unsupported_target_hint(target_arch, target_os),
                ));
            }
        };

        let object_writer = match crate::object::object_writer_for_os(target_os) {
            Ok(w) => w,
            Err(_) => {
                return Err(RasError::UnsupportedTarget(format!(
                    "Unsupported OS for cross-compilation: {:?}. {}",
                    target_os,
                    crate::target::unsupported_target_hint(target_arch, target_os)
                )));
            }
        };

        Ok(Self {
            target_arch,
            target_os,
            encoder,
            object_writer,
            object_write_options,
            function_pointers: std::collections::HashMap::new(),
            #[cfg(feature = "encoder")]
            current_module: None,
        })
    }

    /// Replace the object-file write options used by subsequent assembly calls.
    pub fn set_object_write_options(&mut self, opts: ObjectWriteOptions) {
        self.object_write_options = opts;
    }

    /// Assemble `asm_text` and write a relocatable object file to `output_path`.
    ///
    /// Uses two-pass assembly for `jmp`/`call` instructions with forward label
    /// resolution.
    pub fn assemble_text_to_object(
        &mut self,
        asm_text: &str,
        output_path: &std::path::Path,
    ) -> Result<(), RasError> {
        let parsed = AssemblyParser::new()
            .parse(asm_text)
            .map_err(|e| RasError::ParseError(e.to_string()))?;

        let (code, symbols, relocs) = self.encode_lines_two_pass(&parsed.lines)?;

        self.object_writer
            .write_object_file(
                output_path,
                &ObjectWriteRequest {
                    code: &code,
                    sections: &parsed.sections,
                    symbols: &symbols,
                    relocations: &relocs,
                    target_arch: self.target_arch,
                    target_os: self.target_os,
                    opts: &self.object_write_options,
                },
            )
            .map_err(|e| RasError::ObjectError(e.to_string()))?;

        Ok(())
    }

    fn encode_lines_two_pass(&mut self, lines: &[Line]) -> EncodeResult {
        let mut symbol_offsets: HashMap<String, usize> = HashMap::new();
        let mut patch_points: Vec<PatchPoint> = Vec::new();
        let mut code = Vec::new();
        let mut current_offset = 0usize;

        for line in lines {
            match line {
                Line::Label(sym) => {
                    symbol_offsets.insert(sym.name.clone(), current_offset);
                }
                Line::Data(bytes) => {
                    code.extend_from_slice(bytes);
                    current_offset += bytes.len();
                }
                Line::Instruction(inst) => {
                    let opcode = inst.opcode.to_lowercase();
                    let is_jmp_call = opcode == "jmp" || opcode == "jmpq" || opcode == "call";

                    if self.target_arch == TargetArchitecture::Arx64
                        && let Some(patch) =
                            arx64_label_patch(&opcode, &inst.operands, current_offset)?
                        {
                            code.extend_from_slice(&[0u8; 4]);
                            patch_points.push(patch);
                            current_offset += 4;
                            continue;
                        }

                    // Handle leaq/lea with RIP-relative label: "leaq label(%rip), %reg"
                    if self.target_arch == TargetArchitecture::X86_64
                        && (opcode == "leaq" || opcode == "lea")
                        && inst.operands.len() == 2
                        && let Some(label) = extract_rip_label(inst.operands[0].trim())
                    {
                        let reg = parse_x86_reg(inst.operands[1].trim()).map_err(|e| {
                            RasError::EncodingError(e.to_string())
                        })?;
                        // REX.W=1, REX.R set if reg>=8; opcode=0x8D; ModRM Mod=00 R/M=101(RIP-rel)
                        let rex: u8 = 0x48 | ((reg >> 3) << 2);
                        let modrm: u8 = ((reg & 7) << 3) | 5;
                        code.extend_from_slice(&[rex, 0x8D, modrm, 0, 0, 0, 0]);
                        patch_points.push(PatchPoint::X86RipRel32 {
                            offset: current_offset + 3,
                            target: label.to_string(),
                        });
                        current_offset += 7;
                        continue;
                    }

                    // x86_64 conditional jumps: 2-byte opcode (0x0F 8x) + rel32
                    if self.target_arch == TargetArchitecture::X86_64
                        && inst.operands.len() == 1
                        && let Some(cc_byte) = x86_cond_jmp_byte(&opcode)
                    {
                        let target = inst.operands[0].trim();
                        code.extend_from_slice(&[0x0F, cc_byte, 0, 0, 0, 0]);
                        patch_points.push(PatchPoint::X86Rel32 {
                            offset: current_offset + 2,
                            target: target.to_string(),
                        });
                        current_offset += 6;
                        continue;
                    }

                    if is_jmp_call && inst.operands.len() == 1 {
                        let target = inst.operands[0].trim();
                        if self.target_arch == lamina_platform::TargetArchitecture::X86_64 {
                            let is_call = opcode == "call";
                            let opcode_byte: u8 = if is_call { 0xe8 } else { 0xe9 };
                            code.push(opcode_byte);
                            code.extend_from_slice(&[0u8; 4]);
                            patch_points.push(PatchPoint::X86Rel32 {
                                offset: current_offset + 1,
                                target: target.to_string(),
                            });
                            current_offset += 5;
                        } else {
                            let bytes = self
                                .encoder
                                .encode_instruction(inst)
                                .map_err(|e| RasError::EncodingError(e.to_string()))?;
                            code.extend_from_slice(&bytes);
                            current_offset += bytes.len();
                        }
                    } else {
                        let bytes = self
                            .encoder
                            .encode_instruction(inst)
                            .map_err(|e| RasError::EncodingError(e.to_string()))?;
                        code.extend_from_slice(&bytes);
                        current_offset += bytes.len();
                    }
                }
            }
        }

        let mut external_relocs: Vec<ExternalReloc> = Vec::new();

        for patch in &patch_points {
            let target = match patch {
                PatchPoint::X86Rel32 { target, .. }
                | PatchPoint::X86RipRel32 { target, .. }
                | PatchPoint::Arx64Jal { target, .. }
                | PatchPoint::Arx64Branch { target, .. } => target,
            };
            if let Some(&target_offset) = symbol_offsets.get(target) {
                match patch {
                    PatchPoint::X86Rel32 { offset, .. }
                    | PatchPoint::X86RipRel32 { offset, .. } => {
                        let rel32 = (target_offset as i64) - (*offset as i64 + 4);
                        let rel32_bytes = (rel32 as i32).to_le_bytes();
                        code[*offset..*offset + 4].copy_from_slice(&rel32_bytes);
                    }
                    PatchPoint::Arx64Jal { offset, rd, .. } => {
                        let rel = (target_offset as i64) - (*offset as i64);
                        let word = arx64_j_type(rel as i32, *rd);
                        code[*offset..*offset + 4].copy_from_slice(&word.to_le_bytes());
                    }
                    PatchPoint::Arx64Branch {
                        offset,
                        rs1,
                        rs2,
                        funct3,
                        ..
                    } => {
                        let rel = (target_offset as i64) - (*offset as i64);
                        let word = arx64_b_type(rel as i32, *rs2, *rs1, *funct3);
                        code[*offset..*offset + 4].copy_from_slice(&word.to_le_bytes());
                    }
                }
            } else {
                // Unresolved symbol → external relocation for the linker.
                match patch {
                    PatchPoint::X86Rel32 { offset, target }
                    | PatchPoint::X86RipRel32 { offset, target } => {
                        external_relocs.push(ExternalReloc {
                            offset: *offset,
                            symbol: target.clone(),
                        });
                    }
                    _ => {
                        return Err(RasError::EncodingError(format!(
                            "Undefined label: {}",
                            target
                        )));
                    }
                }
            }
        }

        let symbols = lines
            .iter()
            .filter_map(|l| match l {
                Line::Label(s) => Some(ObjectSymbol {
                    name: s.name.clone(),
                    global: s.global,
                    section: s.section.clone(),
                    value: symbol_offsets.get(&s.name).copied().unwrap_or(0) as u64,
                }),
                _ => None,
            })
            .collect();

        Ok((code, symbols, external_relocs))
    }

    /// Register a function pointer for runtime calls.
    ///
    /// Resolves the named symbol using `dlsym` (Unix) or `GetProcAddress` (Windows)
    /// and stores its address for use in generated code.
    pub fn register_function(&mut self, name: &str) -> Result<(), RasError> {
        #[cfg(unix)]
        {
            use std::ffi::CString;

            let symbol = CString::new(name)
                .map_err(|e| RasError::EncodingError(format!("Invalid function name: {}", e)))?;

            // Try to resolve using RTLD_DEFAULT first (searches already loaded libraries)
            // This is safer and doesn't require opening/closing handles
            let ptr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr()) };

            if ptr.is_null() {
                // Fallback: try opening libc explicitly
                // Clear any previous error
                unsafe {
                    libc::dlerror();
                }

                let handle = unsafe { libc::dlopen(std::ptr::null(), libc::RTLD_LAZY) };
                if handle.is_null() {
                    let err_msg = unsafe {
                        let err_ptr = libc::dlerror();
                        if err_ptr.is_null() {
                            "unknown error (dlerror returned null)"
                        } else {
                            std::ffi::CStr::from_ptr(err_ptr)
                                .to_str()
                                .unwrap_or("unknown error")
                        }
                    };
                    return Err(RasError::EncodingError(format!(
                        "Failed to open libc: {}",
                        err_msg
                    )));
                }

                // Clear error before dlsym
                unsafe {
                    libc::dlerror();
                }

                let ptr2 = unsafe { libc::dlsym(handle, symbol.as_ptr()) };
                if ptr2.is_null() {
                    let err_msg = unsafe {
                        let err_ptr = libc::dlerror();
                        if err_ptr.is_null() {
                            "symbol not found"
                        } else {
                            std::ffi::CStr::from_ptr(err_ptr)
                                .to_str()
                                .unwrap_or("unknown error")
                        }
                    };
                    unsafe { libc::dlclose(handle) };
                    return Err(RasError::EncodingError(format!(
                        "Failed to resolve symbol {}: {}",
                        name, err_msg
                    )));
                }

                self.function_pointers.insert(name.to_string(), ptr2 as u64);
                unsafe { libc::dlclose(handle) };
            } else {
                self.function_pointers.insert(name.to_string(), ptr as u64);
            }

            Ok(())
        }

        #[cfg(windows)]
        {
            use std::ffi::CString;
            use windows_loader::{GetModuleHandleA, GetProcAddress};

            let module = unsafe { GetModuleHandleA(c"msvcrt.dll".as_ptr() as *const i8) };
            if module.is_null() {
                return Err(RasError::EncodingError(
                    "Failed to get msvcrt.dll handle".to_string(),
                ));
            }

            let symbol = CString::new(name)
                .map_err(|e| RasError::EncodingError(format!("Invalid function name: {}", e)))?;

            let ptr = unsafe { GetProcAddress(module, symbol.as_ptr()) };
            if ptr.is_null() {
                return Err(RasError::EncodingError(format!(
                    "Failed to resolve symbol {}",
                    name
                )));
            }

            self.function_pointers.insert(name.to_string(), ptr as u64);
            Ok(())
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::EncodingError(
                "Runtime function resolution not supported on this platform".to_string(),
            ))
        }
    }

    /// Compile all functions in a MIR module to machine code and return the raw bytes.
    ///
    /// Equivalent to calling [`compile_mir_to_binary_function`] with `function_name = None`.
    /// Requires the `encoder` feature.
    ///
    /// [`compile_mir_to_binary_function`]: Self::compile_mir_to_binary_function
    #[cfg(feature = "encoder")]
    pub fn compile_mir_to_binary(
        &mut self,
        module: &lamina_mir::Module,
    ) -> Result<Vec<u8>, RasError> {
        let (code, _) = self.compile_mir_to_binary_function(module, None)?;
        Ok(code)
    }

    /// Compile a specific function from a MIR module to binary.
    ///
    /// If `function_name` is `None`, all functions in the module are compiled.
    /// Returns `(binary_code, function_offsets)` where `function_offsets` maps
    /// each function name to its byte offset within `binary_code`.
    #[cfg(feature = "encoder")]
    pub fn compile_mir_to_binary_function(
        &mut self,
        module: &lamina_mir::Module,
        function_name: Option<&str>,
    ) -> Result<(Vec<u8>, std::collections::HashMap<String, usize>), RasError> {
        // Store module reference for checking internal vs external calls
        self.current_module = Some(module);
        // Reuse register allocation and ABI from mir_codegen
        match self.target_arch {
            TargetArchitecture::X86_64 => {
                crate::assembler::x86_64::compile_mir_x86_64_function(self, module, function_name)
            }
            TargetArchitecture::Aarch64 => {
                crate::assembler::aarch64::compile_mir_aarch64_function(self, module, function_name)
            }
            TargetArchitecture::Riscv64 => crate::assembler::riscv::compile_mir_riscv_function(
                self,
                module,
                function_name,
                true,
            ),
            TargetArchitecture::Riscv32 => crate::assembler::riscv::compile_mir_riscv_function(
                self,
                module,
                function_name,
                false,
            ),
            _ => Err(RasError::UnsupportedTarget(format!(
                "MIR compilation not supported for architecture: {:?}",
                self.target_arch
            ))),
        }
    }
}

fn arx64_label_patch(
    opcode: &str,
    operands: &[String],
    offset: usize,
) -> Result<Option<PatchPoint>, RasError> {
    match opcode {
        "j" if operands.len() == 1 && !is_numeric(&operands[0]) => Ok(Some(PatchPoint::Arx64Jal {
            offset,
            target: operands[0].trim().to_string(),
            rd: 0,
        })),
        "call" if operands.len() == 1 && !is_numeric(&operands[0]) => {
            Ok(Some(PatchPoint::Arx64Jal {
                offset,
                target: operands[0].trim().to_string(),
                rd: 1,
            }))
        }
        "jal" if operands.len() == 2 && !is_numeric(&operands[1]) => {
            Ok(Some(PatchPoint::Arx64Jal {
                offset,
                target: operands[1].trim().to_string(),
                rd: parse_arx64_reg(&operands[0])?,
            }))
        }
        "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu"
            if operands.len() == 3 && !is_numeric(&operands[2]) =>
        {
            Ok(Some(PatchPoint::Arx64Branch {
                offset,
                target: operands[2].trim().to_string(),
                rs1: parse_arx64_reg(&operands[0])?,
                rs2: parse_arx64_reg(&operands[1])?,
                funct3: match opcode {
                    "beq" => 0x0,
                    "bne" => 0x1,
                    "blt" => 0x4,
                    "bge" => 0x5,
                    "bltu" => 0x6,
                    "bgeu" => 0x7,
                    _ => unreachable!(),
                },
            }))
        }
        _ => Ok(None),
    }
}

fn is_numeric(value: &str) -> bool {
    let value = value.trim();
    value.parse::<i64>().is_ok()
        || value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"))
            .is_some_and(|hex| i64::from_str_radix(hex, 16).is_ok())
}

fn parse_arx64_reg(value: &str) -> Result<u8, RasError> {
    let value = value.trim().trim_start_matches('%');
    let raw = match value {
        "zero" => 0,
        "ra" | "lr" => 1,
        "sp" => 2,
        _ => value
            .strip_prefix('r')
            .or_else(|| value.strip_prefix('x'))
            .ok_or_else(|| RasError::EncodingError(format!("Unknown ARX64 register: {}", value)))?
            .parse::<u8>()
            .map_err(|_| RasError::EncodingError(format!("Unknown ARX64 register: {}", value)))?,
    };
    if raw < 32 {
        Ok(raw)
    } else {
        Err(RasError::EncodingError(format!(
            "ARX64 register out of range: {}",
            value
        )))
    }
}

fn arx64_j_type(offset: i32, rd: u8) -> u32 {
    let o = offset as u32;
    (((o >> 20) & 0x1) << 31)
        | (((o >> 1) & 0x03ff) << 21)
        | (((o >> 11) & 0x1) << 20)
        | (((o >> 12) & 0xff) << 12)
        | ((rd as u32) << 7)
        | 0x6f
}

fn arx64_b_type(offset: i32, rs2: u8, rs1: u8, funct3: u8) -> u32 {
    let o = offset as u32;
    (((o >> 12) & 0x1) << 31)
        | (((o >> 5) & 0x3f) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | ((funct3 as u32) << 12)
        | (((o >> 1) & 0x0f) << 8)
        | (((o >> 11) & 0x1) << 7)
        | 0x63
}

/// Maps x86 conditional-jump mnemonic to the second opcode byte of the `0F 8x rel32` encoding.
/// Returns `None` for non-conditional-jump mnemonics.
fn x86_cond_jmp_byte(opcode: &str) -> Option<u8> {
    match opcode {
        "jo" => Some(0x80),
        "jno" => Some(0x81),
        "jb" | "jnae" | "jc" => Some(0x82),
        "jnb" | "jae" | "jnc" => Some(0x83),
        "je" | "jz" => Some(0x84),
        "jne" | "jnz" => Some(0x85),
        "jbe" | "jna" => Some(0x86),
        "ja" | "jnbe" => Some(0x87),
        "js" => Some(0x88),
        "jns" => Some(0x89),
        "jp" | "jpe" => Some(0x8A),
        "jnp" | "jpo" => Some(0x8B),
        "jl" | "jnge" => Some(0x8C),
        "jge" | "jnl" => Some(0x8D),
        "jle" | "jng" => Some(0x8E),
        "jg" | "jnle" => Some(0x8F),
        _ => None,
    }
}

/// Returns the label name if `op` has the form `label(%rip)` (RIP-relative addressing).
fn extract_rip_label(op: &str) -> Option<&str> {
    let paren = op.find('(')?;
    let close = op.find(')')?;
    if close <= paren {
        return None;
    }
    let base = op[paren + 1..close].trim().trim_start_matches('%');
    if !base.eq_ignore_ascii_case("rip") {
        return None;
    }
    let label = op[..paren].trim();
    // Must be a non-empty label name (not a numeric displacement)
    if label.is_empty() || label.starts_with(|c: char| c.is_ascii_digit() || c == '-') {
        return None;
    }
    Some(label)
}

fn parse_x86_reg(s: &str) -> Result<u8, crate::error::RasError> {
    let s = s.trim().trim_start_matches('%');
    match s {
        "rax" | "eax" => Ok(0),
        "rcx" | "ecx" => Ok(1),
        "rdx" | "edx" => Ok(2),
        "rbx" | "ebx" => Ok(3),
        "rsp" | "esp" => Ok(4),
        "rbp" | "ebp" => Ok(5),
        "rsi" | "esi" => Ok(6),
        "rdi" | "edi" => Ok(7),
        "r8" => Ok(8),
        "r9" => Ok(9),
        "r10" => Ok(10),
        "r11" => Ok(11),
        "r12" => Ok(12),
        "r13" => Ok(13),
        "r14" => Ok(14),
        "r15" => Ok(15),
        _ => Err(crate::error::RasError::EncodingError(format!(
            "Unknown x86 register: {}",
            s
        ))),
    }
}
