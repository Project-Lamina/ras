//! Core assembler implementation
//!
//! This module contains the main RasAssembler struct and basic operations
//! that are architecture-independent. Uses two-pass assembly for jmp/call
//! with label resolution.

use crate::encoder::traits::InstructionEncoder;
use crate::error::RasError;
use crate::object::ObjectWriter;
use crate::parser::{AssemblyParser, Line};
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};
use std::collections::HashMap;

/// Assembler: converts assembly text to object files
pub struct RasAssembler {
    pub(crate) target_arch: TargetArchitecture,
    pub(crate) target_os: TargetOperatingSystem,
    encoder: Box<dyn InstructionEncoder>,
    object_writer: Box<dyn ObjectWriter>,
    pub(crate) function_pointers: std::collections::HashMap<String, u64>, // Function name -> address
    #[cfg(feature = "encoder")]
    pub(crate) current_module: Option<*const lamina_mir::Module>, // Current module being compiled (for internal call detection)
}

impl RasAssembler {
    /// Create a new assembler
    pub fn new(
        target_arch: TargetArchitecture,
        target_os: TargetOperatingSystem,
    ) -> Result<Self, RasError> {
        // Create encoder based on target architecture
        let encoder: Box<dyn InstructionEncoder> = match target_arch {
            TargetArchitecture::X86_64 => {
                Box::new(crate::encoder::x86_64::X86_64Encoder::new())
            }
            TargetArchitecture::Aarch64 => {
                Box::new(crate::encoder::aarch64::AArch64Encoder::new())
            }
            TargetArchitecture::Riscv32 | TargetArchitecture::Riscv64 => {
                return Err(RasError::UnsupportedTarget(format!(
                    "RISC-V binary encoder not yet implemented. {}",
                    crate::target::unsupported_target_hint(target_arch, target_os)
                )));
            }
            _ => {
                return Err(RasError::UnsupportedTarget(
                    crate::target::unsupported_target_hint(target_arch, target_os),
                ));
            }
        };

        // Create object writer based on target OS (cross-compilation: use desired target, not host)
        let object_writer: Box<dyn ObjectWriter> = match target_os {
            TargetOperatingSystem::Linux
            | TargetOperatingSystem::FreeBSD
            | TargetOperatingSystem::OpenBSD
            | TargetOperatingSystem::NetBSD
            | TargetOperatingSystem::DragonFly
            | TargetOperatingSystem::Redox => {
                Box::new(crate::object::ElfWriter::new())
            }
            TargetOperatingSystem::MacOS => {
                Box::new(crate::object::MachOWriter::new())
            }
            TargetOperatingSystem::Windows => {
                Box::new(crate::object::CoffWriter::new())
            }
            _ => {
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
            function_pointers: std::collections::HashMap::new(),
            #[cfg(feature = "encoder")]
            current_module: None,
        })
    }

    /// Assemble assembly text to object file.
    /// Uses two-pass assembly for jmp/call with label resolution (x86_64).
    pub fn assemble_text_to_object(
        &mut self,
        asm_text: &str,
        output_path: &std::path::Path,
    ) -> Result<(), RasError> {
        let parsed = AssemblyParser::new()
            .parse(asm_text)
            .map_err(|e| RasError::ParseError(e.to_string()))?;

        let (code, symbols) = self.encode_lines_two_pass(&parsed.lines)?;

        self.object_writer
            .write_object_file(
                output_path,
                &code,
                &parsed.sections,
                &symbols,
                self.target_arch,
                self.target_os,
            )
            .map_err(|e| RasError::ObjectError(e.to_string()))?;

        Ok(())
    }

    fn encode_lines_two_pass(
        &mut self,
        lines: &[Line],
    ) -> Result<(Vec<u8>, Vec<crate::parser::Symbol>), RasError> {
        let mut symbol_offsets: HashMap<String, usize> = HashMap::new();
        let mut patch_points: Vec<(usize, String)> = Vec::new();
        let mut code = Vec::new();
        let mut current_offset = 0usize;

        for line in lines {
            match line {
                Line::Label(sym) => {
                    symbol_offsets.insert(sym.name.clone(), current_offset);
                }
                Line::Instruction(inst) => {
                    let opcode = inst.opcode.to_lowercase();
                    let is_jmp_call = opcode == "jmp" || opcode == "jmpq" || opcode == "call";

                    if is_jmp_call && inst.operands.len() == 1 {
                        let target = inst.operands[0].trim();
                        if self.target_arch == lamina_platform::TargetArchitecture::X86_64 {
                            let is_call = opcode == "call";
                            let opcode_byte: u8 = if is_call { 0xe8 } else { 0xe9 };
                            code.push(opcode_byte);
                            code.extend_from_slice(&[0u8; 4]);
                            patch_points.push((current_offset + 1, target.to_string()));
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

        for (patch_offset, target) in &patch_points {
            let target_offset = symbol_offsets.get(target).ok_or_else(|| {
                RasError::EncodingError(format!("Undefined label: {}", target))
            })?;
            let rel32 = (*target_offset as i64) - (*patch_offset as i64 + 4);
            let rel32_bytes = (rel32 as i32).to_le_bytes();
            code[*patch_offset..*patch_offset + 4].copy_from_slice(&rel32_bytes);
        }

        let symbols = lines
            .iter()
            .filter_map(|l| match l {
                Line::Label(s) => Some(s.clone()),
                _ => None,
            })
            .collect();

        Ok((code, symbols))
    }

    /// Register a function pointer for runtime calls
    /// This resolves the function using dlsym (on Unix) or GetProcAddress (on Windows)
    pub fn register_function(&mut self, name: &str) -> Result<(), RasError> {
        #[cfg(unix)]
        {
            use std::ffi::CString;
            
            let symbol = CString::new(name).map_err(|e| {
                RasError::EncodingError(format!("Invalid function name: {}", e))
            })?;
            
            // Try to resolve using RTLD_DEFAULT first (searches already loaded libraries)
            // This is safer and doesn't require opening/closing handles
            let ptr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr()) };
            
            if ptr.is_null() {
                // Fallback: try opening libc explicitly
                // Clear any previous error
                unsafe { libc::dlerror(); }
                
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
                    return Err(RasError::EncodingError(
                        format!("Failed to open libc: {}", err_msg)
                    ));
                }
                
                // Clear error before dlsym
                unsafe { libc::dlerror(); }
                
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
                    return Err(RasError::EncodingError(
                        format!("Failed to resolve symbol {}: {}", name, err_msg)
                    ));
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
            use winapi::um::libloaderapi::{GetModuleHandleA, GetProcAddress};
            use std::ffi::CString;
            
            let module = unsafe { GetModuleHandleA(b"msvcrt.dll\0".as_ptr() as *const i8) };
            if module.is_null() {
                return Err(RasError::EncodingError("Failed to get msvcrt.dll handle".to_string()));
            }
            
            let symbol = CString::new(name).map_err(|e| {
                RasError::EncodingError(format!("Invalid function name: {}", e))
            })?;
            
            let ptr = unsafe { GetProcAddress(module, symbol.as_ptr()) };
            if ptr.is_null() {
                return Err(RasError::EncodingError(
                    format!("Failed to resolve symbol {}", name)
                ));
            }
            
            self.function_pointers.insert(name.to_string(), ptr as u64);
            Ok(())
        }
        
        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::EncodingError(
                "Runtime function resolution not supported on this platform".to_string()
            ))
        }
    }

    /// Compile MIR module directly to binary (for runtime compilation)
    ///
    /// This method reuses code from lamina's mir_codegen but generates binary
    /// instead of assembly text. It's used for runtime compilation (JIT).
    ///
    /// Requires the `mir` feature to be enabled.
    #[cfg(feature = "encoder")]
    pub fn compile_mir_to_binary(
        &mut self,
        module: &lamina_mir::Module,
    ) -> Result<Vec<u8>, RasError> {
        let (code, _) = self.compile_mir_to_binary_function(module, None)?;
        Ok(code)
    }

    /// Compile a specific function from MIR module to binary
    /// If function_name is None, compiles all functions
    /// Returns (binary code, function_offsets map)
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
            _ => Err(RasError::UnsupportedTarget(format!(
                "MIR compilation not supported for architecture: {:?}",
                self.target_arch
            ))),
        }
    }
}





