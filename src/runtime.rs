//! Runtime compilation (JIT): MIR to executable memory

use crate::error::RasError;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

#[cfg(feature = "encoder")]
use crate::assembler::RasAssembler;

#[cfg(feature = "encoder")]
use lamina_mir::Module as MirModule;

/// Executable memory for JIT-compiled code
pub struct ExecutableMemory {
    ptr: *mut u8,
    size: usize,
}

impl ExecutableMemory {
    pub fn allocate_writable(size: usize) -> Result<Self, RasError> {
        #[cfg(unix)]
        {
            use libc::{MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap};
            use std::ptr;

            let aligned_size = (size + 4095) & !4095;

            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            const MAP_JIT: libc::c_int = 0x0800;

            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let map_flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_JIT;

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let map_flags = MAP_ANONYMOUS | MAP_PRIVATE;

            let ptr = unsafe {
                mmap(
                    ptr::null_mut(),
                    aligned_size,
                    PROT_READ | PROT_WRITE,
                    map_flags,
                    -1,
                    0,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(RasError::IoError(format!(
                    "mmap failed (size: {} bytes)",
                    size
                )));
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size: aligned_size,
            })
        }

        #[cfg(windows)]
        {
            use winapi::um::memoryapi::VirtualAlloc;
            use winapi::um::winnt::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE};

            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                )
            };

            if ptr.is_null() {
                return Err(RasError::IoError("VirtualAlloc failed".to_string()));
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size,
            })
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::UnsupportedTarget(
                "Executable memory allocation not supported on this platform".to_string(),
            ))
        }
    }

    pub fn make_executable(&mut self) -> Result<(), RasError> {
        #[cfg(unix)]
        {
            use libc::{PROT_EXEC, PROT_READ, mprotect};

            let result = unsafe {
                mprotect(
                    self.ptr as *mut libc::c_void,
                    self.size,
                    PROT_READ | PROT_EXEC,
                )
            };

            if result != 0 {
                return Err(RasError::IoError("mprotect failed".to_string()));
            }

            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                    unsafe extern "C" {
                        fn __clear_cache(start: *const u8, end: *const u8);
                    }
                    __clear_cache(self.ptr, self.ptr.add(self.size));
                    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
                }
            }

            Ok(())
        }

        #[cfg(windows)]
        {
            use winapi::um::memoryapi::VirtualProtect;
            use winapi::um::winnt::PAGE_EXECUTE_READ;

            let mut old_protect: winapi::um::winnt::DWORD = 0;
            let result = unsafe {
                VirtualProtect(
                    self.ptr as *mut winapi::um::winnt::c_void,
                    self.size,
                    PAGE_EXECUTE_READ,
                    &mut old_protect,
                )
            };

            if result == 0 {
                return Err(RasError::IoError("VirtualProtect failed".to_string()));
            }

            Ok(())
        }

        #[cfg(not(any(unix, windows)))]
        {
            Err(RasError::UnsupportedTarget(
                "Making memory executable not supported".to_string(),
            ))
        }
    }

    pub fn allocate(size: usize) -> Result<Self, RasError> {
        let mut mem = Self::allocate_writable(size)?;
        mem.make_executable()?;
        Ok(mem)
    }

    pub fn write_code(&mut self, code: &[u8]) -> Result<(), RasError> {
        if code.len() > self.size {
            return Err(RasError::IoError("Code too large".to_string()));
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe {
                unsafe extern "C" {
                    fn pthread_jit_write_protect_np(value: libc::c_int);
                }
                pthread_jit_write_protect_np(0);
            }
        }

        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len());
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe {
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *mut libc::c_void, size: libc::size_t);
                }
                sys_icache_invalidate(self.ptr as *mut libc::c_void, code.len());
            }
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe {
                unsafe extern "C" {
                    fn pthread_jit_write_protect_np(value: libc::c_int);
                }
                pthread_jit_write_protect_np(1);
            }
        }

        Ok(())
    }

    pub fn code_start(&self) -> *const u8 {
        self.ptr
    }

    /// Interprets the first byte of this region as the entry address for a function pointer.
    ///
    /// This is **not** a pointer-to-pointer: the mapping begins with machine code, not a stored
    /// address. Do not use `*const F` to the entry and then dereference; that would load
    /// instruction bytes as a pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure `F` matches the ABI and signature of the generated code, and that
    /// this `ExecutableMemory` outlives every call through `F`.
    pub unsafe fn entry_fn<F: Sized>(&self) -> F {
        debug_assert_eq!(
            core::mem::size_of::<F>(),
            core::mem::size_of::<*mut u8>()
        );
        unsafe { core::mem::transmute_copy(&self.ptr) }
    }
}

impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use libc::munmap;
            unsafe {
                munmap(self.ptr as *mut libc::c_void, self.size);
            }
        }

        #[cfg(windows)]
        {
            use winapi::um::memoryapi::VirtualFree;
            use winapi::um::winnt::MEM_RELEASE;
            unsafe {
                VirtualFree(self.ptr as *mut winapi::um::winnt::c_void, 0, MEM_RELEASE);
            }
        }
    }
}

/// Runtime compiler: MIR to executable memory
pub struct RasRuntime {
    #[cfg(feature = "encoder")]
    target_arch: TargetArchitecture,
    #[cfg(feature = "encoder")]
    target_os: TargetOperatingSystem,
}

impl RasRuntime {
    pub fn new(
        #[cfg(feature = "encoder")] target_arch: TargetArchitecture,
        #[cfg(not(feature = "encoder"))] _target_arch: TargetArchitecture,
        #[cfg(feature = "encoder")] target_os: TargetOperatingSystem,
        #[cfg(not(feature = "encoder"))] _target_os: TargetOperatingSystem,
    ) -> Self {
        Self {
            #[cfg(feature = "encoder")]
            target_arch,
            #[cfg(feature = "encoder")]
            target_os,
        }
    }

    #[cfg(feature = "encoder")]
    pub fn compile_to_memory(&mut self, module: &MirModule) -> Result<ExecutableMemory, RasError> {
        let mut assembler = RasAssembler::new(self.target_arch, self.target_os)?;
        let (code, _) = assembler.compile_mir_to_binary_function(module, None)?;

        let mut mem = ExecutableMemory::allocate_writable(code.len())?;
        mem.write_code(&code)?;
        mem.make_executable()?;

        Ok(mem)
    }

    /// Leaks the executable mapping so the returned pointer stays valid. Prefer
    /// [`compile_to_memory`](Self::compile_to_memory) and keep the `ExecutableMemory` alive.
    #[cfg(feature = "encoder")]
    pub fn compile_function<T>(
        &mut self,
        module: &MirModule,
        _function_name: &str,
    ) -> Result<unsafe extern "C" fn() -> T, RasError> {
        let mem = self.compile_to_memory(module)?;
        let f: unsafe extern "C" fn() -> T = unsafe { core::mem::transmute(mem.ptr) };
        std::mem::forget(mem);
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    #[test]
    fn executable_memory_allocate_and_write() {
        // Allocate a small writable+executable region and verify the written bytes
        // can be read back before making the region executable.
        let mut mem = ExecutableMemory::allocate_writable(4096).expect("alloc failed");
        let code = [0x90u8, 0x90, 0x90, 0xc3]; // three NOPs + ret (x86_64)
        mem.write_code(&code).expect("write failed");
        // After making it executable the pointer should remain non-null.
        mem.make_executable().expect("make_executable failed");
        assert!(!mem.code_start().is_null());
    }

    #[test]
    fn executable_memory_too_large_returns_error() {
        // Allocation rounds up to a page (4096 bytes). Use exactly one page so
        // the next allocation produces a 4096-byte region, then try to write 8192.
        let mem = ExecutableMemory::allocate_writable(4096).expect("alloc failed");
        let big = vec![0u8; 8192];
        let result = { let mut m = mem; m.write_code(&big) };
        assert!(result.is_err(), "writing beyond capacity should fail");
    }

    #[test]
    fn ras_runtime_new_does_not_panic() {
        let _rt = RasRuntime::new(
            TargetArchitecture::X86_64,
            TargetOperatingSystem::Linux,
        );
    }
}

/// Run JIT output on the **host** CPU (x86_64 or aarch64 only). Skips when the host OS string
/// is not mapped in `TargetOperatingSystem` (e.g. some embedded targets).
#[cfg(all(test, feature = "encoder"))]
mod jit_host_exec_tests {
    use std::str::FromStr;

    use lamina_mir::block::Block;
    use lamina_mir::function::{Function, Parameter, Signature};
    use lamina_mir::instruction::{Immediate, Instruction, IntBinOp, Operand};
    use lamina_mir::module::Module;
    use lamina_mir::register::{Register, VirtualReg};
    use lamina_mir::types::{MirType, ScalarType};
    use lamina_platform::{
        TargetArchitecture, TargetOperatingSystem, detect_host_architecture_only, detect_host_os,
    };

    use crate::assembler::core::RasAssembler;
    use crate::error::RasError;
    use crate::runtime::ExecutableMemory;

    fn host_jit_pair() -> Option<(TargetArchitecture, TargetOperatingSystem)> {
        let arch = TargetArchitecture::from_str(detect_host_architecture_only()).ok()?;
        let os = TargetOperatingSystem::from_str(detect_host_os()).ok()?;
        match arch {
            TargetArchitecture::X86_64 | TargetArchitecture::Aarch64 => Some((arch, os)),
            _ => None,
        }
    }

    fn compile_to_executable(
        arch: TargetArchitecture,
        os: TargetOperatingSystem,
        module: &lamina_mir::Module,
    ) -> Result<ExecutableMemory, RasError> {
        let mut asm = RasAssembler::new(arch, os)?;
        let (code, _) = asm.compile_mir_to_binary_function(module, None)?;
        let mut mem = ExecutableMemory::allocate_writable(code.len())?;
        mem.write_code(&code)?;
        mem.make_executable()?;
        Ok(mem)
    }

    fn module_i64_binop(
        op: IntBinOp,
        lhs: Operand,
        rhs: Operand,
    ) -> Module {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let out = Register::Virtual(VirtualReg::gpr(0));
        let sig = Signature::new("f").with_return(i64_ty.clone());
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op,
            ty: i64_ty.clone(),
            dst: out.clone(),
            lhs,
            rhs,
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("exec_i64");
        module.add_function(f);
        module
    }

    fn module_i32_binop(op: IntBinOp, lhs: Operand, rhs: Operand) -> Module {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let out = Register::Virtual(VirtualReg::gpr(0));
        let sig = Signature::new("f").with_return(i32_ty.clone());
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op,
            ty: i32_ty.clone(),
            dst: out.clone(),
            lhs,
            rhs,
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("exec_i32");
        module.add_function(f);
        module
    }

    fn module_i64_add_params() -> Module {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("g")
            .with_params(vec![
                Parameter::new(a.clone(), i64_ty.clone()),
                Parameter::new(b.clone(), i64_ty.clone()),
            ])
            .with_return(i64_ty.clone());
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty.clone(),
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("exec_i64_params");
        module.add_function(f);
        module
    }

    #[test]
    fn jit_host_exec_i64_add_immediates() {
        let Some((arch, os)) = host_jit_pair() else {
            return;
        };
        let module = module_i64_binop(
            IntBinOp::Add,
            Operand::Immediate(Immediate::I64(10)),
            Operand::Immediate(Immediate::I64(32)),
        );
        let mem = compile_to_executable(arch, os, &module).expect("compile to executable");
        let fptr = unsafe { mem.entry_fn::<extern "C" fn() -> i64>() };
        let got = fptr();
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i64_add_two_arguments() {
        let Some((arch, os)) = host_jit_pair() else {
            return;
        };
        let module = module_i64_add_params();
        let mem = compile_to_executable(arch, os, &module).expect("compile to executable");
        let fptr = unsafe { mem.entry_fn::<extern "C" fn(i64, i64) -> i64>() };
        let got = fptr(7, 35);
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i32_udiv_immediates() {
        let Some((arch, os)) = host_jit_pair() else {
            return;
        };
        let module = module_i32_binop(
            IntBinOp::UDiv,
            Operand::Immediate(Immediate::I32(10)),
            Operand::Immediate(Immediate::I32(3)),
        );
        let mem = compile_to_executable(arch, os, &module).expect("compile to executable");
        let fptr = unsafe { mem.entry_fn::<extern "C" fn() -> i32>() };
        let got = fptr();
        assert_eq!(got, 3);
    }

    #[test]
    fn jit_host_exec_i32_sdiv_negative_dividend() {
        let Some((arch, os)) = host_jit_pair() else {
            return;
        };
        let module = module_i32_binop(
            IntBinOp::SDiv,
            Operand::Immediate(Immediate::I32(-7)),
            Operand::Immediate(Immediate::I32(2)),
        );
        let mem = compile_to_executable(arch, os, &module).expect("compile to executable");
        let fptr = unsafe { mem.entry_fn::<extern "C" fn() -> i32>() };
        let got = fptr();
        assert_eq!(got, -3);
    }

    #[test]
    fn jit_host_exec_i32_shl_immediates() {
        let Some((arch, os)) = host_jit_pair() else {
            return;
        };
        let module = module_i32_binop(
            IntBinOp::Shl,
            Operand::Immediate(Immediate::I32(3)),
            Operand::Immediate(Immediate::I32(4)),
        );
        let mem = compile_to_executable(arch, os, &module).expect("compile to executable");
        let fptr = unsafe { mem.entry_fn::<extern "C" fn() -> i32>() };
        let got = fptr();
        assert_eq!(got, 48);
    }
}
