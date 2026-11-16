//! Runtime compilation (JIT): MIR to executable memory

use crate::error::RasError;
use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

#[cfg(windows)]
mod windows_memory {
    use std::ffi::c_void;

    pub const MEM_COMMIT: u32 = 0x1000;
    pub const MEM_RELEASE: u32 = 0x8000;
    pub const MEM_RESERVE: u32 = 0x2000;
    pub const PAGE_EXECUTE_READ: u32 = 0x20;
    pub const PAGE_READWRITE: u32 = 0x04;

    unsafe extern "system" {
        pub fn VirtualAlloc(
            address: *mut c_void,
            size: usize,
            allocation_type: u32,
            protect: u32,
        ) -> *mut c_void;
        pub fn VirtualFree(address: *mut c_void, size: usize, free_type: u32) -> i32;
        pub fn VirtualProtect(
            address: *mut c_void,
            size: usize,
            new_protect: u32,
            old_protect: *mut u32,
        ) -> i32;

        #[cfg(target_arch = "aarch64")]
        pub fn FlushInstructionCache(
            process: *mut c_void,
            base_address: *const c_void,
            size: usize,
        ) -> i32;
        #[cfg(target_arch = "aarch64")]
        pub fn GetCurrentProcess() -> *mut c_void;
    }
}

#[cfg(feature = "encoder")]
use crate::assembler::RasAssembler;

#[cfg(feature = "encoder")]
use lamina_mir::Module as MirModule;
#[cfg(feature = "encoder")]
use std::collections::HashMap;

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
            use crate::runtime::windows_memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };

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
            use crate::runtime::windows_memory::{PAGE_EXECUTE_READ, VirtualProtect};

            let mut old_protect = 0;
            let result = unsafe {
                VirtualProtect(
                    self.ptr.cast(),
                    self.size,
                    PAGE_EXECUTE_READ,
                    &mut old_protect,
                )
            };

            if result == 0 {
                return Err(RasError::IoError("VirtualProtect failed".to_string()));
            }

            // On Windows AArch64, flush the instruction cache so the CPU sees
            // the newly written code after the page protection change.
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use crate::runtime::windows_memory::{FlushInstructionCache, GetCurrentProcess};
                FlushInstructionCache(GetCurrentProcess(), self.ptr.cast(), self.size);
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

    /// Writes `bytes` at `offset`, for load-time values of a JIT image. Call before
    /// `make_executable`, while the mapping is still writable.
    pub(crate) fn patch(&mut self, offset: usize, bytes: &[u8]) -> Result<(), RasError> {
        if offset
            .checked_add(bytes.len())
            .is_none_or(|end| end > self.size)
        {
            return Err(RasError::IoError(
                "JIT image patch out of range".to_string(),
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len());
        }
        Ok(())
    }

    /// Makes the pages from `offset` to the end readable and writable, not executable.
    /// `offset` must be a page boundary.
    pub(crate) fn make_writable_from(&mut self, offset: usize) -> Result<(), RasError> {
        if offset >= self.size {
            return Ok(());
        }
        #[cfg(unix)]
        {
            use libc::{PROT_READ, PROT_WRITE, mprotect};
            let result = unsafe {
                mprotect(
                    self.ptr.add(offset).cast(),
                    self.size - offset,
                    PROT_READ | PROT_WRITE,
                )
            };
            if result != 0 {
                return Err(RasError::IoError("mprotect failed".to_string()));
            }
        }
        #[cfg(windows)]
        {
            use crate::runtime::windows_memory::{PAGE_READWRITE, VirtualProtect};
            let mut old_protect = 0;
            let result = unsafe {
                VirtualProtect(
                    self.ptr.add(offset).cast(),
                    self.size - offset,
                    PAGE_READWRITE,
                    &mut old_protect,
                )
            };
            if result == 0 {
                return Err(RasError::IoError("VirtualProtect failed".to_string()));
            }
        }
        Ok(())
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

    /// Bytes mapped for this region, rounded up to a page.
    pub fn capacity(&self) -> usize {
        self.size
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
        debug_assert_eq!(size_of::<F>(), size_of::<*mut u8>());
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
            use crate::runtime::windows_memory::{MEM_RELEASE, VirtualFree};
            unsafe {
                VirtualFree(self.ptr.cast(), 0, MEM_RELEASE);
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
        self.compile_with_offsets(module).map(|(mem, _)| mem)
    }

    /// Leaks the executable mapping so the returned pointer stays valid. Prefer
    /// [`compile_to_memory`](Self::compile_to_memory) and keep the `ExecutableMemory` alive.
    ///
    /// `function_name` is matched with or without a leading `@`.
    #[cfg(feature = "encoder")]
    pub fn compile_function<T>(
        &mut self,
        module: &MirModule,
        function_name: &str,
    ) -> Result<unsafe extern "C" fn() -> T, RasError> {
        let (mem, offsets) = self.compile_with_offsets(module)?;
        let offset = entry_offset(&offsets, function_name)?;
        let entry = mem.ptr.wrapping_add(offset);
        let f: unsafe extern "C" fn() -> T = unsafe { core::mem::transmute(entry) };
        std::mem::forget(mem);
        Ok(f)
    }

    /// Compiles every function and global of `module` into one mapping and returns it with
    /// the offset of each function from its start.
    ///
    /// Read-only data is mapped with the code. Writable globals sit on their own pages,
    /// which stay writable after the mapping is made executable.
    #[cfg(feature = "encoder")]
    pub fn compile_with_offsets(
        &mut self,
        module: &MirModule,
    ) -> Result<(ExecutableMemory, HashMap<String, usize>), RasError> {
        let mut assembler = RasAssembler::new(self.target_arch, self.target_os)?;
        let (code, offsets) = assembler.compile_mir_to_binary_function(module, None)?;
        let image = assembler.jit_image.take();

        let mut mem = ExecutableMemory::allocate_writable(code.len())?;
        mem.write_code(&code)?;
        if let Some(image) = &image {
            let base = mem.code_start() as usize as i64;
            for reloc in &image.relocs {
                let value = base
                    .wrapping_add(reloc.target as i64)
                    .wrapping_add(reloc.addend) as u64;
                mem.patch(reloc.at, &value.to_le_bytes()[..reloc.size])?;
            }
        }
        mem.make_executable()?;
        if let Some(image) = &image
            && image.writable_start < code.len()
        {
            mem.make_writable_from(image.writable_start)?;
        }

        Ok((mem, offsets))
    }
}

#[cfg(feature = "encoder")]
fn entry_offset(offsets: &HashMap<String, usize>, name: &str) -> Result<usize, RasError> {
    let alternate = match name.strip_prefix('@') {
        Some(bare) => bare.to_string(),
        None => format!("@{name}"),
    };
    if let Some(&offset) = offsets.get(name).or_else(|| offsets.get(&alternate)) {
        return Ok(offset);
    }
    let mut available: Vec<&str> = offsets.keys().map(String::as_str).collect();
    available.sort_unstable();
    Err(RasError::InvalidInput(format!(
        "JIT entry function '{name}' is not in the module (functions: {})",
        available.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_memory_allocate_and_write() {
        let mut mem = ExecutableMemory::allocate_writable(4096).expect("alloc failed");
        let code = [0x90u8, 0x90, 0x90, 0xc3];
        mem.write_code(&code).expect("write failed");
        mem.make_executable().expect("make_executable failed");
        assert!(!mem.code_start().is_null());
    }

    #[test]
    fn executable_memory_too_large_returns_error() {
        // Allocation rounds up to a page (4096 bytes). Use exactly one page so
        // the next allocation produces a 4096-byte region, then try to write 8192.
        let mem = ExecutableMemory::allocate_writable(4096).expect("alloc failed");
        let big = vec![0u8; 8192];
        let result = {
            let mut m = mem;
            m.write_code(&big)
        };
        assert!(result.is_err(), "writing beyond capacity should fail");
    }
}

/// Run JIT output on the **host** CPU (x86_64 or aarch64 only). Skips when the host OS string
/// is not mapped in `TargetOperatingSystem` (e.g. some embedded targets).
#[cfg(all(test, feature = "encoder"))]
mod jit_host_exec_tests {
    use std::str::FromStr;

    use lamina_mir::{
        AddressSpace,
        block::Block,
        function::{Function, Parameter, Signature},
        instruction::{
            AddressMode, FloatBinOp, FloatCmpOp, FloatUnOp, Immediate, Instruction, IntBinOp,
            IntCmpOp, IntOverflowOp, MemoryAttrs, Operand,
        },
        module::Module,
        register::{Register, VirtualReg},
        types::{MirType, ScalarType},
    };
    use lamina_platform::{
        PointerWidth, TargetArchitecture, TargetOperatingSystem, detect_host_architecture_only,
        detect_host_os,
    };

    use crate::{
        assembler::core::RasAssembler,
        error::RasError,
        runtime::{ExecutableMemory, RasRuntime},
    };

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

    fn exec_host_i32(module: &Module) -> Option<i32> {
        let (arch, os) = host_jit_pair()?;
        let mem = compile_to_executable(arch, os, module).ok()?;
        let f = unsafe { mem.entry_fn::<extern "C" fn() -> i32>() };
        Some(f())
    }

    fn exec_host_i64(module: &Module) -> Option<i64> {
        let (arch, os) = host_jit_pair()?;
        let mem = compile_to_executable(arch, os, module).ok()?;
        let f = unsafe { mem.entry_fn::<extern "C" fn() -> i64>() };
        Some(f())
    }

    fn exec_host_i64_binary(module: &Module, a: i64, b: i64) -> Option<i64> {
        let (arch, os) = host_jit_pair()?;
        let mem = compile_to_executable(arch, os, module).ok()?;
        let f = unsafe { mem.entry_fn::<extern "C" fn(i64, i64) -> i64>() };
        Some(f(a, b))
    }

    fn ii32(v: i32) -> Operand {
        Operand::Immediate(Immediate::I32(v))
    }

    fn ii64(v: i64) -> Operand {
        Operand::Immediate(Immediate::I64(v))
    }

    fn module_scalar_binop(scalar: ScalarType, op: IntBinOp, lhs: Operand, rhs: Operand) -> Module {
        let ty = MirType::Scalar(scalar);
        let out = Register::Virtual(VirtualReg::gpr(0));
        let sig = Signature::new("f").with_return(ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op,
            ty,
            dst: out.clone(),
            lhs,
            rhs,
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("host_scalar_binop");
        module.add_function(f);
        module
    }

    fn module_i64_binop_params(op: IntBinOp) -> Module {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("g")
            .with_params(vec![
                Parameter::new(a.clone(), i64_ty),
                Parameter::new(b.clone(), i64_ty),
            ])
            .with_return(i64_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op,
            ty: i64_ty,
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("host_i64_params");
        module.add_function(f);
        module
    }

    fn module_i32_cmp_eq(lhs: Operand, rhs: Operand) -> Module {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let out = Register::Virtual(VirtualReg::gpr(0));
        let sig = Signature::new("cmp").with_return(i32_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntCmp {
            op: IntCmpOp::Eq,
            ty: i32_ty,
            dst: out.clone(),
            lhs,
            rhs,
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("host_i32_cmp");
        module.add_function(f);
        module
    }

    #[test]
    fn jit_host_exec_i64_add_immediates() {
        let Some(got) = exec_host_i64(&module_scalar_binop(
            ScalarType::I64,
            IntBinOp::Add,
            ii64(10),
            ii64(32),
        )) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i64_sub_immediates() {
        let Some(got) = exec_host_i64(&module_scalar_binop(
            ScalarType::I64,
            IntBinOp::Sub,
            ii64(100),
            ii64(33),
        )) else {
            return;
        };
        assert_eq!(got, 67);
    }

    #[test]
    fn jit_host_exec_i64_mul_immediates() {
        let Some(got) = exec_host_i64(&module_scalar_binop(
            ScalarType::I64,
            IntBinOp::Mul,
            ii64(6),
            ii64(7),
        )) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i64_add_two_arguments() {
        let Some(got) = exec_host_i64_binary(&module_i64_binop_params(IntBinOp::Add), 7, 35) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i64_mul_two_arguments() {
        let Some(got) = exec_host_i64_binary(&module_i64_binop_params(IntBinOp::Mul), 6, 7) else {
            return;
        };
        assert_eq!(got, 42);
    }

    fn module_params_added(param_tys: &[ScalarType], lhs: u32, rhs: u32) -> Module {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let params = param_tys
            .iter()
            .enumerate()
            .map(|(i, scalar)| {
                Parameter::new(
                    Register::Virtual(VirtualReg::gpr(i as u32)),
                    MirType::Scalar(*scalar),
                )
            })
            .collect();
        let out = Register::Virtual(VirtualReg::gpr(param_tys.len() as u32));
        let sig = Signature::new("mixed")
            .with_params(params)
            .with_return(i64_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i64_ty,
            dst: out.clone(),
            lhs: Operand::Register(Register::Virtual(VirtualReg::gpr(lhs))),
            rhs: Operand::Register(Register::Virtual(VirtualReg::gpr(rhs))),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out)),
        });
        f.add_block(entry);
        let mut module = Module::new("host_mixed_params");
        module.add_function(f);
        module
    }

    fn module_with_body(
        name: &str,
        params: &[ScalarType],
        ret: ScalarType,
        body: impl FnOnce(&[Operand], Register) -> Instruction,
    ) -> Module {
        let param_regs: Vec<Register> = (0..params.len())
            .map(|i| Register::Virtual(VirtualReg::gpr(i as u32)))
            .collect();
        let out = Register::Virtual(VirtualReg::gpr(params.len() as u32));
        let sig = Signature::new(name)
            .with_params(
                param_regs
                    .iter()
                    .zip(params)
                    .map(|(r, t)| Parameter::new(r.clone(), MirType::Scalar(*t)))
                    .collect(),
            )
            .with_return(MirType::Scalar(ret));
        let operands: Vec<Operand> = param_regs.into_iter().map(Operand::Register).collect();
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(body(&operands, out.clone()));
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out)),
        });
        f.add_block(entry);
        let mut module = Module::new(name);
        module.add_function(f);
        module
    }

    fn jit_host_entry<F: Copy>(module: &Module) -> Option<(ExecutableMemory, F)> {
        let (arch, os) = host_jit_pair()?;
        let mem = compile_to_executable(arch, os, module).expect("compile");
        let f = unsafe { mem.entry_fn::<F>() };
        Some((mem, f))
    }

    #[test]
    fn jit_host_exec_f64_return_passes_through() {
        let module = module_with_body("f64_id", &[ScalarType::F64], ScalarType::F64, |p, out| {
            Instruction::Copy {
                ty: MirType::Scalar(ScalarType::F64),
                dst: out,
                src: p[0].clone(),
            }
        });
        let Some((_mem, f)) = jit_host_entry::<extern "C" fn(f64) -> f64>(&module) else {
            return;
        };
        assert_eq!(f(2.5), 2.5);
    }

    #[test]
    fn jit_host_exec_f64_arithmetic_on_params() {
        let ops = [
            (FloatBinOp::FAdd, 14.0),
            (FloatBinOp::FSub, 6.0),
            (FloatBinOp::FMul, 40.0),
            (FloatBinOp::FDiv, 2.5),
        ];
        for (op, expected) in ops {
            let module = module_with_body(
                "f64_bin",
                &[ScalarType::F64; 2],
                ScalarType::F64,
                |p, out| Instruction::FloatBinary {
                    op,
                    ty: MirType::Scalar(ScalarType::F64),
                    dst: out,
                    lhs: p[0].clone(),
                    rhs: p[1].clone(),
                },
            );
            let Some((_mem, f)) = jit_host_entry::<extern "C" fn(f64, f64) -> f64>(&module) else {
                return;
            };
            assert_eq!(f(10.0, 4.0), expected, "{op:?}");
        }
    }

    #[test]
    fn jit_host_exec_f32_add_on_params() {
        let module = module_with_body(
            "f32_add",
            &[ScalarType::F32; 2],
            ScalarType::F32,
            |p, out| Instruction::FloatBinary {
                op: FloatBinOp::FAdd,
                ty: MirType::Scalar(ScalarType::F32),
                dst: out,
                lhs: p[0].clone(),
                rhs: p[1].clone(),
            },
        );
        let Some((_mem, f)) = jit_host_entry::<extern "C" fn(f32, f32) -> f32>(&module) else {
            return;
        };
        assert_eq!(f(1.5, 0.25), 1.75);
    }

    #[test]
    fn jit_host_exec_f64_neg_and_sqrt() {
        let neg = module_with_body("f64_neg", &[ScalarType::F64], ScalarType::F64, |p, out| {
            Instruction::FloatUnary {
                op: FloatUnOp::FNeg,
                ty: MirType::Scalar(ScalarType::F64),
                dst: out,
                src: p[0].clone(),
            }
        });
        let sqrt = module_with_body("f64_sqrt", &[ScalarType::F64], ScalarType::F64, |p, out| {
            Instruction::FloatUnary {
                op: FloatUnOp::FSqrt,
                ty: MirType::Scalar(ScalarType::F64),
                dst: out,
                src: p[0].clone(),
            }
        });
        let Some((_m1, f)) = jit_host_entry::<extern "C" fn(f64) -> f64>(&neg) else {
            return;
        };
        assert_eq!(f(3.0), -3.0);
        let Some((_m2, g)) = jit_host_entry::<extern "C" fn(f64) -> f64>(&sqrt) else {
            return;
        };
        assert_eq!(g(16.0), 4.0);
    }

    #[test]
    fn jit_host_exec_f64_compare_is_false_on_nan_except_ne() {
        let cases: [(FloatCmpOp, f64, f64, i32); 7] = [
            (FloatCmpOp::Lt, 1.0, 2.0, 1),
            (FloatCmpOp::Le, 2.0, 2.0, 1),
            (FloatCmpOp::Gt, 3.0, 2.0, 1),
            (FloatCmpOp::Ge, 1.0, 2.0, 0),
            (FloatCmpOp::Eq, f64::NAN, f64::NAN, 0),
            (FloatCmpOp::Ne, f64::NAN, f64::NAN, 1),
            (FloatCmpOp::Lt, f64::NAN, 1.0, 0),
        ];
        for (op, a, b, expected) in cases {
            let module = module_with_body(
                "f64_cmp",
                &[ScalarType::F64; 2],
                ScalarType::I32,
                |p, out| Instruction::FloatCmp {
                    op,
                    ty: MirType::Scalar(ScalarType::F64),
                    dst: out,
                    lhs: p[0].clone(),
                    rhs: p[1].clone(),
                },
            );
            let Some((_mem, f)) = jit_host_entry::<extern "C" fn(f64, f64) -> i32>(&module) else {
                return;
            };
            assert_eq!(f(a, b), expected, "{op:?} {a} {b}");
        }
    }

    #[test]
    fn jit_host_exec_float_width_and_int_conversions() {
        let widen = module_with_body("widen", &[ScalarType::F32], ScalarType::F64, |p, out| {
            Instruction::FloatConvert {
                from: MirType::Scalar(ScalarType::F32),
                to: MirType::Scalar(ScalarType::F64),
                dst: out,
                src: p[0].clone(),
            }
        });
        let Some((_m1, f)) = jit_host_entry::<extern "C" fn(f32) -> f64>(&widen) else {
            return;
        };
        assert_eq!(f(1.5), 1.5);

        let to_float =
            module_with_body("to_float", &[ScalarType::I64], ScalarType::F64, |p, out| {
                Instruction::IntToFloat {
                    from: MirType::Scalar(ScalarType::I64),
                    to: MirType::Scalar(ScalarType::F64),
                    signed: true,
                    dst: out,
                    src: p[0].clone(),
                }
            });
        let Some((_m2, g)) = jit_host_entry::<extern "C" fn(i64) -> f64>(&to_float) else {
            return;
        };
        assert_eq!(g(-7), -7.0);

        let to_int = module_with_body("to_int", &[ScalarType::F64], ScalarType::I64, |p, out| {
            Instruction::FloatToInt {
                from: MirType::Scalar(ScalarType::F64),
                to: MirType::Scalar(ScalarType::I64),
                signed: true,
                dst: out,
                src: p[0].clone(),
            }
        });
        let Some((_m3, h)) = jit_host_entry::<extern "C" fn(f64) -> i64>(&to_int) else {
            return;
        };
        assert_eq!(h(-7.9), -7);
    }

    fn module_blocks(
        name: &str,
        params: &[ScalarType],
        blocks: Vec<(&str, Vec<Instruction>)>,
    ) -> Module {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let sig = Signature::new(name)
            .with_params(
                params
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        Parameter::new(
                            Register::Virtual(VirtualReg::gpr(i as u32)),
                            MirType::Scalar(*t),
                        )
                    })
                    .collect(),
            )
            .with_return(i64_ty);
        let mut f = Function::new(sig);
        for (label, instructions) in blocks {
            let mut block = Block::new(label);
            for inst in instructions {
                block.push(inst);
            }
            f.add_block(block);
        }
        let mut module = Module::new(name);
        module.add_function(f);
        module
    }

    fn vreg(id: u32) -> Register {
        Register::Virtual(VirtualReg::gpr(id))
    }

    fn reg_op(id: u32) -> Operand {
        Operand::Register(vreg(id))
    }

    #[test]
    fn jit_host_exec_branch_selects_a_block() {
        let module = module_blocks(
            "branch",
            &[ScalarType::I64],
            vec![
                (
                    "entry",
                    vec![
                        Instruction::IntCmp {
                            op: IntCmpOp::Eq,
                            ty: MirType::Scalar(ScalarType::I64),
                            dst: vreg(1),
                            lhs: reg_op(0),
                            rhs: ii64(0),
                        },
                        Instruction::Br {
                            cond: vreg(1),
                            true_target: "zero".into(),
                            false_target: "other".into(),
                        },
                    ],
                ),
                (
                    "zero",
                    vec![Instruction::Ret {
                        value: Some(ii64(10)),
                    }],
                ),
                (
                    "other",
                    vec![Instruction::Ret {
                        value: Some(ii64(20)),
                    }],
                ),
            ],
        );
        let Some((_mem, f)) = jit_host_entry::<extern "C" fn(i64) -> i64>(&module) else {
            return;
        };
        assert_eq!(f(0), 10);
        assert_eq!(f(5), 20);
    }

    #[test]
    fn jit_host_exec_loop_sums_up_to_its_bound() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let module = module_blocks(
            "sum_to",
            &[ScalarType::I64],
            vec![
                (
                    "entry",
                    vec![
                        Instruction::Copy {
                            ty: i64_ty,
                            dst: vreg(1),
                            src: ii64(0),
                        },
                        Instruction::Copy {
                            ty: i64_ty,
                            dst: vreg(2),
                            src: ii64(0),
                        },
                        Instruction::Jmp {
                            target: "loop".into(),
                        },
                    ],
                ),
                (
                    "loop",
                    vec![
                        Instruction::IntBinary {
                            op: IntBinOp::Add,
                            ty: i64_ty,
                            dst: vreg(2),
                            lhs: reg_op(2),
                            rhs: ii64(1),
                        },
                        Instruction::IntBinary {
                            op: IntBinOp::Add,
                            ty: i64_ty,
                            dst: vreg(1),
                            lhs: reg_op(1),
                            rhs: reg_op(2),
                        },
                        Instruction::IntCmp {
                            op: IntCmpOp::SLt,
                            ty: i64_ty,
                            dst: vreg(3),
                            lhs: reg_op(2),
                            rhs: reg_op(0),
                        },
                        Instruction::Br {
                            cond: vreg(3),
                            true_target: "loop".into(),
                            false_target: "done".into(),
                        },
                    ],
                ),
                (
                    "done",
                    vec![Instruction::Ret {
                        value: Some(reg_op(1)),
                    }],
                ),
            ],
        );
        let Some((_mem, f)) = jit_host_entry::<extern "C" fn(i64) -> i64>(&module) else {
            return;
        };
        assert_eq!(f(10), 55);
        assert_eq!(f(1), 1);
    }

    #[test]
    fn jit_host_exec_switch_dispatches_on_value() {
        let module = module_blocks(
            "dispatch",
            &[ScalarType::I64],
            vec![
                (
                    "entry",
                    vec![Instruction::Switch {
                        value: vreg(0),
                        cases: vec![(1, "one".into()), (2, "two".into())],
                        default: "other".into(),
                    }],
                ),
                (
                    "one",
                    vec![Instruction::Ret {
                        value: Some(ii64(10)),
                    }],
                ),
                (
                    "two",
                    vec![Instruction::Ret {
                        value: Some(ii64(20)),
                    }],
                ),
                (
                    "other",
                    vec![Instruction::Ret {
                        value: Some(ii64(30)),
                    }],
                ),
            ],
        );
        let Some((_mem, f)) = jit_host_entry::<extern "C" fn(i64) -> i64>(&module) else {
            return;
        };
        assert_eq!(f(1), 10);
        assert_eq!(f(2), 20);
        assert_eq!(f(7), 30);
    }

    #[test]
    fn jit_host_exec_float_to_i8_saturates() {
        let module = module_with_body("f2i8", &[ScalarType::F64], ScalarType::I64, |p, out| {
            Instruction::FloatToInt {
                from: MirType::Scalar(ScalarType::F64),
                to: MirType::Scalar(ScalarType::I8),
                signed: true,
                dst: out,
                src: p[0].clone(),
            }
        });
        let Some((_mem, f)) = jit_host_entry::<extern "C" fn(f64) -> i64>(&module) else {
            return;
        };
        assert_eq!(f(128.0), 127);
        assert_eq!(f(-128.9), -128);
        assert_eq!(f(2.5), 2);
        assert_eq!(f(1e30), 127);
        assert_eq!(f(-1e30), -128);
        assert_eq!(f(f64::NAN), 0);
    }

    #[test]
    fn jit_host_exec_float_to_i64_saturates() {
        let module = module_with_body("f2i64", &[ScalarType::F64], ScalarType::I64, |p, out| {
            Instruction::FloatToInt {
                from: MirType::Scalar(ScalarType::F64),
                to: MirType::Scalar(ScalarType::I64),
                signed: true,
                dst: out,
                src: p[0].clone(),
            }
        });
        let Some((_mem, f)) = jit_host_entry::<extern "C" fn(f64) -> i64>(&module) else {
            return;
        };
        assert_eq!(f(1e30), i64::MAX);
        assert_eq!(f(-1e30), i64::MIN);
        assert_eq!(f(f64::NAN), 0);
        assert_eq!(f(-7.9), -7);
    }

    fn overflow_module(
        op: IntOverflowOp,
        ty: ScalarType,
        signed: bool,
        params: &[ScalarType],
    ) -> Module {
        module_with_body("overflow", params, ScalarType::I64, |p, out| {
            Instruction::IntOverflow {
                op,
                ty: MirType::Scalar(ty),
                signed,
                dst: out,
                lhs: p[0].clone(),
                rhs: p[1].clone(),
            }
        })
    }

    #[test]
    fn jit_host_exec_signed_narrow_overflow_flags() {
        use IntOverflowOp::{Add, Mul, Sub};
        let i8_2 = [ScalarType::I8, ScalarType::I8];
        let m = overflow_module(Add, ScalarType::I8, true, &i8_2);
        let Some((_m1, f)) = jit_host_entry::<extern "C" fn(i8, i8) -> i64>(&m) else {
            return;
        };
        assert_eq!(f(100, 100), 1);
        assert_eq!(f(100, 27), 0);
        let m = overflow_module(Sub, ScalarType::I8, true, &i8_2);
        let Some((_m2, g)) = jit_host_entry::<extern "C" fn(i8, i8) -> i64>(&m) else {
            return;
        };
        assert_eq!(g(-128, 1), 1);
        assert_eq!(g(-100, 27), 0);
        let i32_2 = [ScalarType::I32, ScalarType::I32];
        let m = overflow_module(Mul, ScalarType::I32, true, &i32_2);
        let Some((_m3, h)) = jit_host_entry::<extern "C" fn(i32, i32) -> i64>(&m) else {
            return;
        };
        assert_eq!(h(65536, 65536), 1);
        assert_eq!(h(65536, 2), 0);
    }

    #[test]
    fn jit_host_exec_unsigned_overflow_flags() {
        use IntOverflowOp::{Add, Mul, Sub};
        let i8_2 = [ScalarType::I8, ScalarType::I8];
        let m = overflow_module(Add, ScalarType::I8, false, &i8_2);
        let Some((_m1, f)) = jit_host_entry::<extern "C" fn(u8, u8) -> i64>(&m) else {
            return;
        };
        assert_eq!(f(200, 100), 1);
        assert_eq!(f(200, 55), 0);
        let m = overflow_module(Sub, ScalarType::I8, false, &i8_2);
        let Some((_m2, g)) = jit_host_entry::<extern "C" fn(u8, u8) -> i64>(&m) else {
            return;
        };
        assert_eq!(g(1, 2), 1);
        assert_eq!(g(5, 2), 0);
        let i16_2 = [ScalarType::I16, ScalarType::I16];
        let m = overflow_module(Mul, ScalarType::I16, false, &i16_2);
        let Some((_m3, h)) = jit_host_entry::<extern "C" fn(u16, u16) -> i64>(&m) else {
            return;
        };
        assert_eq!(h(300, 300), 1);
        assert_eq!(h(255, 255), 0);
        let i64_2 = [ScalarType::I64, ScalarType::I64];
        let m = overflow_module(Add, ScalarType::I64, true, &i64_2);
        let Some((_m4, k)) = jit_host_entry::<extern "C" fn(i64, i64) -> i64>(&m) else {
            return;
        };
        assert_eq!(k(i64::MAX, 1), 1);
        assert_eq!(k(i64::MAX, -1), 0);
        let m = overflow_module(Mul, ScalarType::I64, false, &i64_2);
        let Some((_m5, u)) = jit_host_entry::<extern "C" fn(u64, u64) -> i64>(&m) else {
            return;
        };
        assert_eq!(u(1 << 32, 1 << 32), 1);
        assert_eq!(u(1 << 32, 3), 0);
    }

    fn global_object(
        name: &str,
        ty: ScalarType,
        bytes: Vec<u8>,
        mutable: bool,
    ) -> lamina_mir::Global {
        let mut g = lamina_mir::Global::new(name, MirType::Scalar(ty), PointerWidth::Bits64);
        g.size = bytes.len().max(8);
        g.mutable = mutable;
        g.initializer = Some(bytes);
        g
    }

    fn module_with_global(global: lamina_mir::Global, function: Function) -> Module {
        let mut module = Module::new("data");
        module.globals.insert(global.name.clone(), global);
        module.add_function(function);
        module
    }

    #[test]
    fn jit_host_exec_writable_global_stores_and_loads() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let f = function_with_body(
            "store_then_load",
            &[],
            ScalarType::I64,
            vec![
                Instruction::SymbolAddress {
                    dst: vreg(0),
                    symbol: "counter".into(),
                },
                Instruction::Store {
                    ty: i64_ty,
                    src: ii64(35),
                    addr: AddressMode::BaseOffset {
                        address_space: AddressSpace::default(),
                        base: vreg(0),
                        offset: 0,
                    },
                    attrs: MemoryAttrs::default(),
                },
                Instruction::Load {
                    ty: i64_ty,
                    dst: vreg(1),
                    addr: AddressMode::BaseOffset {
                        address_space: AddressSpace::default(),
                        base: vreg(0),
                        offset: 0,
                    },
                    attrs: MemoryAttrs::default(),
                },
                Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: i64_ty,
                    dst: vreg(2),
                    lhs: reg_op(1),
                    rhs: ii64(7),
                },
                Instruction::Ret {
                    value: Some(reg_op(2)),
                },
            ],
        );
        let module = module_with_global(
            global_object(
                "counter",
                ScalarType::I64,
                0i64.to_le_bytes().to_vec(),
                true,
            ),
            f,
        );
        let Some((_mem, g)) =
            jit_host_entry_named::<extern "C" fn() -> i64>(&module, "store_then_load")
        else {
            return;
        };
        assert_eq!(g(), 42);
    }

    #[test]
    fn jit_host_exec_readonly_string_is_read_from_the_image() {
        let f = function_with_body(
            "second_byte",
            &[],
            ScalarType::I8,
            vec![
                Instruction::SymbolAddress {
                    dst: vreg(0),
                    symbol: "msg".into(),
                },
                Instruction::Load {
                    ty: MirType::Scalar(ScalarType::I8),
                    dst: vreg(1),
                    addr: AddressMode::BaseOffset {
                        address_space: AddressSpace::default(),
                        base: vreg(0),
                        offset: 1,
                    },
                    attrs: MemoryAttrs::default(),
                },
                Instruction::Ret {
                    value: Some(reg_op(1)),
                },
            ],
        );
        let module = module_with_global(
            global_object("msg", ScalarType::I8, b"hi\0".to_vec(), false),
            f,
        );
        let Some((_mem, g)) = jit_host_entry_named::<extern "C" fn() -> i8>(&module, "second_byte")
        else {
            return;
        };
        assert_eq!(g(), b'i' as i8);
    }

    #[test]
    fn jit_host_exec_global_pointer_follows_relocation() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let ptr_ty = MirType::Scalar(ScalarType::Ptr);
        let f = function_with_body(
            "through_pointer",
            &[],
            ScalarType::I64,
            vec![
                Instruction::SymbolAddress {
                    dst: vreg(0),
                    symbol: "slot".into(),
                },
                Instruction::Load {
                    ty: ptr_ty,
                    dst: vreg(1),
                    addr: AddressMode::BaseOffset {
                        address_space: AddressSpace::default(),
                        base: vreg(0),
                        offset: 0,
                    },
                    attrs: MemoryAttrs::default(),
                },
                Instruction::Load {
                    ty: i64_ty,
                    dst: vreg(2),
                    addr: AddressMode::BaseOffset {
                        address_space: AddressSpace::default(),
                        base: vreg(1),
                        offset: 0,
                    },
                    attrs: MemoryAttrs::default(),
                },
                Instruction::Ret {
                    value: Some(reg_op(2)),
                },
            ],
        );
        let mut slot = global_object("slot", ScalarType::Ptr, vec![0; 8], true);
        slot.relocations.push(lamina_mir::GlobalRelocation {
            offset: 0,
            symbol: "answer".into(),
            addend: 0,
            size: 8,
        });
        let mut module = module_with_global(slot, f);
        module.globals.insert(
            "answer".into(),
            global_object(
                "answer",
                ScalarType::I64,
                42i64.to_le_bytes().to_vec(),
                false,
            ),
        );
        let Some((_mem, g)) =
            jit_host_entry_named::<extern "C" fn() -> i64>(&module, "through_pointer")
        else {
            return;
        };
        assert_eq!(g(), 42);
    }

    #[test]
    fn jit_host_exec_external_symbol_address_is_nonzero() {
        let f = function_with_body(
            "has_llabs",
            &[],
            ScalarType::I64,
            vec![
                Instruction::SymbolAddress {
                    dst: vreg(0),
                    symbol: "llabs".into(),
                },
                Instruction::IntCmp {
                    op: IntCmpOp::Ne,
                    ty: MirType::Scalar(ScalarType::I64),
                    dst: vreg(1),
                    lhs: reg_op(0),
                    rhs: ii64(0),
                },
                Instruction::Ret {
                    value: Some(reg_op(1)),
                },
            ],
        );
        let mut module = Module::new("symbols");
        module.add_function(f);
        let Some((_mem, g)) = jit_host_entry_named::<extern "C" fn() -> i64>(&module, "has_llabs")
        else {
            return;
        };
        assert_eq!(g(), 1);
    }

    #[test]
    fn jit_host_exec_stack_region_round_trips_words() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let at = |offset: i16| AddressMode::BaseOffset {
            address_space: AddressSpace::default(),
            base: vreg(0),
            offset,
        };
        let mut module = module_blocks(
            "stack_pair",
            &[],
            vec![(
                "entry",
                vec![
                    Instruction::StackAddr {
                        dst: vreg(0),
                        slot: VirtualReg::gpr(9),
                    },
                    Instruction::Store {
                        ty: i64_ty,
                        src: ii64(35),
                        addr: at(0),
                        attrs: MemoryAttrs::default(),
                    },
                    Instruction::Store {
                        ty: i64_ty,
                        src: ii64(7),
                        addr: at(8),
                        attrs: MemoryAttrs::default(),
                    },
                    Instruction::Load {
                        ty: i64_ty,
                        dst: vreg(1),
                        addr: at(0),
                        attrs: MemoryAttrs::default(),
                    },
                    Instruction::Load {
                        ty: i64_ty,
                        dst: vreg(2),
                        addr: at(8),
                        attrs: MemoryAttrs::default(),
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty: i64_ty,
                        dst: vreg(3),
                        lhs: reg_op(1),
                        rhs: reg_op(2),
                    },
                    Instruction::Ret {
                        value: Some(reg_op(3)),
                    },
                ],
            )],
        );
        for f in module.functions.values_mut() {
            f.stack_reservations.insert(VirtualReg::gpr(9), 1);
            f.stack_alignments.insert(VirtualReg::gpr(9), 16);
        }
        let Some((_mem, f)) = jit_host_entry::<extern "C" fn() -> i64>(&module) else {
            return;
        };
        assert_eq!(f(), 42);
    }

    fn jit_host_entry_named<F: Copy>(module: &Module, name: &str) -> Option<(ExecutableMemory, F)> {
        let (arch, os) = host_jit_pair()?;
        let (mem, offsets) = RasRuntime::new(arch, os)
            .compile_with_offsets(module)
            .expect("compile");
        let entry = mem.code_start() as usize + offsets[name];
        let f = unsafe { std::mem::transmute_copy::<usize, F>(&entry) };
        Some((mem, f))
    }

    fn function_with_body(
        name: &str,
        params: &[ScalarType],
        ret: ScalarType,
        body: Vec<Instruction>,
    ) -> Function {
        let sig = Signature::new(name)
            .with_params(
                params
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        Parameter::new(
                            Register::Virtual(VirtualReg::gpr(i as u32)),
                            MirType::Scalar(*t),
                        )
                    })
                    .collect(),
            )
            .with_return(MirType::Scalar(ret));
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        for inst in body {
            entry.push(inst);
        }
        f.add_block(entry);
        f
    }

    #[test]
    fn jit_host_exec_call_to_module_function() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let inc = function_with_body(
            "inc",
            &[ScalarType::I64],
            ScalarType::I64,
            vec![
                Instruction::IntBinary {
                    op: IntBinOp::Add,
                    ty: i64_ty,
                    dst: vreg(1),
                    lhs: reg_op(0),
                    rhs: ii64(1),
                },
                Instruction::Ret {
                    value: Some(reg_op(1)),
                },
            ],
        );
        let twice = function_with_body(
            "twice",
            &[ScalarType::I64],
            ScalarType::I64,
            vec![
                Instruction::Call {
                    name: "inc".into(),
                    args: vec![reg_op(0)],
                    ret: Some(vreg(1)),
                },
                Instruction::Call {
                    name: "inc".into(),
                    args: vec![reg_op(1)],
                    ret: Some(vreg(2)),
                },
                Instruction::Ret {
                    value: Some(reg_op(2)),
                },
            ],
        );
        let mut module = Module::new("calls");
        module.add_function(inc);
        module.add_function(twice);
        let Some((_mem, f)) = jit_host_entry_named::<extern "C" fn(i64) -> i64>(&module, "twice")
        else {
            return;
        };
        assert_eq!(f(40), 42);
    }

    #[test]
    fn jit_host_exec_call_to_float_module_function() {
        let f64_ty = MirType::Scalar(ScalarType::F64);
        let half = function_with_body(
            "half",
            &[ScalarType::F64],
            ScalarType::F64,
            vec![
                Instruction::FloatBinary {
                    op: FloatBinOp::FMul,
                    ty: f64_ty,
                    dst: vreg(1),
                    lhs: reg_op(0),
                    rhs: Operand::Immediate(Immediate::F64(0.5)),
                },
                Instruction::Ret {
                    value: Some(reg_op(1)),
                },
            ],
        );
        let quarter = function_with_body(
            "quarter",
            &[ScalarType::F64],
            ScalarType::F64,
            vec![
                Instruction::Call {
                    name: "half".into(),
                    args: vec![reg_op(0)],
                    ret: Some(vreg(1)),
                },
                Instruction::Call {
                    name: "half".into(),
                    args: vec![reg_op(1)],
                    ret: Some(vreg(2)),
                },
                Instruction::Ret {
                    value: Some(reg_op(2)),
                },
            ],
        );
        let mut module = Module::new("float_calls");
        module.add_function(half);
        module.add_function(quarter);
        let Some((_mem, f)) = jit_host_entry_named::<extern "C" fn(f64) -> f64>(&module, "quarter")
        else {
            return;
        };
        assert_eq!(f(10.0), 2.5);
    }

    #[test]
    fn jit_host_exec_call_passes_stack_arguments() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let eight = [ScalarType::I64; 8];
        let mut sum_body = Vec::new();
        let mut acc = ii64(0);
        for i in 0..8u32 {
            sum_body.push(Instruction::IntBinary {
                op: IntBinOp::Add,
                ty: i64_ty,
                dst: vreg(8 + i),
                lhs: acc.clone(),
                rhs: reg_op(i),
            });
            acc = reg_op(8 + i);
        }
        sum_body.push(Instruction::Ret { value: Some(acc) });
        let sum8 = function_with_body("sum8", &eight, ScalarType::I64, sum_body);
        let forward = function_with_body(
            "forward",
            &eight,
            ScalarType::I64,
            vec![
                Instruction::Call {
                    name: "sum8".into(),
                    args: (0..8).map(reg_op).collect(),
                    ret: Some(vreg(8)),
                },
                Instruction::Ret {
                    value: Some(reg_op(8)),
                },
            ],
        );
        let mut module = Module::new("stack_calls");
        module.add_function(sum8);
        module.add_function(forward);
        let Some((_mem, f)) = jit_host_entry_named::<
            extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
        >(&module, "forward") else {
            return;
        };
        assert_eq!(f(1, 2, 3, 4, 5, 6, 7, 8), 36);
    }

    #[test]
    fn jit_host_exec_call_to_libc_symbol() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let module = module_blocks(
            "libc_call",
            &[ScalarType::I64],
            vec![(
                "entry",
                vec![
                    Instruction::Call {
                        name: "llabs".into(),
                        args: vec![reg_op(0)],
                        ret: Some(vreg(1)),
                    },
                    Instruction::IntBinary {
                        op: IntBinOp::Add,
                        ty: i64_ty,
                        dst: vreg(2),
                        lhs: reg_op(1),
                        rhs: ii64(2),
                    },
                    Instruction::Ret {
                        value: Some(reg_op(2)),
                    },
                ],
            )],
        );
        let Some((_mem, f)) = jit_host_entry::<extern "C" fn(i64) -> i64>(&module) else {
            return;
        };
        assert_eq!(f(-40), 42);
    }

    #[test]
    fn jit_host_exec_ints_after_leading_float_param() {
        let Some((arch, os)) = host_jit_pair() else {
            return;
        };
        let module =
            module_params_added(&[ScalarType::F64, ScalarType::I64, ScalarType::I64], 1, 2);
        let mem = compile_to_executable(arch, os, &module).expect("compile");
        let f = unsafe { mem.entry_fn::<extern "C" fn(f64, i64, i64) -> i64>() };
        assert_eq!(f(0.5, 7, 35), 42);
    }

    #[test]
    fn jit_host_exec_ints_around_middle_float_param() {
        let Some((arch, os)) = host_jit_pair() else {
            return;
        };
        let module =
            module_params_added(&[ScalarType::I64, ScalarType::F64, ScalarType::I64], 0, 2);
        let mem = compile_to_executable(arch, os, &module).expect("compile");
        let f = unsafe { mem.entry_fn::<extern "C" fn(i64, f64, i64) -> i64>() };
        assert_eq!(f(7, 0.5, 35), 42);
    }

    #[test]
    fn jit_host_exec_i32_sub_immediates() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Sub,
            ii32(50),
            ii32(8),
        )) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i32_mul_immediates() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Mul,
            ii32(6),
            ii32(7),
        )) else {
            return;
        };
        assert_eq!(got, 42);
    }

    #[test]
    fn jit_host_exec_i32_and_or_xor_immediates() {
        let Some(a) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::And,
            ii32(0x0F),
            ii32(0x33),
        )) else {
            return;
        };
        assert_eq!(a, 3);
        let Some(o) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Or,
            ii32(8),
            ii32(2),
        )) else {
            return;
        };
        assert_eq!(o, 10);
        let Some(x) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Xor,
            ii32(15),
            ii32(5),
        )) else {
            return;
        };
        assert_eq!(x, 10);
    }

    #[test]
    fn jit_host_exec_i32_udiv_immediates() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::UDiv,
            ii32(10),
            ii32(3),
        )) else {
            return;
        };
        assert_eq!(got, 3);
    }

    #[test]
    fn jit_host_exec_i32_sdiv_negative_dividend() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::SDiv,
            ii32(-7),
            ii32(2),
        )) else {
            return;
        };
        assert_eq!(got, -3);
    }

    #[test]
    fn jit_host_exec_i32_urem_srem_immediates() {
        let Some(u) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::URem,
            ii32(10),
            ii32(3),
        )) else {
            return;
        };
        assert_eq!(u, 1);
        let Some(s) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::SRem,
            ii32(-7),
            ii32(3),
        )) else {
            return;
        };
        assert_eq!(s, -1);
    }

    #[test]
    fn jit_host_exec_i32_shl_immediates() {
        let Some(got) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::Shl,
            ii32(3),
            ii32(4),
        )) else {
            return;
        };
        assert_eq!(got, 48);
    }

    #[test]
    fn jit_host_exec_i32_lshr_ashr_immediates() {
        let Some(l) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::LShr,
            ii32(128),
            ii32(3),
        )) else {
            return;
        };
        assert_eq!(l, 16);
        let Some(a) = exec_host_i32(&module_scalar_binop(
            ScalarType::I32,
            IntBinOp::AShr,
            ii32(-16),
            ii32(2),
        )) else {
            return;
        };
        assert_eq!(a, -4);
    }

    #[test]
    fn jit_host_exec_i32_intcmp_eq_immediates() {
        let Some(eq) = exec_host_i32(&module_i32_cmp_eq(ii32(9), ii32(9))) else {
            return;
        };
        assert_eq!(eq, 1);
        let Some(ne) = exec_host_i32(&module_i32_cmp_eq(ii32(2), ii32(3))) else {
            return;
        };
        assert_eq!(ne, 0);
    }

    fn module_two_constants() -> Module {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let mut module = Module::new("host_two_functions");
        for (name, value) in [("first", 1), ("second", 2)] {
            let mut f = Function::new(Signature::new(name).with_return(i64_ty));
            let mut entry = Block::new("entry");
            entry.push(Instruction::Ret {
                value: Some(ii64(value)),
            });
            f.add_block(entry);
            module.add_function(f);
        }
        module
    }

    #[test]
    fn compile_function_enters_the_named_function() {
        let Some((arch, os)) = host_jit_pair() else {
            return;
        };
        let module = module_two_constants();
        let mut runtime = RasRuntime::new(arch, os);
        for (name, expected) in [("first", 1), ("second", 2), ("@second", 2)] {
            let f = runtime
                .compile_function::<i64>(&module, name)
                .expect("compile");
            assert_eq!(unsafe { f() }, expected, "entry {name}");
        }
    }

    #[test]
    fn compile_function_rejects_an_unknown_name() {
        let Some((arch, os)) = host_jit_pair() else {
            return;
        };
        let result =
            RasRuntime::new(arch, os).compile_function::<i64>(&module_two_constants(), "third");
        assert!(matches!(result, Err(RasError::InvalidInput(_))));
    }
}
