//! x86_64 architecture-specific code generation
//!
//! This module handles compilation of MIR to x86_64 binary machine code.

use crate::{assembler::core::RasAssembler, error::RasError};
use lamina_mir::{MirType, ScalarType};
use lamina_platform::{PointerWidth, TargetArchitecture, TargetOperatingSystem};
use std::collections::{HashMap, HashSet};

const X86_64_POINTER: PointerWidth = TargetArchitecture::X86_64.pointer_width();

/// Compile MIR to binary for x86_64
#[cfg(feature = "encoder")]
pub fn compile_mir_x86_64_function(
    assembler: &mut RasAssembler,
    module: &lamina_mir::Module,
    function_name: Option<&str>,
) -> Result<(Vec<u8>, HashMap<String, usize>), RasError> {
    use lamina_codegen::x86_64::{X64RegAlloc, X86ABI, X86Frame};
    use lamina_mir::Register;

    let abi = X86ABI::new(assembler.target_os);
    let mut code = Vec::new();
    let mut function_offsets: HashMap<String, usize> = HashMap::new();
    let internal_names: HashSet<String> = module
        .functions
        .iter()
        .filter(|(_, f)| !f.blocks.is_empty())
        .map(|(name, _)| name.clone())
        .collect();
    let mut call_fixups: Vec<(usize, String)> = Vec::new();

    let functions_to_compile: Vec<(String, &lamina_mir::Function)> =
        if let Some(name) = function_name {
            module
                .functions
                .get(name)
                .map(|f| vec![(name.to_string(), f)])
                .unwrap_or_default()
        } else {
            // Deterministic order for offset stability across runs.
            let mut names: Vec<String> = module.functions.keys().cloned().collect();
            names.sort();
            names
                .into_iter()
                .filter_map(|n| module.functions.get(&n).map(|f| (n, f)))
                .collect()
        };

    let reserve_hint: usize = functions_to_compile
        .iter()
        .map(|(_, f)| {
            128usize.saturating_add(
                f.blocks
                    .iter()
                    .map(|b| b.instructions.len().saturating_mul(48))
                    .sum::<usize>(),
            )
        })
        .sum::<usize>()
        .max(512);
    code.reserve(reserve_hint);

    for (func_name, func) in functions_to_compile {
        if func.blocks.is_empty() {
            continue;
        }
        for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
            match &inst.instr {
                lamina_mir::Instruction::TailCall { .. } => {
                    return Err(RasError::EncodingError(format!(
                        "x86_64 JIT encoder does not yet support control-flow/calls in '{}': {:?}",
                        func_name, inst
                    )));
                }
                _ => {}
            }
        }

        function_offsets.insert(func_name.clone(), code.len());

        let mut reg_alloc = X64RegAlloc::new(assembler.target_os);
        let mut stack_slots: std::collections::HashMap<lamina_mir::VirtualReg, i32> =
            std::collections::HashMap::new();
        let mut def_regs: std::collections::HashSet<lamina_mir::VirtualReg> =
            std::collections::HashSet::new();
        let mut used_regs: std::collections::HashSet<lamina_mir::VirtualReg> =
            std::collections::HashSet::new();

        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(dst) = inst.def_reg()
                    && let Register::Virtual(vreg) = dst
                {
                    def_regs.insert(*vreg);
                }
                for reg in inst.use_regs() {
                    if let Register::Virtual(vreg) = reg {
                        used_regs.insert(*vreg);
                    }
                }
            }
        }

        for vreg in &def_regs {
            if !stack_slots.contains_key(vreg) {
                let slot_index = stack_slots.len();
                stack_slots.insert(*vreg, X86Frame::calculate_stack_offset(slot_index));
            }
        }
        for vreg in used_regs {
            if !def_regs.contains(&vreg) && !stack_slots.contains_key(&vreg) {
                let slot_index = stack_slots.len();
                stack_slots.insert(vreg, X86Frame::calculate_stack_offset(slot_index));
            }
        }

        // rbx is the second scratch register this encoder works in, and the C ABI
        // leaves it to the callee to put back. One slot below the locals holds the
        // caller's copy for as long as the function runs.
        let saved_rbx = X86Frame::calculate_stack_offset(stack_slots.len());
        let mut frame_bytes = (stack_slots.len() + 1) * 8;
        for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
            let lamina_mir::Instruction::StackAddr { slot, .. } = &inst.instr else {
                continue;
            };
            if stack_slots.contains_key(slot) {
                continue;
            }
            let words = 1 + func.stack_reservations.get(slot).copied().unwrap_or(0);
            let align = func.stack_alignments.get(slot).copied().unwrap_or(8);
            if align == 0 || align > 16 {
                return Err(RasError::EncodingError(format!(
                    "x86_64 JIT encoder: stack alignment {} in '{}' exceeds the 16-byte frame",
                    align, func_name
                )));
            }
            frame_bytes = (frame_bytes + words * 8).div_ceil(align) * align;
            stack_slots.insert(*slot, -(frame_bytes as i32));
        }
        let windows = matches!(assembler.target_os, TargetOperatingSystem::Windows);
        let mut outgoing_stack_args = 0usize;
        let mut makes_call = false;
        for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
            let lamina_mir::Instruction::Call { name, args, .. } = &inst.instr else {
                continue;
            };
            if is_lamina_io_intrinsic(name, args) {
                continue;
            }
            makes_call = true;
            let floats: Vec<bool> = args.iter().map(operand_is_float).collect();
            let stack_args = assign_arg_sources(&floats, abi.arg_registers(), windows, 0)
                .iter()
                .filter(|source| matches!(source, ArgSource::Stack(_)))
                .count();
            outgoing_stack_args = outgoing_stack_args.max(stack_args);
        }
        let shadow = if makes_call && windows { 32 } else { 0 };
        let outgoing_bytes = shadow + outgoing_stack_args * 8;
        let aligned_stack_size = (frame_bytes + outgoing_bytes + 15) & !15;
        let call_env = CallEnv {
            windows,
            gp_arg_regs: abi.arg_registers(),
            internal: &internal_names,
        };

        let prologue = encode_prologue_x86_64(aligned_stack_size as u32)?;
        code.extend_from_slice(&prologue);
        code.extend_from_slice(&encode_mov_reg_mem_x86_64("rbx", saved_rbx)?);

        let has_ret = func
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .any(|inst| matches!(inst.instr, lamina_mir::Instruction::Ret { .. }));

        let windows = matches!(assembler.target_os, TargetOperatingSystem::Windows);
        let float_params: Vec<bool> = func
            .sig
            .params
            .iter()
            .map(|p| matches!(p.ty, MirType::Scalar(ScalarType::F32 | ScalarType::F64)))
            .collect();
        let callee_stack_base = if windows { 48 } else { 16 };
        let sources = assign_arg_sources(
            &float_params,
            abi.arg_registers(),
            windows,
            callee_stack_base,
        );
        for (param, source) in func.sig.params.iter().zip(sources) {
            let double = matches!(param.ty, MirType::Scalar(ScalarType::F64));
            let Register::Virtual(vreg) = &param.reg else {
                continue;
            };
            let Some(slot_off) = stack_slots.get(vreg) else {
                continue;
            };
            match source {
                ArgSource::Gp(reg) => {
                    code.extend_from_slice(&encode_mov_reg_mem_x86_64(reg, *slot_off)?);
                }
                ArgSource::Xmm(xmm) => {
                    code.extend_from_slice(&encode_store_xmm_rbp_x86_64(xmm, *slot_off, double));
                }
                ArgSource::Stack(caller_off) => {
                    code.extend_from_slice(&encode_mov_mem_reg_x86_64(caller_off, "rax")?);
                    code.extend_from_slice(&encode_mov_reg_mem_x86_64("rax", *slot_off)?);
                }
            }
        }

        let mut labels: HashMap<String, usize> = HashMap::new();
        let mut fixups: Vec<(usize, String)> = Vec::new();
        for block in &func.blocks {
            labels.insert(block.label.clone(), code.len());
            for inst in &block.instructions {
                let mut local_fixups = Vec::new();
                let inst_bytes = encode_mir_instruction_x86_64(
                    inst,
                    &mut reg_alloc,
                    &stack_slots,
                    aligned_stack_size,
                    &func_name,
                    &func.sig,
                    &call_env,
                    &mut local_fixups,
                )?;
                let base = code.len();
                fixups.extend(
                    local_fixups
                        .into_iter()
                        .map(|(at, label)| (base + at, label)),
                );
                code.extend_from_slice(&inst_bytes);
            }
        }

        if has_ret {
            labels.insert(EXIT_LABEL.to_string(), code.len());
            code.extend_from_slice(&encode_mov_mem_reg_x86_64(saved_rbx, "rbx")?);
            let epilogue = encode_epilogue_x86_64()?;
            code.extend_from_slice(&epilogue);
            code.push(0xC3);
        }

        for (at, label) in fixups {
            if let Some(callee) = label.strip_prefix(CALL_LABEL_PREFIX) {
                call_fixups.push((at, callee.to_string()));
                continue;
            }
            let target = *labels.get(&label).ok_or_else(|| {
                RasError::EncodingError(format!(
                    "x86_64 JIT encoder: branch to unknown block '{}' in '{}'",
                    label, func_name
                ))
            })?;
            patch_rel32(&mut code, at, target);
        }
    }

    for (at, callee) in call_fixups {
        let target = *function_offsets.get(&callee).ok_or_else(|| {
            RasError::EncodingError(format!(
                "x86_64 JIT encoder: call to function '{}' not in the module",
                callee
            ))
        })?;
        patch_rel32(&mut code, at, target);
    }

    Ok((code, function_offsets))
}

/// Label of the shared epilogue that every `ret` jumps to.
const EXIT_LABEL: &str = "\u{0}exit";

/// Prefix of the fixup labels that name a function to call. Such fixups resolve
/// against the start of every function in the module, not the current one.
const CALL_LABEL_PREFIX: &str = "\u{1}call:";

struct CallEnv<'a> {
    windows: bool,
    gp_arg_regs: &'static [&'static str],
    internal: &'a HashSet<String>,
}

impl CallEnv<'_> {
    fn shadow_bytes(&self) -> i32 {
        if self.windows { 32 } else { 0 }
    }

    fn internal_name(&self, name: &str) -> Option<String> {
        [
            name.to_string(),
            name.trim_start_matches('@').to_string(),
            format!("@{}", name.trim_start_matches('@')),
        ]
        .into_iter()
        .find(|candidate| self.internal.contains(candidate))
    }
}

fn is_lamina_io_intrinsic(name: &str, args: &[lamina_mir::Operand]) -> bool {
    (name == "lamina.writebyte" && args.len() == 1)
        || (name == "lamina.readbyte" && args.is_empty())
        || ((name == "lamina.write" || name == "lamina.read") && args.len() == 2)
}

fn operand_is_float(op: &lamina_mir::Operand) -> bool {
    use lamina_mir::{Immediate, Operand, Register, RegisterClass};
    match op {
        Operand::Immediate(Immediate::F32(_) | Immediate::F64(_)) => true,
        Operand::Register(Register::Virtual(v)) => v.class == RegisterClass::Fpr,
        _ => false,
    }
}

/// Address of a symbol in the running process, when the process has one.
#[cfg(unix)]
fn resolve_host_symbol(name: &str) -> Option<usize> {
    let cname = std::ffi::CString::new(name).ok()?;
    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, cname.as_ptr()) };
    (!addr.is_null()).then_some(addr as usize)
}

#[cfg(not(unix))]
fn resolve_host_symbol(_name: &str) -> Option<usize> {
    None
}

fn encode_prologue_x86_64(stack_size: u32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::with_capacity(16);
    // push rbp: 55
    code.push(0x55);
    // mov rbp, rsp: 48 89 E5
    code.extend_from_slice(&[0x48, 0x89, 0xE5]);
    if stack_size > 0 {
        // Keep stack 16-byte aligned after `push rbp` (SysV/Microsoft x64 requirement).
        // Use imm8 when possible, else imm32.
        if stack_size <= 0x7f {
            // sub rsp, imm8 : 48 83 EC ib
            code.extend_from_slice(&[0x48, 0x83, 0xEC, stack_size as u8]);
        } else {
            // sub rsp, imm32 : 48 81 EC id
            code.extend_from_slice(&[0x48, 0x81, 0xEC]);
            code.extend_from_slice(&stack_size.to_le_bytes());
        }
    }
    Ok(code)
}

fn encode_epilogue_x86_64() -> Result<Vec<u8>, RasError> {
    let mut code = Vec::with_capacity(8);
    // mov rsp, rbp: 48 89 EC
    code.extend_from_slice(&[0x48, 0x89, 0xEC]);
    // pop rbp: 5D
    code.push(0x5D);
    Ok(code)
}

#[derive(Clone, Copy)]
enum ArgSource {
    Gp(&'static str),
    Xmm(u8),
    /// Displacement from the stack pointer on the caller's side, or from the
    /// frame base on the callee's, depending on which side assigned it.
    Stack(i32),
}

/// Places each argument in the register or stack slot the C ABI gives it. SysV hands
/// integers and floats out of separate register sequences; Windows counts positions.
fn assign_arg_sources(
    float_args: &[bool],
    gp_regs: &'static [&'static str],
    windows: bool,
    stack_base: i32,
) -> Vec<ArgSource> {
    let xmm_count = if windows { 4 } else { 8 };
    let mut next_gp = 0usize;
    let mut next_xmm = 0usize;
    let mut next_stack = 0i32;
    float_args
        .iter()
        .enumerate()
        .map(|(position, &is_float)| {
            let slot = |next: usize| if windows { position } else { next };
            let source = if is_float {
                if slot(next_xmm) < xmm_count {
                    next_xmm += 1;
                    Some(ArgSource::Xmm(slot(next_xmm - 1) as u8))
                } else {
                    None
                }
            } else if slot(next_gp) < gp_regs.len() {
                next_gp += 1;
                Some(ArgSource::Gp(gp_regs[slot(next_gp - 1)]))
            } else {
                None
            };
            source.unwrap_or_else(|| {
                let off = stack_base + next_stack * 8;
                next_stack += 1;
                ArgSource::Stack(off)
            })
        })
        .collect()
}

/// `movsd`/`movss [rbp+disp], xmm`: stores the low 64 or 32 bits of an argument register.
fn encode_store_xmm_rbp_x86_64(xmm: u8, disp: i32, double: bool) -> Vec<u8> {
    let mut code = vec![if double { 0xF2 } else { 0xF3 }];
    if xmm >= 8 {
        code.push(0x44);
    }
    code.extend_from_slice(&[0x0F, 0x11]);
    if (-128..=127).contains(&disp) {
        code.push(modrm(0b01, xmm, 0b101));
        code.push(disp as u8);
    } else {
        code.push(modrm(0b10, xmm, 0b101));
        code.extend_from_slice(&disp.to_le_bytes());
    }
    code
}

fn encode_mov_reg_mem_x86_64(src_reg: &str, offset: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    let src = parse_register_x86_64(src_reg)?;
    // MOV [rbp+disp], r64: REX.W + 89 /r
    let rex = rex_w_r_b(true, src, 5);
    code.push(rex);
    code.push(0x89);

    if (-128..=127).contains(&offset) {
        // mod=01, r/m=101 (rbp)
        code.push(modrm(0b01, src, 0b101));
        code.push(offset as u8);
    } else {
        // mod=10, r/m=101 (rbp), disp32
        code.push(modrm(0b10, src, 0b101));
        code.extend_from_slice(&offset.to_le_bytes());
    }
    Ok(code)
}

/// `mov dst64, [base64 + disp]` (Intel syntax). `base` and `dst` are 0..16 GPR encodings.
fn encode_mov_reg_membase_x86_64(dst: u8, base: u8, disp: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    let base_low = base & 7;
    let need_sib = base_low == 4;

    let (mod_bits, disp_bytes): (u8, Vec<u8>) = if base_low == 5 && disp == 0 {
        (0b01, vec![0])
    } else if disp == 0 {
        (0b00, vec![])
    } else if (-128..=127).contains(&disp) {
        (0b01, vec![disp as u8])
    } else {
        (0b10, disp.to_le_bytes().to_vec())
    };

    let rm = if need_sib { 0b100 } else { base_low };
    let rex = 0x48u8 | (((dst >> 3) & 1) << 2) | ((base >> 3) & 1);
    code.push(rex);
    code.push(0x8B);
    code.push(modrm(mod_bits, dst & 7, rm));
    if need_sib {
        code.push(0x24);
    }
    code.extend_from_slice(&disp_bytes);
    Ok(code)
}

/// `lea dst64, [base64 + disp]` (same ModRM/SIB rules as `mov` loads).
fn encode_lea_reg_membase_x86_64(dst: u8, base: u8, disp: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    let base_low = base & 7;
    let need_sib = base_low == 4;

    let (mod_bits, disp_bytes): (u8, Vec<u8>) = if base_low == 5 && disp == 0 {
        (0b01, vec![0])
    } else if disp == 0 {
        (0b00, vec![])
    } else if (-128..=127).contains(&disp) {
        (0b01, vec![disp as u8])
    } else {
        (0b10, disp.to_le_bytes().to_vec())
    };

    let rm = if need_sib { 0b100 } else { base_low };
    let rex = 0x48u8 | (((dst >> 3) & 1) << 2) | ((base >> 3) & 1);
    code.push(rex);
    code.push(0x8D);
    code.push(modrm(mod_bits, dst & 7, rm));
    if need_sib {
        code.push(0x24);
    }
    code.extend_from_slice(&disp_bytes);
    Ok(code)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum X86GpLoadFromMemKind {
    I64,
    I32S,
    I16S,
    I8S,
    I8Z,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum X86GpStoreToMemKind {
    I64,
    I32,
    I16,
    I8,
}

fn mir_ty_x86_gp_load_kind(ty: &lamina_mir::MirType) -> Result<X86GpLoadFromMemKind, RasError> {
    match ty {
        MirType::Scalar(ScalarType::I1) => Ok(X86GpLoadFromMemKind::I8Z),
        MirType::Scalar(ScalarType::I8) => Ok(X86GpLoadFromMemKind::I8S),
        MirType::Scalar(ScalarType::I16) => Ok(X86GpLoadFromMemKind::I16S),
        MirType::Scalar(ScalarType::I32) => Ok(X86GpLoadFromMemKind::I32S),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok(X86GpLoadFromMemKind::I64),
        MirType::Scalar(ScalarType::F32) => Ok(X86GpLoadFromMemKind::I32S),
        MirType::Scalar(ScalarType::F64) => Ok(X86GpLoadFromMemKind::I64),
        _ => Err(RasError::EncodingError(format!(
            "x86_64 JIT Load: unsupported MIR type {:?}",
            ty
        ))),
    }
}

fn mir_ty_x86_gp_store_kind(ty: &lamina_mir::MirType) -> Result<X86GpStoreToMemKind, RasError> {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => Ok(X86GpStoreToMemKind::I8),
        MirType::Scalar(ScalarType::I16) => Ok(X86GpStoreToMemKind::I16),
        MirType::Scalar(ScalarType::I32) => Ok(X86GpStoreToMemKind::I32),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok(X86GpStoreToMemKind::I64),
        MirType::Scalar(ScalarType::F32) => Ok(X86GpStoreToMemKind::I32),
        MirType::Scalar(ScalarType::F64) => Ok(X86GpStoreToMemKind::I64),
        _ => Err(RasError::EncodingError(format!(
            "x86_64 JIT Store: unsupported MIR type {:?}",
            ty
        ))),
    }
}

fn encode_gp_load_membase_x86_64(
    dst: u8,
    base: u8,
    disp: i32,
    kind: X86GpLoadFromMemKind,
) -> Result<Vec<u8>, RasError> {
    if kind == X86GpLoadFromMemKind::I64 {
        return encode_mov_reg_membase_x86_64(dst, base, disp);
    }
    let mut code = Vec::new();
    let base_low = base & 7;
    let need_sib = base_low == 4;

    let (mod_bits, disp_bytes): (u8, Vec<u8>) = if base_low == 5 && disp == 0 {
        (0b01, vec![0])
    } else if disp == 0 {
        (0b00, vec![])
    } else if (-128..=127).contains(&disp) {
        (0b01, vec![disp as u8])
    } else {
        (0b10, disp.to_le_bytes().to_vec())
    };

    let rm = if need_sib { 0b100 } else { base_low };
    let rex = 0x48u8 | (((dst >> 3) & 1) << 2) | ((base >> 3) & 1);
    code.push(rex);
    match kind {
        X86GpLoadFromMemKind::I64 => {
            unreachable!("I64 loads use encode_mov_reg_membase_x86_64 at function entry");
        }
        X86GpLoadFromMemKind::I32S => code.push(0x63),
        X86GpLoadFromMemKind::I8S => {
            code.push(0x0F);
            code.push(0xBE);
        }
        X86GpLoadFromMemKind::I8Z => {
            code.push(0x0F);
            code.push(0xB6);
        }
        X86GpLoadFromMemKind::I16S => {
            code.push(0x0F);
            code.push(0xBF);
        }
    }
    code.push(modrm(mod_bits, dst & 7, rm));
    if need_sib {
        code.push(0x24);
    }
    code.extend_from_slice(&disp_bytes);
    Ok(code)
}

fn encode_gp_store_membase_x86_64(
    src: u8,
    base: u8,
    disp: i32,
    kind: X86GpStoreToMemKind,
) -> Result<Vec<u8>, RasError> {
    if kind == X86GpStoreToMemKind::I64 {
        return encode_mov_membase_reg_x86_64(src, base, disp);
    }
    let mut code = Vec::new();
    let base_low = base & 7;
    let need_sib = base_low == 4;

    let (mod_bits, disp_bytes): (u8, Vec<u8>) = if base_low == 5 && disp == 0 {
        (0b01, vec![0])
    } else if disp == 0 {
        (0b00, vec![])
    } else if (-128..=127).contains(&disp) {
        (0b01, vec![disp as u8])
    } else {
        (0b10, disp.to_le_bytes().to_vec())
    };

    let rm = if need_sib { 0b100 } else { base_low };
    match kind {
        X86GpStoreToMemKind::I64 => {
            unreachable!("I64 stores use encode_mov_membase_reg_x86_64 at function entry");
        }
        X86GpStoreToMemKind::I32 => {
            if let Some(rex) = rex_optional_for_low_regs(src, base) {
                code.push(rex);
            }
            code.push(0x89);
        }
        X86GpStoreToMemKind::I16 => {
            code.push(0x66);
            if let Some(rex) = rex_optional_for_low_regs(src, base) {
                code.push(rex);
            }
            code.push(0x89);
        }
        X86GpStoreToMemKind::I8 => {
            if let Some(rex) = rex_for_mov_m8_membase(src, base) {
                code.push(rex);
            }
            code.push(0x88);
        }
    }
    code.push(modrm(mod_bits, src & 7, rm));
    if need_sib {
        code.push(0x24);
    }
    code.extend_from_slice(&disp_bytes);
    Ok(code)
}

/// `mov [base64 + disp], src64`
fn encode_mov_membase_reg_x86_64(src: u8, base: u8, disp: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    let base_low = base & 7;
    let need_sib = base_low == 4;

    let (mod_bits, disp_bytes): (u8, Vec<u8>) = if base_low == 5 && disp == 0 {
        (0b01, vec![0])
    } else if disp == 0 {
        (0b00, vec![])
    } else if (-128..=127).contains(&disp) {
        (0b01, vec![disp as u8])
    } else {
        (0b10, disp.to_le_bytes().to_vec())
    };

    let rm = if need_sib { 0b100 } else { base_low };
    let rex = 0x48u8 | (((src >> 3) & 1) << 2) | ((base >> 3) & 1);
    code.push(rex);
    code.push(0x89);
    code.push(modrm(mod_bits, src & 7, rm));
    if need_sib {
        code.push(0x24);
    }
    code.extend_from_slice(&disp_bytes);
    Ok(code)
}

fn encode_mov_mem_reg_x86_64(offset: i32, dst_reg: &str) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    let dst = parse_register_x86_64(dst_reg)?;
    // MOV r64, [rbp+disp]: REX.W + 8B /r
    let rex = rex_w_r_b(true, dst, 5);
    code.push(rex);
    code.push(0x8B);
    if (-128..=127).contains(&offset) {
        code.push(modrm(0b01, dst, 0b101));
        code.push(offset as u8);
    } else {
        code.push(modrm(0b10, dst, 0b101));
        code.extend_from_slice(&offset.to_le_bytes());
    }
    Ok(code)
}

fn rex_optional_for_low_regs(reg_field: u8, rm_field: u8) -> Option<u8> {
    let r = (reg_field >> 3) & 1;
    let b = (rm_field >> 3) & 1;
    if r == 0 && b == 0 {
        None
    } else {
        Some(0x40 | (r << 2) | b)
    }
}

/// REX for `MOV m8, r8` / `88 /r`: required for `spl`..`dil` (encodings 4..7) and for
/// high-byte GPR sources (`r8b`..`r15b`); `REX.B` extends ModRM r/m when base is `r13` etc.
fn rex_for_mov_m8_membase(src: u8, base: u8) -> Option<u8> {
    let rex_r = ((src >> 3) & 1) << 2;
    let rex_b = (base >> 3) & 1;
    let spl_to_dil = (4..8).contains(&src);
    if rex_r != 0 || rex_b != 0 || spl_to_dil {
        Some(0x40u8 | rex_r | rex_b)
    } else {
        None
    }
}

fn encode_mov_r32_mem_rbp(dst: u8, disp: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    if let Some(rex) = rex_optional_for_low_regs(dst, 5) {
        code.push(rex);
    }
    code.push(0x8B);
    if (-128..=127).contains(&disp) {
        code.push(modrm(0b01, dst & 7, 0b101));
        code.push(disp as u8);
    } else {
        code.push(modrm(0b10, dst & 7, 0b101));
        code.extend_from_slice(&disp.to_le_bytes());
    }
    Ok(code)
}

fn encode_mov_r32_reg_mem_rbp(src: u8, disp: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    if let Some(rex) = rex_optional_for_low_regs(src, 5) {
        code.push(rex);
    }
    code.push(0x89);
    if (-128..=127).contains(&disp) {
        code.push(modrm(0b01, src & 7, 0b101));
        code.push(disp as u8);
    } else {
        code.push(modrm(0b10, src & 7, 0b101));
        code.extend_from_slice(&disp.to_le_bytes());
    }
    Ok(code)
}

fn encode_mov_r16_reg_mem_rbp(src: u8, disp: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    code.push(0x66);
    if let Some(rex) = rex_optional_for_low_regs(src, 5) {
        code.push(rex);
    }
    code.push(0x89);
    if (-128..=127).contains(&disp) {
        code.push(modrm(0b01, src & 7, 0b101));
        code.push(disp as u8);
    } else {
        code.push(modrm(0b10, src & 7, 0b101));
        code.extend_from_slice(&disp.to_le_bytes());
    }
    Ok(code)
}

fn encode_mov_r8_reg_mem_rbp(src: u8, disp: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    if let Some(rex) = rex_for_mov_m8_membase(src, 5) {
        code.push(rex);
    }
    code.push(0x88);
    if (-128..=127).contains(&disp) {
        code.push(modrm(0b01, src & 7, 0b101));
        code.push(disp as u8);
    } else {
        code.push(modrm(0b10, src & 7, 0b101));
        code.extend_from_slice(&disp.to_le_bytes());
    }
    Ok(code)
}

fn encode_mov_rexw_secondary_mem_rbp(dst: u8, disp: i32, opc: u8) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    code.push(rex_w_r_b(true, dst, 5));
    code.push(0x0F);
    code.push(opc);
    if (-128..=127).contains(&disp) {
        code.push(modrm(0b01, dst & 7, 0b101));
        code.push(disp as u8);
    } else {
        code.push(modrm(0b10, dst & 7, 0b101));
        code.extend_from_slice(&disp.to_le_bytes());
    }
    Ok(code)
}

fn imm_as_i64_for_intcmp(imm: &lamina_mir::Immediate) -> Result<i64, RasError> {
    match imm {
        lamina_mir::Immediate::I8(v) => Ok(*v as i64),
        lamina_mir::Immediate::I16(v) => Ok(*v as i64),
        lamina_mir::Immediate::I32(v) => Ok(*v as i64),
        lamina_mir::Immediate::I64(v) => Ok(*v),
        lamina_mir::Immediate::F32(_) | lamina_mir::Immediate::F64(_) => Err(
            RasError::EncodingError("x86_64 JIT IntCmp: floating-point immediate".into()),
        ),
    }
}

fn narrow_imm_i64(bits: u8, raw: i64, signed_style: bool) -> i64 {
    if bits == 0 || bits >= 64 {
        return raw;
    }
    let bits_u = u32::from(bits);
    let mask = (1i64 << bits_u) - 1;
    let v = raw & mask;
    if signed_style {
        let sign_bit = 1i64 << (bits_u - 1);
        if (v & sign_bit) != 0 { v | !mask } else { v }
    } else {
        v
    }
}

#[cfg(feature = "encoder")]
/// Widens the low `size` bytes of rax to the full register, the way the assembly
/// backend's `extend_integer` does, so the two paths agree bit for bit.
fn encode_extend_rax(size: usize, signed: bool) -> Vec<u8> {
    match (size, signed) {
        (1, true) => vec![0x48, 0x0F, 0xBE, 0xC0],
        (2, true) => vec![0x48, 0x0F, 0xBF, 0xC0],
        (4, true) => vec![0x48, 0x63, 0xC0],
        (1, false) => vec![0x0F, 0xB6, 0xC0],
        (2, false) => vec![0x0F, 0xB7, 0xC0],
        (4, false) => vec![0x89, 0xC0],
        _ => Vec::new(),
    }
}

fn mov_imm64_reg_x86_64(reg: u8, imm: i64) -> Vec<u8> {
    let mut code = Vec::new();
    let rex = 0x48 | ((reg >> 3) & 1);
    code.push(rex);
    code.push(0xB8 + (reg & 0b111));
    code.extend_from_slice(&imm.to_le_bytes());
    code
}

#[cfg(feature = "encoder")]
fn materialize_int_binop_operand_x86_64(
    slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    ty: &lamina_mir::MirType,
    operand: &lamina_mir::Operand,
    dst_reg_enc: u8,
    unsigned_div_like: bool,
) -> Result<Vec<u8>, RasError> {
    use lamina_mir::{MirType, Operand, Register, ScalarType};

    let dst_name = if dst_reg_enc == 0 { "rax" } else { "rcx" };
    let signed_integral_extend = !unsigned_div_like;

    match operand {
        Operand::Register(Register::Physical(p)) => {
            let dst = parse_register_x86_64(dst_name)?;
            let src = parse_register_x86_64(p.name)?;
            if dst == src {
                Ok(Vec::new())
            } else {
                let rex = rex_w_r_b(true, src, dst);
                Ok(vec![rex, 0x89, modrm(0b11, src & 7, dst & 7)])
            }
        }
        Operand::Register(Register::Virtual(v)) => {
            let off = slots.get(v).copied().ok_or_else(|| {
                RasError::EncodingError(format!("Missing stack slot for vreg {:?}", v))
            })?;
            match ty {
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => {
                    encode_mov_mem_reg_x86_64(off, dst_name)
                }
                MirType::Scalar(ScalarType::I32) => encode_mov_r32_mem_rbp(dst_reg_enc, off),
                MirType::Scalar(ScalarType::I16) => {
                    let opc = if signed_integral_extend {
                        0xBFu8
                    } else {
                        0xB7u8
                    };
                    encode_mov_rexw_secondary_mem_rbp(dst_reg_enc, off, opc)
                }
                MirType::Scalar(ScalarType::I8) => {
                    let opc = if signed_integral_extend {
                        0xBEu8
                    } else {
                        0xB6u8
                    };
                    encode_mov_rexw_secondary_mem_rbp(dst_reg_enc, off, opc)
                }
                MirType::Scalar(ScalarType::I1) => {
                    encode_mov_rexw_secondary_mem_rbp(dst_reg_enc, off, 0xB6u8)
                }
                MirType::Scalar(ScalarType::F32 | ScalarType::F64) => Err(RasError::EncodingError(
                    "x86_64 JIT IntBinary: floating-point MIR type".into(),
                )),
                _ => Err(RasError::EncodingError(format!(
                    "x86_64 JIT IntBinary: unsupported type {:?}",
                    ty
                ))),
            }
        }
        Operand::Immediate(imm) => {
            let raw = imm_as_i64_for_intcmp(imm)?;
            let narrowed = match ty {
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => raw,
                MirType::Scalar(ScalarType::I32) => raw as i32 as i64,
                MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I1) => {
                    let bits: u8 = match ty {
                        MirType::Scalar(ScalarType::I8) => 8,
                        MirType::Scalar(ScalarType::I16) => 16,
                        MirType::Scalar(ScalarType::I1) => 1,
                        _ => 64,
                    };
                    narrow_imm_i64(bits, raw, signed_integral_extend)
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "x86_64 JIT IntBinary: unsupported type {:?}",
                        ty
                    )));
                }
            };
            let reg = parse_register_x86_64(dst_name)?;
            Ok(mov_imm64_reg_x86_64(reg, narrowed))
        }
    }
}

#[cfg(feature = "encoder")]
fn emit_int_binop_i32_alu_x86_64(op: lamina_mir::IntBinOp) -> Vec<u8> {
    use lamina_mir::IntBinOp;
    let mut code = Vec::new();
    match op {
        IntBinOp::Add => code.extend_from_slice(&[0x01, 0xC8]),
        IntBinOp::Sub => code.extend_from_slice(&[0x29, 0xC8]),
        IntBinOp::Mul => code.extend_from_slice(&[0x0F, 0xAF, 0xC1]),
        IntBinOp::UDiv => {
            code.extend_from_slice(&[0x31, 0xD2]);
            code.extend_from_slice(&[0xF7, 0xF1]);
        }
        IntBinOp::SDiv => {
            code.extend_from_slice(&[0x99]);
            code.extend_from_slice(&[0xF7, 0xF9]);
        }
        IntBinOp::URem => {
            code.extend_from_slice(&[0x31, 0xD2]);
            code.extend_from_slice(&[0xF7, 0xF1]);
            code.extend_from_slice(&[0x89, 0xD0]);
        }
        IntBinOp::SRem => {
            code.extend_from_slice(&[0x99]);
            code.extend_from_slice(&[0xF7, 0xF9]);
            code.extend_from_slice(&[0x89, 0xD0]);
        }
        IntBinOp::Shl => code.extend_from_slice(&[0xD3, 0xE0]),
        IntBinOp::LShr => code.extend_from_slice(&[0xD3, 0xE8]),
        IntBinOp::AShr => code.extend_from_slice(&[0xD3, 0xF8]),
        IntBinOp::And => code.extend_from_slice(&[0x21, 0xC8]),
        IntBinOp::Or => code.extend_from_slice(&[0x09, 0xC8]),
        IntBinOp::Xor => code.extend_from_slice(&[0x31, 0xC8]),
    }
    code
}

/// Place the return value in `rax` per the x86-64 SysV ABI (narrow integers sign- or zero-extended).
#[cfg(feature = "encoder")]
fn materialize_return_value_x86_64(
    slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    op: &lamina_mir::Operand,
    ret_ty: &lamina_mir::MirType,
) -> Result<Vec<u8>, RasError> {
    use lamina_mir::{Immediate, MirType, Operand, Register, ScalarType};

    match op {
        Operand::Register(Register::Virtual(vreg)) => {
            let off = slots.get(vreg).copied().ok_or_else(|| {
                RasError::EncodingError(format!("Missing stack slot for vreg {:?}", vreg))
            })?;
            let mut code = Vec::new();
            match ret_ty {
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => {
                    code.extend_from_slice(&encode_mov_mem_reg_x86_64(off, "rax")?);
                }
                MirType::Scalar(ScalarType::I32) => {
                    code.extend_from_slice(&encode_mov_r32_mem_rbp(0, off)?);
                    code.extend_from_slice(&[0x48, 0x63, 0xC0]);
                }
                MirType::Scalar(ScalarType::I16) => {
                    code.extend_from_slice(&encode_mov_rexw_secondary_mem_rbp(0, off, 0xBF)?);
                }
                MirType::Scalar(ScalarType::I8) => {
                    code.extend_from_slice(&encode_mov_rexw_secondary_mem_rbp(0, off, 0xBE)?);
                }
                MirType::Scalar(ScalarType::I1) => {
                    code.extend_from_slice(&encode_mov_rexw_secondary_mem_rbp(0, off, 0xB6)?);
                    code.extend_from_slice(&[0x48, 0x83, 0xE0, 0x01]);
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "x86_64 JIT: unsupported return type {:?}",
                        ret_ty
                    )));
                }
            }
            Ok(code)
        }
        Operand::Immediate(imm) => {
            let v = match imm {
                Immediate::I8(v) => *v as i64,
                Immediate::I16(v) => *v as i64,
                Immediate::I32(v) => *v as i64,
                Immediate::I64(v) => *v,
                Immediate::F32(_) | Immediate::F64(_) => {
                    return Err(RasError::EncodingError(
                        "x86_64 JIT: float immediates not supported for return".into(),
                    ));
                }
            };
            Ok(mov_imm64_reg_x86_64(0, v))
        }
        Operand::Register(Register::Physical(p)) => {
            let dst = parse_register_x86_64("rax")?;
            let src = parse_register_x86_64(p.name)?;
            if dst == src {
                Ok(Vec::new())
            } else {
                let rex = rex_w_r_b(true, src, dst);
                Ok(vec![rex, 0x89, modrm(0b11, src & 7, dst & 7)])
            }
        }
    }
}

fn store_int_binop_result_x86_64(
    slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    v: &lamina_mir::VirtualReg,
    ty: &lamina_mir::MirType,
) -> Result<Vec<u8>, RasError> {
    let off = slots
        .get(v)
        .copied()
        .ok_or_else(|| RasError::EncodingError(format!("Missing stack slot for vreg {:?}", v)))?;
    match ty {
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => encode_mov_reg_mem_x86_64("rax", off),
        MirType::Scalar(ScalarType::I32) => encode_mov_r32_reg_mem_rbp(0, off),
        MirType::Scalar(ScalarType::I16) => encode_mov_r16_reg_mem_rbp(0, off),
        MirType::Scalar(ScalarType::I8 | ScalarType::I1) => encode_mov_r8_reg_mem_rbp(0, off),
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) => Err(RasError::EncodingError(
            "x86_64 JIT IntBinary store: floating-point MIR type".into(),
        )),
        _ => Err(RasError::EncodingError(format!(
            "x86_64 JIT IntBinary store: unsupported type {:?}",
            ty
        ))),
    }
}

fn parse_register_x86_64(reg: &str) -> Result<u8, RasError> {
    let reg = reg.trim_start_matches('%');
    match reg {
        "rax" => Ok(0),
        "rcx" => Ok(1),
        "rdx" => Ok(2),
        "rbx" => Ok(3),
        "rsp" => Ok(4),
        "rbp" => Ok(5),
        "rsi" => Ok(6),
        "rdi" => Ok(7),
        "r8" => Ok(8),
        "r9" => Ok(9),
        "r10" => Ok(10),
        "r11" => Ok(11),
        "r12" => Ok(12),
        "r13" => Ok(13),
        "r14" => Ok(14),
        "r15" => Ok(15),
        _ => Err(RasError::EncodingError(format!(
            "Unknown register: {}",
            reg
        ))),
    }
}

fn patch_rel32(code: &mut [u8], at: usize, target: usize) {
    let rel = target as i64 - (at as i64 + 4);
    code[at..at + 4].copy_from_slice(&(rel as i32).to_le_bytes());
}

fn push_rel32(code: &mut Vec<u8>, fixups: &mut Vec<(usize, String)>, label: &str) {
    fixups.push((code.len(), label.to_string()));
    code.extend_from_slice(&[0, 0, 0, 0]);
}

fn push_jmp_rel32(code: &mut Vec<u8>, fixups: &mut Vec<(usize, String)>, label: &str) {
    code.push(0xE9);
    push_rel32(code, fixups, label);
}

fn float_is_double(ty: &lamina_mir::MirType) -> Result<bool, RasError> {
    match ty {
        MirType::Scalar(ScalarType::F64) => Ok(true),
        MirType::Scalar(ScalarType::F32) => Ok(false),
        _ => Err(RasError::EncodingError(format!(
            "x86_64 JIT encoder: expected a floating-point type, found {:?}",
            ty
        ))),
    }
}

/// `movq xmm, r64` (double) or `movd xmm, r32` (single).
fn encode_gp_to_xmm_x86_64(xmm: u8, gp: u8, double: bool) -> Vec<u8> {
    let mut code = vec![0x66];
    if double {
        code.push(rex_w_r_b(true, xmm, gp));
    } else if (xmm | gp) & 8 != 0 {
        code.push(rex_w_r_b(false, xmm, gp));
    }
    code.extend_from_slice(&[0x0F, 0x6E, modrm(0b11, xmm, gp)]);
    code
}

/// `movq r64, xmm` (double) or `movd r32, xmm` (single); single zero-extends into the GP register.
fn encode_xmm_to_gp_x86_64(gp: u8, xmm: u8, double: bool) -> Vec<u8> {
    let mut code = vec![0x66];
    if double {
        code.push(rex_w_r_b(true, xmm, gp));
    } else if (xmm | gp) & 8 != 0 {
        code.push(rex_w_r_b(false, xmm, gp));
    }
    code.extend_from_slice(&[0x0F, 0x7E, modrm(0b11, xmm, gp)]);
    code
}

/// Scalar SSE op `op xmm_dst, xmm_src` with the double (F2) or single (F3) prefix.
fn encode_sse_scalar_x86_64(opcode: u8, dst: u8, src: u8, double: bool) -> Vec<u8> {
    vec![
        if double { 0xF2 } else { 0xF3 },
        0x0F,
        opcode,
        modrm(0b11, dst, src),
    ]
}

#[inline]
fn modrm(mod_bits: u8, reg: u8, rm: u8) -> u8 {
    ((mod_bits & 0b11) << 6) | ((reg & 0b111) << 3) | (rm & 0b111)
}

#[inline]
fn rex_w_r_b(w: bool, reg: u8, rm: u8) -> u8 {
    0x40 | ((w as u8) << 3) | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1)
}

#[cfg(feature = "encoder")]
fn encode_mir_instruction_x86_64(
    inst: &lamina_mir::Instruction,
    _reg_alloc: &mut lamina_codegen::x86_64::X64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    _stack_size: usize,
    _func_name: &str,
    func_sig: &lamina_mir::Signature,
    calls: &CallEnv,
    fixups: &mut Vec<(usize, String)>,
) -> Result<Vec<u8>, RasError> {
    use lamina_mir::{
        AddressMode, Immediate, IntBinOp, IntCmpOp, MirType, Operand, Register, ScalarType,
    };

    fn vreg_slot(
        slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
        reg: &lamina_mir::VirtualReg,
    ) -> Result<i32, RasError> {
        slots.get(reg).copied().ok_or_else(|| {
            RasError::EncodingError(format!("Missing stack slot for vreg {:?}", reg))
        })
    }

    fn mov_imm64(reg: u8, imm: i64) -> Vec<u8> {
        mov_imm64_reg_x86_64(reg, imm)
    }

    fn load_vreg_to_reg(
        slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
        v: &lamina_mir::VirtualReg,
        dst: &str,
    ) -> Result<Vec<u8>, RasError> {
        let off = vreg_slot(slots, v)?;
        encode_mov_mem_reg_x86_64(off, dst)
    }

    fn store_reg_to_vreg(
        slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
        v: &lamina_mir::VirtualReg,
        src: &str,
    ) -> Result<Vec<u8>, RasError> {
        let off = vreg_slot(slots, v)?;
        encode_mov_reg_mem_x86_64(src, off)
    }

    fn materialize_reg_pointer(
        slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
        base: &Register,
        tmp: &str,
    ) -> Result<Vec<u8>, RasError> {
        match base {
            Register::Virtual(v) => load_vreg_to_reg(slots, v, tmp),
            Register::Physical(p) => {
                let t = parse_register_x86_64(tmp)?;
                let s = parse_register_x86_64(p.name)?;
                if t == s {
                    Ok(Vec::new())
                } else {
                    let rex = rex_w_r_b(true, s, t);
                    Ok(vec![rex, 0x89, modrm(0b11, s & 7, t & 7)])
                }
            }
        }
    }

    fn materialize_operand_to_reg(
        slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
        op: &Operand,
        dst_reg: &str,
    ) -> Result<Vec<u8>, RasError> {
        match op {
            Operand::Immediate(imm) => {
                let v = match imm {
                    Immediate::I8(v) => *v as i64,
                    Immediate::I16(v) => *v as i64,
                    Immediate::I32(v) => *v as i64,
                    Immediate::I64(v) => *v,
                    Immediate::F32(v) => i64::from(v.to_bits()),
                    Immediate::F64(v) => v.to_bits() as i64,
                };
                let reg = parse_register_x86_64(dst_reg)?;
                Ok(mov_imm64(reg, v))
            }
            Operand::Register(reg) => match reg {
                Register::Virtual(v) => load_vreg_to_reg(slots, v, dst_reg),
                Register::Physical(p) => {
                    let dst = parse_register_x86_64(dst_reg)?;
                    let src = parse_register_x86_64(p.name)?;
                    if dst == src {
                        Ok(Vec::new())
                    } else {
                        let rex = rex_w_r_b(true, src, dst);
                        Ok(vec![rex, 0x89, modrm(0b11, src, dst)])
                    }
                }
            },
        }
    }

    let mut code = Vec::with_capacity(64);

    match inst {
        lamina_mir::Instruction::Ret { value } => {
            if let Some(v) = value {
                if let Some(rt @ MirType::Scalar(ScalarType::F32 | ScalarType::F64)) =
                    &func_sig.ret_ty
                {
                    let double = matches!(rt, MirType::Scalar(ScalarType::F64));
                    code.extend_from_slice(&materialize_operand_to_reg(stack_slots, v, "rax")?);
                    code.extend_from_slice(&encode_gp_to_xmm_x86_64(0, 0, double));
                } else if let Some(rt) = &func_sig.ret_ty {
                    code.extend_from_slice(&materialize_return_value_x86_64(stack_slots, v, rt)?);
                } else {
                    code.extend_from_slice(&materialize_operand_to_reg(stack_slots, v, "rax")?);
                }
            }
            push_jmp_rel32(&mut code, fixups, EXIT_LABEL);
            Ok(code)
        }
        lamina_mir::Instruction::IntBinary {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            if let Register::Virtual(v) = dst {
                let unsigned_div_like = matches!(op, IntBinOp::UDiv | IntBinOp::URem);
                match ty {
                    MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => {
                        code.extend_from_slice(&materialize_operand_to_reg(
                            stack_slots,
                            lhs,
                            "rax",
                        )?);
                        code.extend_from_slice(&materialize_operand_to_reg(
                            stack_slots,
                            rhs,
                            "rcx",
                        )?);
                        let rex = 0x48u8;
                        match op {
                            IntBinOp::Add => code.extend_from_slice(&[rex, 0x01, 0xC8]),
                            IntBinOp::Sub => code.extend_from_slice(&[rex, 0x29, 0xC8]),
                            IntBinOp::Mul => code.extend_from_slice(&[rex, 0x0F, 0xAF, 0xC1]),
                            IntBinOp::UDiv => {
                                code.extend_from_slice(&[rex, 0x31, 0xD2]);
                                code.extend_from_slice(&[rex, 0xF7, 0xF1]);
                            }
                            IntBinOp::SDiv => {
                                code.extend_from_slice(&[rex, 0x99]);
                                code.extend_from_slice(&[rex, 0xF7, 0xF9]);
                            }
                            IntBinOp::URem => {
                                code.extend_from_slice(&[rex, 0x31, 0xD2]);
                                code.extend_from_slice(&[rex, 0xF7, 0xF1]);
                                code.extend_from_slice(&[rex, 0x89, 0xD0]);
                            }
                            IntBinOp::SRem => {
                                code.extend_from_slice(&[rex, 0x99]);
                                code.extend_from_slice(&[rex, 0xF7, 0xF9]);
                                code.extend_from_slice(&[rex, 0x89, 0xD0]);
                            }
                            IntBinOp::Shl => code.extend_from_slice(&[rex, 0xD3, 0xE0]),
                            IntBinOp::LShr => code.extend_from_slice(&[rex, 0xD3, 0xE8]),
                            IntBinOp::AShr => code.extend_from_slice(&[rex, 0xD3, 0xF8]),
                            IntBinOp::And => code.extend_from_slice(&[rex, 0x21, 0xC8]),
                            IntBinOp::Or => code.extend_from_slice(&[rex, 0x09, 0xC8]),
                            IntBinOp::Xor => code.extend_from_slice(&[rex, 0x31, 0xC8]),
                        }
                        code.extend_from_slice(&store_reg_to_vreg(stack_slots, v, "rax")?);
                    }
                    MirType::Scalar(
                        ScalarType::I32 | ScalarType::I16 | ScalarType::I8 | ScalarType::I1,
                    ) => {
                        code.extend_from_slice(&materialize_int_binop_operand_x86_64(
                            stack_slots,
                            ty,
                            lhs,
                            0,
                            unsigned_div_like,
                        )?);
                        code.extend_from_slice(&materialize_int_binop_operand_x86_64(
                            stack_slots,
                            ty,
                            rhs,
                            1,
                            unsigned_div_like,
                        )?);
                        code.extend_from_slice(&emit_int_binop_i32_alu_x86_64(*op));
                        if matches!(ty, MirType::Scalar(ScalarType::I1)) {
                            code.extend_from_slice(&[0x83, 0xE0, 0x01]);
                        }
                        code.extend_from_slice(&store_int_binop_result_x86_64(stack_slots, v, ty)?);
                    }
                    MirType::Scalar(ScalarType::F32 | ScalarType::F64) => {
                        return Err(RasError::EncodingError(
                            "x86_64 JIT IntBinary: floating-point MIR type".into(),
                        ));
                    }
                    _ => {
                        return Err(RasError::EncodingError(format!(
                            "x86_64 JIT IntBinary: unsupported type {:?}",
                            ty
                        )));
                    }
                }
                Ok(code)
            } else {
                Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst registers not supported".into(),
                ))
            }
        }
        lamina_mir::Instruction::StackAddr { dst, slot } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for StackAddr not supported".into(),
                ));
            };
            let disp = vreg_slot(stack_slots, slot)?;
            code.extend_from_slice(&encode_lea_reg_membase_x86_64(0, 5, disp)?);
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::Lea { dst, base, offset } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for Lea not supported".into(),
                ));
            };
            code.extend_from_slice(&materialize_reg_pointer(stack_slots, base, "rax")?);
            let b = parse_register_x86_64("rax")?;
            let d = parse_register_x86_64("rbx")?;
            code.extend_from_slice(&encode_lea_reg_membase_x86_64(d, b, *offset)?);
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rbx")?);
            Ok(code)
        }
        lamina_mir::Instruction::Select {
            dst,
            cond,
            true_val,
            false_val,
            ..
        } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for Select not supported".into(),
                ));
            };
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, false_val, "rax")?);
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, true_val, "rdx")?);
            code.extend_from_slice(&materialize_operand_to_reg(
                stack_slots,
                &Operand::Register(cond.clone()),
                "rcx",
            )?);
            code.extend_from_slice(&[0x48, 0x85, 0xC9]);
            code.extend_from_slice(&[0x48, 0x0F, 0x45, 0xC2]);
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::IntConvert {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for IntConvert not supported".into(),
                ));
            };
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, src, "rax")?);
            code.extend_from_slice(&encode_extend_rax(
                from.size_bytes(X86_64_POINTER)
                    .min(to.size_bytes(X86_64_POINTER)),
                *signed,
            ));
            if to.size_bytes(X86_64_POINTER) < 8 {
                code.extend_from_slice(&encode_extend_rax(to.size_bytes(X86_64_POINTER), true));
            }
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::Copy { dst, src, .. } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for Copy not supported".into(),
                ));
            };
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, src, "rax")?);
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::Unreachable => Ok(vec![0x0F, 0x0B]),
        lamina_mir::Instruction::Bitcast { dst, src, .. } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for Bitcast not supported".into(),
                ));
            };
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, src, "rax")?);
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::Call { name, args, ret } if is_lamina_io_intrinsic(name, args) => {
            if calls.windows {
                return Err(RasError::EncodingError(format!(
                    "x86_64 JIT encoder: {} is not available on Windows",
                    name
                )));
            }
            if name == "lamina.write" || name == "lamina.read" {
                let writing = name == "lamina.write";
                code.extend_from_slice(&materialize_operand_to_reg(stack_slots, &args[0], "rax")?);
                code.extend_from_slice(&[0x49, 0x89, 0xC2]);
                code.extend_from_slice(&materialize_operand_to_reg(stack_slots, &args[1], "rax")?);
                let fd = if writing { 1u32 } else { 0 };
                let mut transfer = vec![0x48, 0x89, 0xC2, 0x4C, 0x89, 0xD6, 0xBF];
                transfer.extend_from_slice(&fd.to_le_bytes());
                let syscall = if writing { 1u32 } else { 0 };
                transfer.push(0xB8);
                transfer.extend_from_slice(&syscall.to_le_bytes());
                transfer.extend_from_slice(&[0x0F, 0x05]);
                transfer.extend_from_slice(&[0x48, 0x85, 0xC0]);
                transfer.extend_from_slice(&[0x48, 0xC7, 0xC1, 0xFE, 0xFF, 0xFF, 0xFF]);
                transfer.extend_from_slice(&[0x48, 0x0F, 0x48, 0xC1]);
                if !writing {
                    transfer.extend_from_slice(&[0x48, 0x85, 0xC0]);
                    transfer.extend_from_slice(&[0x48, 0xC7, 0xC1, 0xFF, 0xFF, 0xFF, 0xFF]);
                    transfer.extend_from_slice(&[0x48, 0x0F, 0x44, 0xC1]);
                    code.extend_from_slice(&[0x48, 0x85, 0xC0, 0x0F, 0x84]);
                    code.extend_from_slice(&(transfer.len() as u32).to_le_bytes());
                }
                code.extend_from_slice(&transfer);
                if let Some(Register::Virtual(dst_v)) = ret {
                    code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
                }
                return Ok(code);
            }
            if name == "lamina.writebyte" {
                code.extend_from_slice(&materialize_operand_to_reg(stack_slots, &args[0], "rax")?);
                code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x10]);
                code.extend_from_slice(&[0x88, 0x04, 0x24]);
                code.extend_from_slice(&[0xB8, 1, 0, 0, 0]);
                code.extend_from_slice(&[0xBF, 1, 0, 0, 0]);
                code.extend_from_slice(&[0x48, 0x89, 0xE6]);
                code.extend_from_slice(&[0xBA, 1, 0, 0, 0]);
                code.extend_from_slice(&[0x0F, 0x05]);
                code.extend_from_slice(&[0x48, 0x85, 0xC0]);
                code.extend_from_slice(&[0x48, 0xC7, 0xC1, 0xFE, 0xFF, 0xFF, 0xFF]);
                code.extend_from_slice(&[0x48, 0x0F, 0x48, 0xC1]);
                code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x10]);
            } else {
                code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x10]);
                code.extend_from_slice(&[0xB8, 0, 0, 0, 0]);
                code.extend_from_slice(&[0xBF, 0, 0, 0, 0]);
                code.extend_from_slice(&[0x48, 0x89, 0xE6]);
                code.extend_from_slice(&[0xBA, 1, 0, 0, 0]);
                code.extend_from_slice(&[0x0F, 0x05]);
                code.extend_from_slice(&[0x0F, 0xB6, 0x0C, 0x24]);
                code.extend_from_slice(&[0x48, 0xC7, 0xC2, 0xFF, 0xFF, 0xFF, 0xFF]);
                code.extend_from_slice(&[0x48, 0x85, 0xC0]);
                code.extend_from_slice(&[0x48, 0x0F, 0x44, 0xCA]);
                code.extend_from_slice(&[0x48, 0xC7, 0xC2, 0xFE, 0xFF, 0xFF, 0xFF]);
                code.extend_from_slice(&[0x48, 0x83, 0xF8, 0x01]);
                code.extend_from_slice(&[0x48, 0x0F, 0x47, 0xCA]);
                code.extend_from_slice(&[0x48, 0x89, 0xC8]);
                code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x10]);
            }
            if let Some(Register::Virtual(dst_v)) = ret {
                code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            }
            Ok(code)
        }
        lamina_mir::Instruction::Call { name, args, ret } => {
            let floats: Vec<bool> = args.iter().map(operand_is_float).collect();
            let sources = assign_arg_sources(
                &floats,
                calls.gp_arg_regs,
                calls.windows,
                calls.shadow_bytes(),
            );
            for (arg, source) in args.iter().zip(&sources) {
                if let ArgSource::Stack(off) = source {
                    code.extend_from_slice(&materialize_operand_to_reg(stack_slots, arg, "rax")?);
                    code.extend_from_slice(&encode_mov_membase_reg_x86_64(0, 4, *off)?);
                }
            }
            for (arg, source) in args.iter().zip(&sources) {
                if let ArgSource::Xmm(xmm) = source {
                    code.extend_from_slice(&materialize_operand_to_reg(stack_slots, arg, "rax")?);
                    code.extend_from_slice(&encode_gp_to_xmm_x86_64(*xmm, 0, true));
                }
            }
            for (arg, source) in args.iter().zip(&sources) {
                if let ArgSource::Gp(reg) = source {
                    code.extend_from_slice(&materialize_operand_to_reg(stack_slots, arg, reg)?);
                }
            }
            let vector_args = sources
                .iter()
                .filter(|source| matches!(source, ArgSource::Xmm(_)))
                .count();
            code.push(0xB8);
            code.extend_from_slice(&(vector_args as u32).to_le_bytes());
            if let Some(callee) = calls.internal_name(name) {
                code.push(0xE8);
                push_rel32(&mut code, fixups, &format!("{CALL_LABEL_PREFIX}{callee}"));
            } else {
                let addr = resolve_host_symbol(name).ok_or_else(|| {
                    RasError::EncodingError(format!(
                        "x86_64 JIT encoder: cannot resolve external symbol '{}' in this process",
                        name
                    ))
                })?;
                code.extend_from_slice(&[0x49, 0xBB]);
                code.extend_from_slice(&(addr as u64).to_le_bytes());
                code.extend_from_slice(&[0x41, 0xFF, 0xD3]);
            }
            if let Some(Register::Virtual(dst_v)) = ret {
                if dst_v.class == lamina_mir::RegisterClass::Fpr {
                    code.extend_from_slice(&encode_xmm_to_gp_x86_64(0, 0, true));
                }
                code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            }
            Ok(code)
        }
        lamina_mir::Instruction::Jmp { target } => {
            push_jmp_rel32(&mut code, fixups, target);
            Ok(code)
        }
        lamina_mir::Instruction::Br {
            cond,
            true_target,
            false_target,
        } => {
            code.extend_from_slice(&materialize_operand_to_reg(
                stack_slots,
                &Operand::Register(cond.clone()),
                "rax",
            )?);
            code.extend_from_slice(&[0x84, 0xC0]);
            code.extend_from_slice(&[0x0F, 0x85]);
            push_rel32(&mut code, fixups, true_target);
            push_jmp_rel32(&mut code, fixups, false_target);
            Ok(code)
        }
        lamina_mir::Instruction::Switch {
            value,
            cases,
            default,
        } => {
            code.extend_from_slice(&materialize_operand_to_reg(
                stack_slots,
                &Operand::Register(value.clone()),
                "rax",
            )?);
            for (case, label) in cases {
                match i32::try_from(*case) {
                    Ok(imm) => {
                        code.extend_from_slice(&[0x48, 0x3D]);
                        code.extend_from_slice(&imm.to_le_bytes());
                    }
                    Err(_) => {
                        code.extend_from_slice(&mov_imm64(1, *case));
                        code.extend_from_slice(&[0x48, 0x39, 0xC8]);
                    }
                }
                code.extend_from_slice(&[0x0F, 0x84]);
                push_rel32(&mut code, fixups, label);
            }
            push_jmp_rel32(&mut code, fixups, default);
            Ok(code)
        }
        lamina_mir::Instruction::SafePoint => Ok(vec![0x0F, 0x1F, 0x40, 0x00]),
        lamina_mir::Instruction::Comment { .. }
        | lamina_mir::Instruction::StackMap { .. }
        | lamina_mir::Instruction::PatchPoint { .. } => Ok(Vec::new()),
        lamina_mir::Instruction::Load { dst, addr, ty, .. } => {
            use lamina_mir::{MirType, ScalarType};
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for Load not supported".into(),
                ));
            };
            let ld_kind = mir_ty_x86_gp_load_kind(ty)?;
            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    code.extend_from_slice(&materialize_reg_pointer(stack_slots, base, "rax")?);
                    let b = parse_register_x86_64("rax")?;
                    let d = parse_register_x86_64("rbx")?;
                    code.extend_from_slice(&encode_gp_load_membase_x86_64(
                        d,
                        b,
                        i32::from(*offset),
                        ld_kind,
                    )?);
                    if matches!(ty, MirType::Scalar(ScalarType::I1)) {
                        code.extend_from_slice(&[0x48, 0x83, 0xE3, 0x01]);
                    }
                    code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rbx")?);
                }
                AddressMode::BaseIndexScale {
                    base,
                    index,
                    scale,
                    offset,
                    ..
                } => {
                    let sh = match *scale {
                        1 => 0u8,
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        _ => {
                            return Err(RasError::EncodingError(format!(
                                "x86_64 JIT: unsupported scale {}",
                                scale
                            )));
                        }
                    };
                    code.extend_from_slice(&materialize_reg_pointer(stack_slots, base, "rax")?);
                    code.extend_from_slice(&materialize_reg_pointer(stack_slots, index, "rbx")?);
                    if sh > 0 {
                        code.extend_from_slice(&[0x48, 0xC1, 0xE3, sh]);
                    }
                    code.extend_from_slice(&[0x48, 0x01, 0xD8]);
                    let b = parse_register_x86_64("rax")?;
                    let d = parse_register_x86_64("rbx")?;
                    code.extend_from_slice(&encode_gp_load_membase_x86_64(
                        d,
                        b,
                        i32::from(*offset),
                        ld_kind,
                    )?);
                    if matches!(ty, MirType::Scalar(ScalarType::I1)) {
                        code.extend_from_slice(&[0x48, 0x83, 0xE3, 0x01]);
                    }
                    code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rbx")?);
                }
            }
            Ok(code)
        }
        lamina_mir::Instruction::Store { src, addr, ty, .. } => {
            let st_kind = mir_ty_x86_gp_store_kind(ty)?;
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, src, "rbx")?);
            match addr {
                AddressMode::BaseOffset { base, offset, .. } => {
                    code.extend_from_slice(&materialize_reg_pointer(stack_slots, base, "rax")?);
                    let b = parse_register_x86_64("rax")?;
                    let s = parse_register_x86_64("rbx")?;
                    code.extend_from_slice(&encode_gp_store_membase_x86_64(
                        s,
                        b,
                        i32::from(*offset),
                        st_kind,
                    )?);
                }
                AddressMode::BaseIndexScale {
                    base,
                    index,
                    scale,
                    offset,
                    ..
                } => {
                    let sh = match *scale {
                        1 => 0u8,
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        _ => {
                            return Err(RasError::EncodingError(format!(
                                "x86_64 JIT: unsupported scale {}",
                                scale
                            )));
                        }
                    };
                    code.extend_from_slice(&materialize_reg_pointer(stack_slots, base, "rax")?);
                    code.extend_from_slice(&materialize_reg_pointer(stack_slots, index, "rcx")?);
                    if sh > 0 {
                        code.extend_from_slice(&[0x48, 0xC1, 0xE1, sh]);
                    }
                    code.extend_from_slice(&[0x48, 0x01, 0xC8]);
                    let b = parse_register_x86_64("rax")?;
                    let s = parse_register_x86_64("rbx")?;
                    code.extend_from_slice(&encode_gp_store_membase_x86_64(
                        s,
                        b,
                        i32::from(*offset),
                        st_kind,
                    )?);
                }
            }
            Ok(code)
        }
        lamina_mir::Instruction::IntCmp {
            op,
            dst,
            lhs,
            rhs,
            ty,
        } => {
            let signed_rel = matches!(
                op,
                IntCmpOp::SLt | IntCmpOp::SLe | IntCmpOp::SGt | IntCmpOp::SGe
            );

            let emit_operand = |buf: &mut Vec<u8>,
                                operand: &Operand,
                                dst_enc: u8|
             -> Result<(), RasError> {
                match operand {
                    Operand::Register(Register::Virtual(v)) => {
                        let off = vreg_slot(stack_slots, v)?;
                        match ty {
                            MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => {
                                buf.extend_from_slice(&encode_mov_mem_reg_x86_64(
                                    off,
                                    if dst_enc == 0 { "rax" } else { "rcx" },
                                )?);
                            }
                            MirType::Scalar(ScalarType::I32) => {
                                buf.extend_from_slice(&encode_mov_r32_mem_rbp(dst_enc, off)?);
                            }
                            MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I1) => {
                                let bits: u8 = match ty {
                                    MirType::Scalar(ScalarType::I8) => 8,
                                    MirType::Scalar(ScalarType::I16) => 16,
                                    MirType::Scalar(ScalarType::I1) => 1,
                                    _ => {
                                        return Err(RasError::EncodingError(
                                            "x86_64 JIT IntCmp narrow-type mismatch".into(),
                                        ));
                                    }
                                };
                                let opc = if bits == 16 {
                                    if signed_rel { 0xBFu8 } else { 0xB7u8 }
                                } else if signed_rel {
                                    0xBEu8
                                } else {
                                    0xB6u8
                                };
                                buf.extend_from_slice(&encode_mov_rexw_secondary_mem_rbp(
                                    dst_enc, off, opc,
                                )?);
                            }
                            MirType::Scalar(ScalarType::F32 | ScalarType::F64) => {
                                return Err(RasError::EncodingError(
                                    "x86_64 JIT IntCmp: floating-point MIR type".into(),
                                ));
                            }
                            _ => {
                                return Err(RasError::EncodingError(format!(
                                    "x86_64 JIT IntCmp: unsupported type {:?}",
                                    ty
                                )));
                            }
                        }
                    }
                    Operand::Immediate(imm) => {
                        let raw = imm_as_i64_for_intcmp(imm)?;
                        let narrowed = match ty {
                            MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => raw,
                            MirType::Scalar(ScalarType::I32) => raw as i32 as i64,
                            MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I1) => {
                                let bits: u8 = match ty {
                                    MirType::Scalar(ScalarType::I8) => 8,
                                    MirType::Scalar(ScalarType::I16) => 16,
                                    MirType::Scalar(ScalarType::I1) => 1,
                                    _ => 64,
                                };
                                narrow_imm_i64(bits, raw, signed_rel)
                            }
                            _ => {
                                return Err(RasError::EncodingError(format!(
                                    "x86_64 JIT IntCmp: unsupported type {:?}",
                                    ty
                                )));
                            }
                        };
                        buf.extend_from_slice(&mov_imm64(dst_enc, narrowed));
                    }
                    Operand::Register(Register::Physical(_)) => {
                        return Err(RasError::EncodingError(
                            "x86_64 JIT IntCmp: physical register operand not supported".into(),
                        ));
                    }
                }
                Ok(())
            };

            match ty {
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => {
                    emit_operand(&mut code, lhs, 0)?;
                    emit_operand(&mut code, rhs, 1)?;
                    code.extend_from_slice(&[0x48, 0x39, 0xC8]);
                }
                MirType::Scalar(ScalarType::I32) => {
                    emit_operand(&mut code, lhs, 0)?;
                    emit_operand(&mut code, rhs, 1)?;
                    code.extend_from_slice(&[0x39, 0xC8]);
                }
                MirType::Scalar(ScalarType::I8 | ScalarType::I16 | ScalarType::I1) => {
                    emit_operand(&mut code, lhs, 0)?;
                    if matches!(ty, MirType::Scalar(ScalarType::I1)) && !signed_rel {
                        code.extend_from_slice(&[0x48, 0x83, 0xE0, 0x01]);
                    }
                    emit_operand(&mut code, rhs, 1)?;
                    if matches!(ty, MirType::Scalar(ScalarType::I1)) && !signed_rel {
                        code.extend_from_slice(&[0x48, 0x83, 0xE1, 0x01]);
                    }
                    code.extend_from_slice(&[0x48, 0x39, 0xC8]);
                }
                MirType::Scalar(ScalarType::F32 | ScalarType::F64) => {
                    return Err(RasError::EncodingError(
                        "x86_64 JIT IntCmp: floating-point MIR type".into(),
                    ));
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "x86_64 JIT IntCmp: unsupported type {:?}",
                        ty
                    )));
                }
            }

            let setcc = match op {
                IntCmpOp::Eq => 0x94,
                IntCmpOp::Ne => 0x95,
                IntCmpOp::SLt => 0x9C,
                IntCmpOp::SLe => 0x9E,
                IntCmpOp::SGt => 0x9F,
                IntCmpOp::SGe => 0x9D,
                IntCmpOp::ULt => 0x92,
                IntCmpOp::ULe => 0x96,
                IntCmpOp::UGt => 0x97,
                IntCmpOp::UGe => 0x93,
            };
            code.extend_from_slice(&[0x0F, setcc, 0xC0]);
            code.extend_from_slice(&[0x0F, 0xB6, 0xC0]);

            if let Register::Virtual(v) = dst {
                code.extend_from_slice(&store_reg_to_vreg(stack_slots, v, "rax")?);
                Ok(code)
            } else {
                Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst registers not supported".into(),
                ))
            }
        }
        lamina_mir::Instruction::FloatBinary {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for FloatBinary not supported".into(),
                ));
            };
            let double = float_is_double(ty)?;
            let opcode = match op {
                lamina_mir::FloatBinOp::FAdd => 0x58,
                lamina_mir::FloatBinOp::FSub => 0x5C,
                lamina_mir::FloatBinOp::FMul => 0x59,
                lamina_mir::FloatBinOp::FDiv => 0x5E,
            };
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, lhs, "rax")?);
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, rhs, "rcx")?);
            code.extend_from_slice(&encode_gp_to_xmm_x86_64(0, 0, double));
            code.extend_from_slice(&encode_gp_to_xmm_x86_64(1, 1, double));
            code.extend_from_slice(&encode_sse_scalar_x86_64(opcode, 0, 1, double));
            code.extend_from_slice(&encode_xmm_to_gp_x86_64(0, 0, double));
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::FloatUnary { op, ty, dst, src } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for FloatUnary not supported".into(),
                ));
            };
            let double = float_is_double(ty)?;
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, src, "rax")?);
            match op {
                lamina_mir::FloatUnOp::FNeg => {
                    if double {
                        code.extend_from_slice(&[0x48, 0x0F, 0xBA, 0xF8, 63]);
                    } else {
                        code.extend_from_slice(&[0x0F, 0xBA, 0xF8, 31]);
                    }
                }
                lamina_mir::FloatUnOp::FSqrt => {
                    code.extend_from_slice(&encode_gp_to_xmm_x86_64(0, 0, double));
                    code.extend_from_slice(&encode_sse_scalar_x86_64(0x51, 0, 0, double));
                    code.extend_from_slice(&encode_xmm_to_gp_x86_64(0, 0, double));
                }
            }
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::FloatCmp {
            op,
            ty,
            dst,
            lhs,
            rhs,
        } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for FloatCmp not supported".into(),
                ));
            };
            let double = float_is_double(ty)?;
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, lhs, "rax")?);
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, rhs, "rcx")?);
            code.extend_from_slice(&encode_gp_to_xmm_x86_64(0, 0, double));
            code.extend_from_slice(&encode_gp_to_xmm_x86_64(1, 1, double));
            // ucomis sets CF/ZF/PF, and an unordered (NaN) pair sets all three, so
            // seta/setae are false for NaN. Lt and Le compare rhs against lhs.
            let ucomis_modrm = match op {
                lamina_mir::FloatCmpOp::Gt | lamina_mir::FloatCmpOp::Ge => 0xC1u8,
                _ => 0xC8u8,
            };
            let ucomis = |code: &mut Vec<u8>| {
                if double {
                    code.push(0x66);
                }
                code.extend_from_slice(&[0x0F, 0x2E, ucomis_modrm]);
            };
            match op {
                lamina_mir::FloatCmpOp::Eq | lamina_mir::FloatCmpOp::Ne => {
                    ucomis(&mut code);
                    if matches!(op, lamina_mir::FloatCmpOp::Eq) {
                        code.extend_from_slice(&[0x0F, 0x94, 0xC0, 0x0F, 0x9B, 0xC1, 0x20, 0xC8]);
                    } else {
                        code.extend_from_slice(&[0x0F, 0x95, 0xC0, 0x0F, 0x9A, 0xC1, 0x08, 0xC8]);
                    }
                }
                _ => {
                    ucomis(&mut code);
                    let setcc =
                        if matches!(op, lamina_mir::FloatCmpOp::Gt | lamina_mir::FloatCmpOp::Lt) {
                            0x97
                        } else {
                            0x93
                        };
                    code.extend_from_slice(&[0x0F, setcc, 0xC0]);
                }
            }
            code.extend_from_slice(&[0x0F, 0xB6, 0xC0]);
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::FloatConvert { from, to, dst, src } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for FloatConvert not supported".into(),
                ));
            };
            let from_double = float_is_double(from)?;
            let to_double = float_is_double(to)?;
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, src, "rax")?);
            if from_double != to_double {
                code.extend_from_slice(&encode_gp_to_xmm_x86_64(0, 0, from_double));
                // cvtss2sd (F3 0F 5A) widens; cvtsd2ss (F2 0F 5A) narrows.
                code.extend_from_slice(&encode_sse_scalar_x86_64(0x5A, 0, 0, from_double));
                code.extend_from_slice(&encode_xmm_to_gp_x86_64(0, 0, to_double));
            }
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::IntToFloat {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for IntToFloat not supported".into(),
                ));
            };
            let to_double = float_is_double(to)?;
            let from_size = from.size_bytes(X86_64_POINTER);
            if from_size == 8 && !*signed {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: unsigned 64-bit to float not supported".into(),
                ));
            }
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, src, "rax")?);
            code.extend_from_slice(&encode_extend_rax(from_size, *signed));
            code.extend_from_slice(&[if to_double { 0xF2 } else { 0xF3 }, 0x48, 0x0F, 0x2A, 0xC0]);
            code.extend_from_slice(&encode_xmm_to_gp_x86_64(0, 0, to_double));
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        lamina_mir::Instruction::FloatToInt {
            from,
            to,
            signed,
            dst,
            src,
        } => {
            let Register::Virtual(dst_v) = dst else {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: physical dst for FloatToInt not supported".into(),
                ));
            };
            if !*signed {
                return Err(RasError::EncodingError(
                    "x86_64 JIT encoder: unsigned float to integer not supported".into(),
                ));
            }
            let from_double = float_is_double(from)?;
            code.extend_from_slice(&materialize_operand_to_reg(stack_slots, src, "rax")?);
            code.extend_from_slice(&encode_gp_to_xmm_x86_64(0, 0, from_double));
            // cvttsd2si / cvttss2si with a 64-bit destination, then saturate to the
            // target width the way the other backends do.
            code.extend_from_slice(&[
                if from_double { 0xF2 } else { 0xF3 },
                0x48,
                0x0F,
                0x2C,
                0xC0,
            ]);
            // The hardware gives 2^63 for NaN and for anything at or past 2^63, so a
            // value there becomes i64::MAX, and NaN becomes zero, before the clamp.
            let two_pow_63 = if from_double {
                0x43E0_0000_0000_0000i64
            } else {
                i64::from(0x5F00_0000u32)
            };
            code.extend_from_slice(&mov_imm64(2, i64::MAX));
            code.extend_from_slice(&mov_imm64(1, two_pow_63));
            code.extend_from_slice(&encode_gp_to_xmm_x86_64(1, 1, from_double));
            if from_double {
                code.push(0x66);
            }
            code.extend_from_slice(&[0x0F, 0x2E, 0xC1]);
            code.extend_from_slice(&[0x48, 0x0F, 0x43, 0xC2]);
            code.extend_from_slice(&[0xB9, 0, 0, 0, 0]);
            code.extend_from_slice(&[0x48, 0x0F, 0x4A, 0xC1]);
            let bits = to.size_bytes(X86_64_POINTER) * 8;
            if bits < 64 {
                let max = (1i64 << (bits - 1)) - 1;
                let min = -(1i64 << (bits - 1));
                for (bound, cmov) in [(max, 0x4Fu8), (min, 0x4C)] {
                    code.extend_from_slice(&[0x48, 0xC7, 0xC1]);
                    code.extend_from_slice(&(bound as i32).to_le_bytes());
                    code.extend_from_slice(&[0x48, 0x39, 0xC8, 0x48, 0x0F, cmov, 0xC1]);
                }
            }
            code.extend_from_slice(&store_reg_to_vreg(stack_slots, dst_v, "rax")?);
            Ok(code)
        }
        other => Err(RasError::EncodingError(format!(
            "x86_64 JIT encoder: unsupported instruction {:?}",
            other
        ))),
    }
}

#[cfg(all(test, feature = "encoder"))]
mod x86_64_jit_inst_tests {
    use super::*;
    use lamina_mir::{
        block::Block,
        function::{Function, Parameter, Signature},
        instruction::{Immediate, Instruction, IntBinOp, IntCmpOp, Operand},
        module::Module,
        register::{Register, VirtualReg},
        types::{MirType, ScalarType},
    };
    use lamina_platform::{TargetArchitecture, TargetOperatingSystem};

    fn subslice_present(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    /// The C ABI leaves rbx to the callee to put back, and this encoder works in it,
    /// so a caller that kept something there must find it again.
    #[test]
    fn a_compiled_function_puts_rbx_back() {
        let code = compile_single_i64_binop(IntBinOp::Add);
        // push rbp; mov rbp, rsp
        assert_eq!(&code[..4], &[0x55, 0x48, 0x89, 0xE5], "{code:02x?}");
        // The save comes first, before anything that could write rbx. A prologue is
        // eleven bytes at the most, which is `sub` with a 32-bit immediate.
        let at = code
            .windows(3)
            .position(|w| w == [0x48, 0x89, 0x5D])
            .expect("no mov %rbx, d(%rbp) anywhere");
        assert!(
            at <= 11,
            "rbx is saved at {at}, after the prologue: {code:02x?}"
        );
        let slot = code[at + 3];
        // The restore is the last thing before `mov %rsp, %rbp; pop %rbp; ret`, and it
        // reads back the slot the save wrote.
        let tail = [0x48, 0x8B, 0x5D, slot, 0x48, 0x89, 0xEC, 0x5D, 0xC3];
        assert_eq!(&code[code.len() - tail.len()..], &tail, "{code:02x?}");
    }

    fn compile_single_i32_binop(op: IntBinOp) -> Vec<u8> {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("t")
            .with_params(vec![
                Parameter::new(a.clone(), i32_ty),
                Parameter::new(b.clone(), i32_ty),
            ])
            .with_return(i32_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op,
            ty: i32_ty,
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("t");
        module.add_function(f);
        let mut asm = RasAssembler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .expect("assembler");
        compile_mir_x86_64_function(&mut asm, &module, None)
            .expect("compile")
            .0
    }

    fn compile_single_i64_binop(op: IntBinOp) -> Vec<u8> {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("t")
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
        let mut module = Module::new("t");
        module.add_function(f);
        let mut asm = RasAssembler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .expect("assembler");
        compile_mir_x86_64_function(&mut asm, &module, None)
            .expect("compile")
            .0
    }

    #[test]
    fn mov_r8_sil_to_rbp_disp_requires_mandatory_rex() {
        let b = encode_mov_r8_reg_mem_rbp(6, -8).expect("encode");
        assert_eq!(b.as_slice(), &[0x40, 0x88, 0x75, 0xF8]);
    }

    #[test]
    fn mov_r8_al_to_rbp_disp_has_no_rex() {
        let b = encode_mov_r8_reg_mem_rbp(0, -8).expect("encode");
        assert_eq!(b.as_slice(), &[0x88, 0x45, 0xF8]);
    }

    #[test]
    fn gp_store_i8_sil_base_rbp_includes_rex_before_88() {
        let b = encode_gp_store_membase_x86_64(6, 5, -8, X86GpStoreToMemKind::I8).expect("encode");
        assert!(
            b.len() >= 4 && b[0] == 0x40 && b[1] == 0x88,
            "expected REX + 0x88, got {:02x?}",
            b
        );
    }

    #[test]
    fn compile_mir_x86_64_lea_select_encodes() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let base_v = Register::Virtual(VirtualReg::gpr(0));
        let lea_v = Register::Virtual(VirtualReg::gpr(1));
        let sel_v = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("mix")
            .with_params(vec![Parameter::new(base_v.clone(), i64_ty)])
            .with_return(i64_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::Lea {
            dst: lea_v.clone(),
            base: base_v.clone(),
            offset: -8,
        });
        entry.push(Instruction::Select {
            ty: i64_ty,
            dst: sel_v.clone(),
            cond: lea_v.clone(),
            true_val: Operand::Immediate(Immediate::I64(1)),
            false_val: Operand::Immediate(Immediate::I64(0)),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(sel_v.clone())),
        });
        f.add_block(entry);

        let mut module = Module::new("x86_lea_sel");
        module.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_x86_64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() > 24,
            "expected prologue + lea + select + epilogue + ret, got {} bytes",
            code.len()
        );
        assert!(code.ends_with(&[0xC3]), "expected RET at end");
    }

    #[test]
    fn compile_mir_x86_64_unreachable_terminator_skips_ret() {
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let sig = Signature::new("ud2_fn").with_return(i64_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::Unreachable);
        f.add_block(entry);

        let mut module = Module::new("x86_ud2");
        module.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_x86_64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.ends_with(&[0x0F, 0x0B]),
            "expected trailing UD2, got last bytes {:?}",
            &code[code.len().saturating_sub(4)..]
        );
    }

    #[test]
    fn compile_mir_x86_64_intcmp_i32_and_comment_encodes() {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let i64_ty = MirType::Scalar(ScalarType::I64);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("cmp32")
            .with_params(vec![
                Parameter::new(a.clone(), i32_ty),
                Parameter::new(b.clone(), i32_ty),
            ])
            .with_return(i64_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::Comment {
            text: "marker".into(),
        });
        entry.push(Instruction::IntCmp {
            op: IntCmpOp::ULt,
            ty: i32_ty,
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);

        let mut module = Module::new("cmp32_x86");
        module.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_x86_64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() > 32,
            "expected prologue + cmp32 path + epilogue + ret, got {} bytes",
            code.len()
        );
        assert!(code.ends_with(&[0xC3]));
    }

    #[test]
    fn compile_mir_x86_64_int_binop_i32_add_encodes() {
        let i32_ty = MirType::Scalar(ScalarType::I32);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("add32")
            .with_params(vec![
                Parameter::new(a.clone(), i32_ty),
                Parameter::new(b.clone(), i32_ty),
            ])
            .with_return(i32_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i32_ty,
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);

        let mut module = Module::new("add32_x86");
        module.add_function(f);

        let mut asm = RasAssembler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_x86_64_function(&mut asm, &module, None).expect("compile");
        assert!(
            code.len() >= 32,
            "expected prologue + 32-bit add + store + epilogue + ret, got {} bytes",
            code.len()
        );
        assert!(code.ends_with(&[0xC3]));
    }

    #[test]
    fn jit_x86_i32_udiv_emits_xor_edx_and_div_ecx() {
        let code = compile_single_i32_binop(IntBinOp::UDiv);
        assert!(subslice_present(&code, &[0x31, 0xD2, 0xF7, 0xF1]));
    }

    #[test]
    fn jit_x86_i32_urem_emits_div_then_mov_eax_edx() {
        let code = compile_single_i32_binop(IntBinOp::URem);
        assert!(subslice_present(
            &code,
            &[0x31, 0xD2, 0xF7, 0xF1, 0x89, 0xD0]
        ));
    }

    #[test]
    fn jit_x86_i32_srem_emits_idiv_then_mov_eax_edx() {
        let code = compile_single_i32_binop(IntBinOp::SRem);
        assert!(subslice_present(&code, &[0x99, 0xF7, 0xF9, 0x89, 0xD0]));
    }

    #[test]
    fn jit_x86_i32_sdiv_emits_cdq_and_idiv_ecx() {
        let code = compile_single_i32_binop(IntBinOp::SDiv);
        assert!(subslice_present(&code, &[0x99, 0xF7, 0xF9]));
    }

    #[test]
    fn jit_x86_i32_shl_emits_shl_eax_cl() {
        let code = compile_single_i32_binop(IntBinOp::Shl);
        assert!(subslice_present(&code, &[0xD3, 0xE0]));
    }

    #[test]
    fn jit_x86_i32_lshr_emits_shr_eax_cl() {
        let code = compile_single_i32_binop(IntBinOp::LShr);
        assert!(subslice_present(&code, &[0xD3, 0xE8]));
    }

    #[test]
    fn jit_x86_i32_ashr_emits_sar_eax_cl() {
        let code = compile_single_i32_binop(IntBinOp::AShr);
        assert!(subslice_present(&code, &[0xD3, 0xF8]));
    }

    #[test]
    fn jit_x86_i64_sdiv_emits_rex_cdq_and_rex_idiv() {
        let code = compile_single_i64_binop(IntBinOp::SDiv);
        assert!(subslice_present(&code, &[0x48, 0x99, 0x48, 0xF7, 0xF9]));
    }

    #[test]
    fn jit_x86_i8_add_store_uses_mov_m8_no_rex_when_al() {
        let i8_ty = MirType::Scalar(ScalarType::I8);
        let a = Register::Virtual(VirtualReg::gpr(0));
        let b = Register::Virtual(VirtualReg::gpr(1));
        let out = Register::Virtual(VirtualReg::gpr(2));
        let sig = Signature::new("add8")
            .with_params(vec![
                Parameter::new(a.clone(), i8_ty),
                Parameter::new(b.clone(), i8_ty),
            ])
            .with_return(i8_ty);
        let mut f = Function::new(sig);
        let mut entry = Block::new("entry");
        entry.push(Instruction::IntBinary {
            op: IntBinOp::Add,
            ty: i8_ty,
            dst: out.clone(),
            lhs: Operand::Register(a.clone()),
            rhs: Operand::Register(b.clone()),
        });
        entry.push(Instruction::Ret {
            value: Some(Operand::Register(out.clone())),
        });
        f.add_block(entry);
        let mut module = Module::new("add8_x86");
        module.add_function(f);
        let mut asm = RasAssembler::new(TargetArchitecture::X86_64, TargetOperatingSystem::Linux)
            .expect("assembler");
        let (code, _) = compile_mir_x86_64_function(&mut asm, &module, None).expect("compile");
        assert!(
            subslice_present(&code, &[0x88, 0x45]) || subslice_present(&code, &[0x88, 0x85]),
            "expected 8-bit store 88 /rbp, al in {:?}",
            code
        );
    }
}
