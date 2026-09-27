//! x86_64 architecture-specific code generation
//!
//! This module handles compilation of MIR to x86_64 binary machine code.

use crate::{assembler::core::RasAssembler, error::RasError};
use lamina_mir::{MirType, ScalarType};
use std::collections::HashMap;

/// Compile MIR to binary for x86_64
///
/// This reuses the instruction emission logic from mir_codegen/x86_64
/// but generates binary instead of assembly text.
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
        // Current x86_64 binary encoder is intentionally minimal.
        // Refuse complex control flow for now so we don't produce nonsense binaries.
        if func.blocks.len() != 1 {
            return Err(RasError::EncodingError(format!(
                "x86_64 JIT encoder currently supports only single-block functions; '{}' has {} block(s)",
                func_name,
                func.blocks.len()
            )));
        }

        for inst in &func.blocks[0].instructions {
            match inst {
                lamina_mir::Instruction::Jmp { .. }
                | lamina_mir::Instruction::Br { .. }
                | lamina_mir::Instruction::Switch { .. }
                | lamina_mir::Instruction::Call { .. }
                | lamina_mir::Instruction::TailCall { .. } => {
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

        // Collect register usage (reuse logic from mir_codegen)
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

        // Allocate stack slots (reuse logic from mir_codegen)
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

        let stack_size = stack_slots.len() * 8;
        let aligned_stack_size = (stack_size + 15) & !15;

        // Generate function prologue (binary encoded)
        let prologue = encode_prologue_x86_64(aligned_stack_size as u32)?;
        code.extend_from_slice(&prologue);

        let skip_epilogue_and_ret = matches!(
            func.blocks[0].instructions.last(),
            Some(lamina_mir::Instruction::Unreachable)
        );

        // Handle function parameters (reuse ABI logic)
        if !func.sig.params.is_empty() {
            let arg_regs = abi.arg_registers();
            for (index, param) in func.sig.params.iter().enumerate() {
                if let Register::Virtual(vreg) = &param.reg
                    && let Some(slot_off) = stack_slots.get(vreg)
                {
                    if index < arg_regs.len() {
                        // MOV from argument register to stack slot
                        let mov_bytes = encode_mov_reg_mem_x86_64(arg_regs[index], *slot_off)?;
                        code.extend_from_slice(&mov_bytes);
                    } else {
                        // Handle stack arguments
                        let caller_off = 16 + ((index - arg_regs.len()) as i32) * 8;
                        let mov1 = encode_mov_mem_reg_x86_64(caller_off, "rax")?;
                        code.extend_from_slice(&mov1);
                        let mov2 = encode_mov_reg_mem_x86_64("rax", *slot_off)?;
                        code.extend_from_slice(&mov2);
                    }
                }
            }
        }

        // Compile each block
        for block in &func.blocks {
            for inst in &block.instructions {
                let inst_bytes = encode_mir_instruction_x86_64(
                    inst,
                    &mut reg_alloc,
                    &stack_slots,
                    aligned_stack_size,
                    &func_name,
                    &func.sig,
                )?;
                code.extend_from_slice(&inst_bytes);
            }
        }

        if !skip_epilogue_and_ret {
            let epilogue = encode_epilogue_x86_64()?;
            code.extend_from_slice(&epilogue);
            code.push(0xC3);
        }
    }

    Ok((code, function_offsets))
}

/// Encode x86_64 prologue
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

/// Encode x86_64 epilogue
fn encode_epilogue_x86_64() -> Result<Vec<u8>, RasError> {
    let mut code = Vec::with_capacity(8);
    // mov rsp, rbp: 48 89 EC
    code.extend_from_slice(&[0x48, 0x89, 0xEC]);
    // pop rbp: 5D
    code.push(0x5D);
    Ok(code)
}

/// Encode MOV register to memory
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
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) => Err(RasError::EncodingError(
            "x86_64 JIT Load of floating-point type is not supported".into(),
        )),
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
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) => Err(RasError::EncodingError(
            "x86_64 JIT Store of floating-point type is not supported".into(),
        )),
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

/// Encode MOV memory to register
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

/// Parse register name to encoding (x86_64)
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

#[inline]
fn modrm(mod_bits: u8, reg: u8, rm: u8) -> u8 {
    ((mod_bits & 0b11) << 6) | ((reg & 0b111) << 3) | (rm & 0b111)
}

#[inline]
fn rex_w_r_b(w: bool, reg: u8, rm: u8) -> u8 {
    0x40 | ((w as u8) << 3) | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1)
}

/// Encode MIR instruction to binary (x86_64)
///
/// This reuses the instruction emission logic from mir_codegen/x86_64
/// but generates binary instead of assembly text.
#[cfg(feature = "encoder")]
fn encode_mir_instruction_x86_64(
    inst: &lamina_mir::Instruction,
    _reg_alloc: &mut lamina_codegen::x86_64::X64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    _stack_size: usize,
    _func_name: &str,
    func_sig: &lamina_mir::Signature,
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
                    Immediate::F32(_) | Immediate::F64(_) => {
                        return Err(RasError::EncodingError(
                            "float immediates not supported in x86_64 JIT encoder".into(),
                        ));
                    }
                };
                let reg = parse_register_x86_64(dst_reg)?;
                Ok(mov_imm64(reg, v))
            }
            Operand::Register(reg) => match reg {
                Register::Virtual(v) => load_vreg_to_reg(slots, v, dst_reg),
                Register::Physical(p) => {
                    // mov dst, phys
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
                if let Some(rt) = &func_sig.ret_ty {
                    code.extend_from_slice(&materialize_return_value_x86_64(stack_slots, v, rt)?);
                } else {
                    code.extend_from_slice(&materialize_operand_to_reg(stack_slots, v, "rax")?);
                }
            }
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
        lamina_mir::Instruction::Unreachable => Ok(vec![0x0F, 0x0B]),
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
