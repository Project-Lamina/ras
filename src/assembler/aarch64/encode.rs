//! AArch64 instruction encoding helpers used when lowering MIR.

use super::operand::materialize_operand;
use crate::assembler::core::RasAssembler;
use crate::encoder::aarch64::fixed::{enc_ldp_post_64, enc_stp_pre_64, encode_add_rrr};
use crate::encoder::aarch64::ldst_imm64::{
    AArch64ScalarLdKind, AArch64ScalarStKind, encode_ldr_imm64, encode_ldr_scalar,
    encode_str_imm64, encode_str_scalar,
};
use crate::error::RasError;
use lamina_mir::{Immediate, MirType, Operand, Register, ScalarType};

/// `SUB`/`ADD` sp in up to two immediates: the low 12 bits and the next 12 shifted.
fn adjust_sp(code: &mut Vec<u8>, size: usize, low: u32, high: u32) -> Result<(), RasError> {
    if size > 0xFF_FFFF {
        return Err(RasError::EncodingError(format!(
            "Stack size {} too large for AArch64 JIT frame",
            size
        )));
    }
    if size >> 12 != 0 {
        let inst = high | (((size >> 12) as u32) << 10);
        code.extend_from_slice(&inst.to_le_bytes());
    }
    if size & 0xFFF != 0 {
        let inst = low | (((size & 0xFFF) as u32) << 10);
        code.extend_from_slice(&inst.to_le_bytes());
    }
    Ok(())
}

pub(super) fn encode_prologue(stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();

    let stp = enc_stp_pre_64(29, 30, 31, -16)?;
    code.extend_from_slice(&stp.to_le_bytes());

    // add x29, sp, #0   (aka mov x29, sp)
    let mov_fp = 0x9100_03FDu32;
    code.extend_from_slice(&mov_fp.to_le_bytes());

    // The prologue already saved x29, x30 (16 bytes), maintaining alignment
    if stack_size > 0 {
        let aligned_size = (stack_size + 15) & !15;
        adjust_sp(&mut code, aligned_size, 0xD100_03FF, 0xD140_03FF)?;
    }

    Ok(code)
}

/// Must restore SP before LDP (undo the SUB sp, sp, #aligned_size from prologue)
pub(super) fn encode_epilogue(aligned_stack_size: usize) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();

    if aligned_stack_size > 0 {
        adjust_sp(&mut code, aligned_stack_size, 0x9100_03FF, 0x9140_03FF)?;
    }

    let ldp = enc_ldp_post_64(29, 30, 31, 16)?;
    code.extend_from_slice(&ldp.to_le_bytes());

    Ok(code)
}

pub(super) fn encode_str(src_reg: &str, base_reg: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let src = parse_register(src_reg)?;
    encode_str_imm64(src, base_reg, offset)
}

pub(super) fn encode_ldr(dst_reg: &str, base_reg: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let dst = parse_register(dst_reg)?;
    encode_ldr_imm64(dst, base_reg, offset)
}

pub(super) fn mir_scalar_ld_kind(
    ty: &lamina_mir::MirType,
) -> Result<AArch64ScalarLdKind, RasError> {
    match ty {
        MirType::Scalar(ScalarType::I1) => Ok(AArch64ScalarLdKind::I8U),
        MirType::Scalar(ScalarType::I8) => Ok(AArch64ScalarLdKind::I8S),
        MirType::Scalar(ScalarType::I16) => Ok(AArch64ScalarLdKind::I16S),
        MirType::Scalar(ScalarType::I32) => Ok(AArch64ScalarLdKind::I32S),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok(AArch64ScalarLdKind::I64),
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) => Err(RasError::EncodingError(
            "AArch64 JIT Load of floating-point type is not supported".into(),
        )),
        _ => Err(RasError::EncodingError(format!(
            "AArch64 JIT Load: unsupported MIR type {:?}",
            ty
        ))),
    }
}

pub(super) fn mir_scalar_st_kind(
    ty: &lamina_mir::MirType,
) -> Result<AArch64ScalarStKind, RasError> {
    match ty {
        MirType::Scalar(ScalarType::I1 | ScalarType::I8) => Ok(AArch64ScalarStKind::I8),
        MirType::Scalar(ScalarType::I16) => Ok(AArch64ScalarStKind::I16),
        MirType::Scalar(ScalarType::I32) => Ok(AArch64ScalarStKind::I32),
        MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => Ok(AArch64ScalarStKind::I64),
        MirType::Scalar(ScalarType::F32 | ScalarType::F64) => Err(RasError::EncodingError(
            "AArch64 JIT Store of floating-point type is not supported".into(),
        )),
        _ => Err(RasError::EncodingError(format!(
            "AArch64 JIT Store: unsupported MIR type {:?}",
            ty
        ))),
    }
}

pub(super) fn encode_ldr_typed(
    dst_reg: &str,
    base_reg: u8,
    offset: i32,
    ty: &lamina_mir::MirType,
) -> Result<Vec<u8>, RasError> {
    let dst = parse_register(dst_reg)?;
    let kind = mir_scalar_ld_kind(ty)?;
    encode_ldr_scalar(dst, base_reg, offset, kind)
}

pub(super) fn encode_str_typed(
    src_reg: &str,
    base_reg: u8,
    offset: i32,
    ty: &lamina_mir::MirType,
) -> Result<Vec<u8>, RasError> {
    let src = parse_register(src_reg)?;
    let kind = mir_scalar_st_kind(ty)?;
    encode_str_scalar(src, base_reg, offset, kind)
}

/// RET Xn = 0xD65F0000 | (n << 5)
pub(super) fn encode_ret(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD65F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

/// BR Xn = 0xD61F0000 | (n << 5)
#[allow(dead_code)]
pub(super) fn encode_br(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD61F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

/// BLR Xn = 0xD63F0000 | (n << 5)
pub(super) fn encode_blr(reg: u8) -> Result<Vec<u8>, RasError> {
    let instr: u32 = 0xD63F_0000 | ((reg as u32) << 5);
    Ok(instr.to_le_bytes().to_vec())
}

pub(super) fn encode_brk(imm16: u16) -> Vec<u8> {
    let w = 0xD420_0000u32 | (((imm16 as u32) & 0xFFFF) << 5);
    w.to_le_bytes().to_vec()
}

pub(super) fn encode_nop() -> Vec<u8> {
    0xD503_201Fu32.to_le_bytes().to_vec()
}

#[cfg(feature = "encoder")]
pub(super) fn apply_i32_offset_to_reg(
    assembler: &mut RasAssembler,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    stack_size: usize,
    code: &mut Vec<u8>,
    base_reg: u8,
    offset: i32,
) -> Result<(), RasError> {
    if offset == 0 {
        return Ok(());
    }
    let o = i64::from(offset);
    if o > 0 && o <= 4095 {
        let inst = 0x9100_0000u32
            | (((o as u32) & 0xFFF) << 10)
            | ((base_reg as u32) << 5)
            | (base_reg as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(());
    }
    if (-4096..0).contains(&o) {
        let imm = (-o) as u32;
        let inst =
            0xD100_0000u32 | ((imm & 0xFFF) << 10) | ((base_reg as u32) << 5) | (base_reg as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(());
    }
    let imm_reg_str = reg_alloc.alloc_scratch().unwrap_or("x14");
    let imm_reg = parse_register(imm_reg_str)?;
    materialize_operand(
        assembler,
        &Operand::Immediate(Immediate::I64(o)),
        imm_reg,
        stack_slots,
        reg_alloc,
        code,
        stack_size,
    )?;
    code.extend_from_slice(&encode_add_rrr(base_reg, base_reg, imm_reg));
    reg_alloc.free_scratch(imm_reg_str);
    Ok(())
}

/// SBFM Xd, Xn, #immr, #imms (64-bit); used for sign-extension of low (imms+1) bits.
pub(super) fn encode_sbfm64(rd: u8, rn: u8, immr: u8, imms: u8) -> Vec<u8> {
    let w = 0x9340_0000u32
        | ((immr as u32) << 16)
        | ((imms as u32) << 10)
        | ((rn as u32) << 5)
        | (rd as u32);
    w.to_le_bytes().to_vec()
}

pub(super) enum ShiftVKind {
    Lsl,
    Lsr,
    Asr,
}

/// LSLV / LSRV / ASRV Xd, Xn, Xm (64-bit variable shift; shift amount uses low 6 bits of Xm)
pub(super) fn encode_shiftv(which: ShiftVKind, rd: u8, rn: u8, rm: u8) -> Vec<u8> {
    let op = match which {
        ShiftVKind::Lsl => 0x9AC02000u32,
        ShiftVKind::Lsr => 0x9AC02400u32,
        ShiftVKind::Asr => 0x9AC02800u32,
    };
    let w = op | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32);
    w.to_le_bytes().to_vec()
}

pub(super) fn encode_shiftv_w(which: ShiftVKind, rd: u8, rn: u8, rm: u8) -> Vec<u8> {
    let op = match which {
        ShiftVKind::Lsl => 0x1AC0_2000u32,
        ShiftVKind::Lsr => 0x1AC0_2400u32,
        ShiftVKind::Asr => 0x1AC0_2800u32,
    };
    let w = op | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32);
    w.to_le_bytes().to_vec()
}

#[cfg(feature = "encoder")]
pub(super) fn load_base_gpr(
    base: &lamina_mir::Register,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
) -> Result<(u8, Option<&'static str>), RasError> {
    match base {
        Register::Virtual(vreg) => {
            let base_offset = stack_slots.get(vreg).ok_or_else(|| {
                RasError::EncodingError(format!("No stack slot for base register: {:?}", base))
            })?;
            let scratch = reg_alloc.alloc_scratch().unwrap_or("x11");
            code.extend_from_slice(&encode_ldr(scratch, 29, *base_offset)?);
            let rn = parse_register(scratch)?;
            Ok((rn, Some(scratch)))
        }
        Register::Physical(p) => {
            let rn = parse_register(p.name)?;
            Ok((rn, None))
        }
    }
}

#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_indexed_address(
    _assembler: &mut RasAssembler,
    base: &lamina_mir::Register,
    index: &lamina_mir::Register,
    log2_scale: u16,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    _stack_size: usize,
    out_addr_scratch: &'static str,
) -> Result<(), RasError> {
    fn mov_x_dst_src(dst: u8, src: u8) -> Vec<u8> {
        let w = 0xAA00_03E0u32 | ((src as u32) << 16) | (dst as u32);
        w.to_le_bytes().to_vec()
    }

    let base_s = reg_alloc.alloc_scratch().unwrap_or("x13");
    let idx_s = reg_alloc.alloc_scratch().unwrap_or("x14");
    let base_r = parse_register(base_s)?;
    let idx_r = parse_register(idx_s)?;
    let out_r = parse_register(out_addr_scratch)?;

    match base {
        Register::Virtual(v) => {
            let off = stack_slots.get(v).ok_or_else(|| {
                RasError::EncodingError("indexed address: missing base stack slot".into())
            })?;
            code.extend_from_slice(&encode_ldr(base_s, 29, *off)?);
        }
        Register::Physical(p) => {
            let br = parse_register(p.name)?;
            if br != base_r {
                code.extend_from_slice(&mov_x_dst_src(base_r, br));
            }
        }
    }

    match index {
        Register::Virtual(v) => {
            let off = stack_slots.get(v).ok_or_else(|| {
                RasError::EncodingError("indexed address: missing index stack slot".into())
            })?;
            code.extend_from_slice(&encode_ldr(idx_s, 29, *off)?);
        }
        Register::Physical(p) => {
            let ir = parse_register(p.name)?;
            if ir != idx_r {
                code.extend_from_slice(&mov_x_dst_src(idx_r, ir));
            }
        }
    }

    if log2_scale > 0 {
        let sh_s = reg_alloc.alloc_scratch().unwrap_or("x15");
        let sh_r = parse_register(sh_s)?;
        let movz = 0xD280_0000u32 | (((log2_scale as u32) & 0xFFFF) << 5) | (sh_r as u32);
        code.extend_from_slice(&movz.to_le_bytes());
        code.extend_from_slice(&encode_shiftv(ShiftVKind::Lsl, idx_r, idx_r, sh_r));
        reg_alloc.free_scratch(sh_s);
    }

    code.extend_from_slice(&encode_add_rrr(out_r, base_r, idx_r));

    reg_alloc.free_scratch(base_s);
    reg_alloc.free_scratch(idx_s);

    Ok(())
}

/// Store a scratch register in the slot of `dst`.
#[cfg(feature = "encoder")]
pub(super) fn store_to_slot(
    code: &mut Vec<u8>,
    src_reg: &str,
    dst: &Register,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
) -> Result<(), RasError> {
    let Register::Virtual(vreg) = dst else {
        return Err(RasError::EncodingError(
            "AArch64 JIT: physical destination not supported".into(),
        ));
    };
    let offset = stack_slots.get(vreg).ok_or_else(|| {
        RasError::EncodingError(format!("No stack slot for virtual register: {:?}", vreg))
    })?;
    code.extend_from_slice(&encode_str(src_reg, 29, *offset)?);
    Ok(())
}

/// UBFM Xd, Xn, #immr, #imms (64-bit); used for zero-extension of low (imms+1) bits.
pub(super) fn encode_ubfm64(rd: u8, rn: u8, immr: u8, imms: u8) -> Vec<u8> {
    let w = 0xD340_0000u32
        | ((immr as u32) << 16)
        | ((imms as u32) << 10)
        | ((rn as u32) << 5)
        | (rd as u32);
    w.to_le_bytes().to_vec()
}

/// Byte I/O builtins as raw Darwin syscalls, same sequences as the AOT backend.
/// Returns false when `name` is not one of them.
#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_macos_io(
    assembler: &mut RasAssembler,
    name: &str,
    args: &[Operand],
    ret: Option<&Register>,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    stack_size: usize,
) -> Result<bool, RasError> {
    const SUB_SP_16: u32 = 0xD100_43FF;
    const ADD_SP_16: u32 = 0x9100_43FF;
    const MOV_X0_0: u32 = 0xD280_0000;
    const MOV_X0_1: u32 = 0xD280_0020;
    const MOV_X1_SP: u32 = 0x9100_03E1;
    const MOV_X2_1: u32 = 0xD280_0022;
    const MOV_X16_READ: u32 = 0xD280_0070;
    const MOV_X16_WRITE: u32 = 0xD280_0090;
    const SVC_0: u32 = 0xD400_0001;
    const MOV_X9_NEG2: u32 = 0x9280_0029;
    const MOV_X9_NEG1: u32 = 0x9280_0009;
    const CSEL_X0_X9_X0_CS: u32 = 0x9A80_2120;
    const CSEL_X0_X9_X0_EQ: u32 = 0x9A80_0120;
    const CMP_X0_0: u32 = 0xF100_001F;
    const DMB_SY: u32 = 0xD503_3FBF;
    const STRB_W9_SP: u32 = 0x3900_03E9;
    const LDRB_W9_SP: u32 = 0x3940_03E9;
    const CSINV_X0_X0_XZR_CC: u32 = 0xDA9F_3000;
    const SUBS_X0_X0_1: u32 = 0xF100_0400;
    const CSEL_X9_X9_X0_EQ: u32 = 0x9A80_0129;
    // skips x16, svc and the five result words
    const CBZ_X2_OVER_READ: u32 = 0xB400_0102;

    fn words(code: &mut Vec<u8>, list: &[u32]) {
        for word in list {
            code.extend_from_slice(&word.to_le_bytes());
        }
    }

    let mut load = |op: &Operand, dst: u8, code: &mut Vec<u8>, reg_alloc: &mut _| {
        materialize_operand(assembler, op, dst, stack_slots, reg_alloc, code, stack_size)
    };

    match (name, args) {
        ("lamina.writebyte", [value]) => {
            words(code, &[SUB_SP_16]);
            load(value, 9, code, reg_alloc)?;
            words(
                code,
                &[
                    STRB_W9_SP,
                    MOV_X0_1,
                    MOV_X1_SP,
                    MOV_X2_1,
                    MOV_X16_WRITE,
                    SVC_0,
                    MOV_X9_NEG2,
                    CSEL_X0_X9_X0_CS,
                    DMB_SY,
                ],
            );
            if let Some(dst) = ret {
                store_to_slot(code, "x0", dst, stack_slots)?;
            }
            words(code, &[ADD_SP_16]);
        }
        ("lamina.readbyte", []) => {
            words(
                code,
                &[
                    SUB_SP_16,
                    MOV_X0_0,
                    MOV_X1_SP,
                    MOV_X2_1,
                    MOV_X16_READ,
                    SVC_0,
                    CSINV_X0_X0_XZR_CC,
                ],
            );
            if let Some(dst) = ret {
                words(code, &[LDRB_W9_SP, SUBS_X0_X0_1, CSEL_X9_X9_X0_EQ]);
                store_to_slot(code, "x9", dst, stack_slots)?;
            }
            words(code, &[ADD_SP_16]);
        }
        ("lamina.write", [pointer, length]) => {
            load(pointer, 1, code, reg_alloc)?;
            load(length, 2, code, reg_alloc)?;
            words(
                code,
                &[
                    MOV_X0_1,
                    MOV_X16_WRITE,
                    SVC_0,
                    MOV_X9_NEG2,
                    CSEL_X0_X9_X0_CS,
                ],
            );
            if let Some(dst) = ret {
                store_to_slot(code, "x0", dst, stack_slots)?;
            }
        }
        ("lamina.read", [pointer, length]) => {
            load(pointer, 1, code, reg_alloc)?;
            load(length, 2, code, reg_alloc)?;
            words(
                code,
                &[
                    MOV_X0_0,
                    CBZ_X2_OVER_READ,
                    MOV_X16_READ,
                    SVC_0,
                    MOV_X9_NEG2,
                    CSEL_X0_X9_X0_CS,
                    CMP_X0_0,
                    MOV_X9_NEG1,
                    CSEL_X0_X9_X0_EQ,
                ],
            );
            if let Some(dst) = ret {
                store_to_slot(code, "x0", dst, stack_slots)?;
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// This encoder passes every value as one register word. Refuse what the AOT
/// backend passes any other way, so the caller falls back instead of running bad code.
#[cfg(feature = "encoder")]
pub(super) fn reject_unmodelled_abi(func: &lamina_mir::Function) -> Result<(), RasError> {
    let name = &func.sig.name;
    let refuse = |what: &str| {
        Err(RasError::EncodingError(format!(
            "AArch64 JIT: @{name} has {what}, not supported"
        )))
    };
    if func.sig.params.iter().any(|p| p.aggregate.is_some()) {
        return refuse("an aggregate parameter");
    }
    if func.sig.ret_aggregate.is_some() || func.sig.sret_claimed {
        return refuse("an aggregate return");
    }
    if !func.variadic_wide_arguments.is_empty() || !func.variadic_wide_results.is_empty() {
        return refuse("a wide variadic value");
    }
    // A slot that is also a parameter or call result holds a value split over registers.
    let mut slots = std::collections::HashSet::new();
    for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
        if let lamina_mir::Instruction::StackAddr { slot, .. } = &inst.instr {
            slots.insert(*slot);
        }
    }
    let is_slot = |reg: &Register| matches!(reg, Register::Virtual(v) if slots.contains(v));
    if func.sig.params.iter().any(|p| is_slot(&p.reg)) {
        return refuse("a parameter held in memory");
    }
    for inst in func.blocks.iter().flat_map(|b| &b.instructions) {
        if let lamina_mir::Instruction::Call { ret: Some(ret), .. } = &inst.instr
            && is_slot(ret)
        {
            return refuse("a call result held in memory");
        }
    }
    Ok(())
}

pub(super) fn parse_register(reg: &str) -> Result<u8, RasError> {
    let reg = reg.trim_start_matches('%');
    match reg {
        "x0" | "w0" => Ok(0),
        "x1" | "w1" => Ok(1),
        "x2" | "w2" => Ok(2),
        "x3" | "w3" => Ok(3),
        "x4" | "w4" => Ok(4),
        "x5" | "w5" => Ok(5),
        "x6" | "w6" => Ok(6),
        "x7" | "w7" => Ok(7),
        "x8" | "w8" => Ok(8),
        "x9" | "w9" => Ok(9),
        "x10" | "w10" => Ok(10),
        "x11" | "w11" => Ok(11),
        "x12" | "w12" => Ok(12),
        "x13" | "w13" => Ok(13),
        "x14" | "w14" => Ok(14),
        "x15" | "w15" => Ok(15),
        "x16" | "w16" | "ip0" => Ok(16),
        "x17" | "w17" | "ip1" => Ok(17),
        "x18" | "w18" => Ok(18),
        "x19" | "w19" => Ok(19),
        "x20" | "w20" => Ok(20),
        "x21" | "w21" => Ok(21),
        "x22" | "w22" => Ok(22),
        "x23" | "w23" => Ok(23),
        "x24" | "w24" => Ok(24),
        "x25" | "w25" => Ok(25),
        "x26" | "w26" => Ok(26),
        "x27" | "w27" => Ok(27),
        "x28" | "w28" => Ok(28),
        "x29" | "w29" | "fp" => Ok(29),
        "x30" | "w30" | "lr" => Ok(30),
        "x31" | "w31" | "sp" | "xzr" | "wzr" => Ok(31),
        _ => Err(RasError::EncodingError(format!(
            "Unknown register: {}",
            reg
        ))),
    }
}
