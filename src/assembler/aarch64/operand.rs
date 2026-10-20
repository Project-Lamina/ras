//! Materialising MIR operands into AArch64 registers.

use super::encode::{encode_ldr, encode_ldr_typed, encode_sbfm64};
use crate::assembler::core::RasAssembler;
use crate::error::RasError;
use lamina_mir::{Immediate, MirType, Operand, Register, ScalarType};

pub(super) fn narrow_imm_i64(bits: u8, raw: i64, signed_style: bool) -> i64 {
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
pub(super) fn mov_imm_to_w(dst: u8, v: i32) -> Vec<u8> {
    let u = v as u32;
    let lo = u & 0xFFFF;
    let hi = (u >> 16) & 0xFFFF;
    let mut b = Vec::new();
    let movz = 0x5280_0000u32 | (lo << 5) | (dst as u32);
    b.extend_from_slice(&movz.to_le_bytes());
    if hi != 0 {
        let movk = 0x7280_0000u32 | (1u32 << 21) | (hi << 5) | (dst as u32);
        b.extend_from_slice(&movk.to_le_bytes());
    }
    b
}

/// Narrow integer scalar operands for W-width `IntBinary` (I32 / I16 / I8 / I1).
#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
pub(super) fn materialize_scalar_operand_int_binary(
    _assembler: &mut RasAssembler,
    op: &lamina_mir::Operand,
    dst_reg: u8,
    ty: &lamina_mir::MirType,
    unsigned_atom: bool,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    _reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    _stack_size: usize,
) -> Result<(), RasError> {
    let dst_w = format!("w{}", dst_reg);
    let signed_style = !unsigned_atom;

    match op {
        Operand::Immediate(imm) => {
            let raw = match imm {
                Immediate::I8(x) => *x as i64,
                Immediate::I16(x) => *x as i64,
                Immediate::I32(x) => *x as i64,
                Immediate::I64(x) => *x,
                Immediate::F32(_) | Immediate::F64(_) => {
                    return Err(RasError::EncodingError(
                        "AArch64 JIT IntBinary: floating-point immediate".into(),
                    ));
                }
            };
            let v: i32 = match ty {
                MirType::Scalar(ScalarType::I32) => raw as i32,
                MirType::Scalar(ScalarType::I16) => {
                    narrow_imm_i64(16, raw, signed_style) as i32
                }
                MirType::Scalar(ScalarType::I8) => {
                    narrow_imm_i64(8, raw, signed_style) as i32
                }
                MirType::Scalar(ScalarType::I1) => (raw & 1) as i32,
                _ => {
                    return Err(RasError::EncodingError(
                        "AArch64 JIT IntBinary: internal type mismatch for immediate".into(),
                    ));
                }
            };
            code.extend_from_slice(&mov_imm_to_w(dst_reg, v));
        }
        Operand::Register(Register::Virtual(vreg)) => {
            let off = stack_slots.get(vreg).copied().ok_or_else(|| {
                RasError::EncodingError(format!("No stack slot for virtual register: {:?}", vreg))
            })?;
            match ty {
                MirType::Scalar(ScalarType::F32 | ScalarType::F64) => {
                    return Err(RasError::EncodingError(
                        "AArch64 JIT IntBinary: floating-point MIR type".into(),
                    ));
                }
                MirType::Scalar(_) => {
                    code.extend_from_slice(&encode_ldr_typed(&dst_w, 29, off, ty)?);
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "AArch64 JIT IntBinary: unsupported type {:?}",
                        ty
                    )));
                }
            }
        }
        Operand::Register(Register::Physical(_)) => {
            return Err(RasError::EncodingError(
                "Physical register operands not yet fully supported".to_string(),
            ));
        }
    }
    Ok(())
}

/// Place the return value in `x0` per AAPCS64 (narrow integers are sign-extended in `x0`).
#[cfg(feature = "encoder")]
pub(super) fn materialize_return_value(
    assembler: &mut RasAssembler,
    op: &lamina_mir::Operand,
    ret_ty: &lamina_mir::MirType,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    stack_size: usize,
) -> Result<(), RasError> {
    match op {
        Operand::Register(Register::Virtual(vreg)) => {
            let offset = stack_slots.get(vreg).copied().ok_or_else(|| {
                RasError::EncodingError(format!("No stack slot for virtual register: {:?}", vreg))
            })?;
            match ret_ty {
                MirType::Scalar(ScalarType::I64 | ScalarType::Ptr) => {
                    code.extend_from_slice(&encode_ldr_typed("x0", 29, offset, ret_ty)?);
                }
                MirType::Scalar(
                    ScalarType::I32 | ScalarType::I16 | ScalarType::I8 | ScalarType::I1,
                ) => {
                    code.extend_from_slice(&encode_ldr_typed("w0", 29, offset, ret_ty)?);
                    code.extend_from_slice(&encode_sbfm64(0, 0, 0, 31));
                }
                _ => {
                    return Err(RasError::EncodingError(format!(
                        "AArch64 JIT: unsupported return type {:?}",
                        ret_ty
                    )));
                }
            }
            Ok(())
        }
        Operand::Immediate(_) | Operand::Register(Register::Physical(_)) => {
            materialize_operand(assembler, op, 0, stack_slots, reg_alloc, code, stack_size)
        }
    }
}

/// Loads from stack slot if operand is a virtual register, or moves immediate
#[cfg(feature = "encoder")]
pub(super) fn materialize_operand(
    _assembler: &mut RasAssembler,
    op: &lamina_mir::Operand,
    dst_reg: u8,
    stack_slots: &std::collections::HashMap<lamina_mir::VirtualReg, i32>,
    _reg_alloc: &mut lamina_codegen::aarch64::A64RegAlloc,
    code: &mut Vec<u8>,
    _stack_size: usize,
) -> Result<(), RasError> {
    match op {
        Operand::Immediate(imm) => {
            let imm_val: u64 = match imm {
                Immediate::I8(v) => *v as i64 as u64,
                Immediate::I16(v) => *v as i64 as u64,
                Immediate::I32(v) => *v as i64 as u64,
                Immediate::I64(v) => *v as u64,
                _ => {
                    return Err(RasError::EncodingError(
                        "Floating-point immediates not yet supported".to_string(),
                    ));
                }
            };

            if imm_val <= 0xFFFF {
                let movz = 0xD280_0000u32 | ((imm_val as u32) << 5) | (dst_reg as u32);
                code.extend_from_slice(&movz.to_le_bytes());
                return Ok(());
            }

            let chunk0 = (imm_val & 0xFFFF) as u16;
            let chunk1 = ((imm_val >> 16) & 0xFFFF) as u16;
            let chunk2 = ((imm_val >> 32) & 0xFFFF) as u16;
            let chunk3 = ((imm_val >> 48) & 0xFFFF) as u16;

            let movz = 0xD280_0000u32 | ((chunk0 as u32) << 5) | (dst_reg as u32);
            code.extend_from_slice(&movz.to_le_bytes());

            if chunk1 != 0 {
                let movk =
                    0xF280_0000u32 | (0b01u32 << 21) | ((chunk1 as u32) << 5) | (dst_reg as u32);
                code.extend_from_slice(&movk.to_le_bytes());
            }
            if chunk2 != 0 {
                let movk =
                    0xF280_0000u32 | (0b10u32 << 21) | ((chunk2 as u32) << 5) | (dst_reg as u32);
                code.extend_from_slice(&movk.to_le_bytes());
            }
            if chunk3 != 0 {
                let movk =
                    0xF280_0000u32 | (0b11u32 << 21) | ((chunk3 as u32) << 5) | (dst_reg as u32);
                code.extend_from_slice(&movk.to_le_bytes());
            }
        }
        Operand::Register(Register::Virtual(vreg)) => {
            // FrameMap stack slots are FP-relative offsets (negative for locals).
            if let Some(offset) = stack_slots.get(vreg) {
                let dst_reg_str = format!("x{}", dst_reg);
                code.extend_from_slice(&encode_ldr(
                    &dst_reg_str,
                    29,
                    *offset,
                )?);
            } else {
                return Err(RasError::EncodingError(format!(
                    "No stack slot for virtual register: {:?}",
                    vreg
                )));
            }
        }
        Operand::Register(Register::Physical(_)) => {
            // Physical registers are already in place, but we need to move to dst
            // For now, assume it's already correct (this is a simplification)
            return Err(RasError::EncodingError(
                "Physical register operands not yet fully supported".to_string(),
            ));
        }
    }

    Ok(())
}
