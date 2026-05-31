//! AArch64 `LDR` / `STR` with immediate offset: shared by the MIR JIT assembler and the
//! `.s` text encoder so encoding rules stay in one place.

use crate::error::RasError;

const STUR_BASE: u32 = 0xF800_0000;
const LDUR_BASE: u32 = 0xF840_0000;
const STR_SCALED_BASE: u32 = 0xF900_0000;
const LDR_SCALED_BASE: u32 = 0xF940_0000;
const SUB_IMM_BASE: u32 = 0xD100_0000;
const TEMP_RN: u32 = 10;

/// Integer scalar load shape for JIT (GP registers only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AArch64ScalarLdKind {
    I8S,
    I8U,
    I16S,
    I32S,
    I64,
}

/// Integer scalar store shape for JIT (GP registers only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AArch64ScalarStKind {
    I8,
    I16,
    I32,
    I64,
}

#[inline]
fn ldur_simm9_base(kind: AArch64ScalarLdKind) -> u32 {
    match kind {
        AArch64ScalarLdKind::I8S => 0x3880_0000,
        AArch64ScalarLdKind::I8U => 0x3840_0000,
        AArch64ScalarLdKind::I16S => 0x7880_0000,
        AArch64ScalarLdKind::I32S => 0xB980_0000,
        AArch64ScalarLdKind::I64 => LDUR_BASE,
    }
}

#[inline]
fn stur_simm9_base(kind: AArch64ScalarStKind) -> u32 {
    match kind {
        AArch64ScalarStKind::I8 => 0x3900_0000,
        AArch64ScalarStKind::I16 => 0x7900_0000,
        AArch64ScalarStKind::I32 => 0xB900_0000,
        AArch64ScalarStKind::I64 => STUR_BASE,
    }
}

#[inline]
fn ldr_scaled_base(kind: AArch64ScalarLdKind) -> (u32, u32) {
    match kind {
        AArch64ScalarLdKind::I8S => (0x3980_0000, 1),
        AArch64ScalarLdKind::I8U => (0x3940_0000, 1),
        AArch64ScalarLdKind::I16S => (0x7980_0000, 2),
        AArch64ScalarLdKind::I32S => (0xB980_0000, 4),
        AArch64ScalarLdKind::I64 => (LDR_SCALED_BASE, 8),
    }
}

#[inline]
fn str_scaled_base(kind: AArch64ScalarStKind) -> (u32, u32) {
    match kind {
        AArch64ScalarStKind::I8 => (0x3900_0000, 1),
        AArch64ScalarStKind::I16 => (0x7900_0000, 2),
        AArch64ScalarStKind::I32 => (0xB900_0000, 4),
        AArch64ScalarStKind::I64 => (STR_SCALED_BASE, 8),
    }
}

/// `LDUR` / scaled `LDR` / `SUB`+`LDUR` for scalar integer loads.
pub fn encode_ldr_scalar(
    rt: u8,
    rn: u8,
    offset: i32,
    kind: AArch64ScalarLdKind,
) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    if (-256..=255).contains(&offset) {
        let imm9 = (offset as u32) & 0x1FF;
        let inst = ldur_simm9_base(kind) | (imm9 << 12) | ((rn as u32) << 5) | (rt as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }
    if offset >= 0 {
        let (base, align) = ldr_scaled_base(kind);
        if (offset as u32).is_multiple_of(align) {
            let imm12 = (offset as u32) / align;
            if imm12 <= 0xFFF {
                let inst = base | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32);
                code.extend_from_slice(&inst.to_le_bytes());
                return Ok(code);
            }
        }
    }
    if offset < -256 {
        let abs_offset = (-offset) as u32;
        if abs_offset > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "scalar LDR offset {} out of range",
                offset
            )));
        }
        let sub_inst = SUB_IMM_BASE | ((abs_offset & 0xFFF) << 10) | ((rn as u32) << 5) | TEMP_RN;
        code.extend_from_slice(&sub_inst.to_le_bytes());
        let ldur = ldur_simm9_base(kind) | (TEMP_RN << 5) | (rt as u32);
        code.extend_from_slice(&ldur.to_le_bytes());
        return Ok(code);
    }
    Err(RasError::EncodingError(format!(
        "scalar LDR offset {} out of range for {:?}",
        offset, kind
    )))
}

/// `STUR` / scaled `STR` / `SUB`+`STUR` for scalar integer stores.
pub fn encode_str_scalar(
    rt: u8,
    rn: u8,
    offset: i32,
    kind: AArch64ScalarStKind,
) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    if (-256..=255).contains(&offset) {
        let imm9 = (offset as u32) & 0x1FF;
        let inst = stur_simm9_base(kind) | (imm9 << 12) | ((rn as u32) << 5) | (rt as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }
    if offset >= 0 {
        let (base, align) = str_scaled_base(kind);
        if (offset as u32).is_multiple_of(align) {
            let imm12 = (offset as u32) / align;
            if imm12 <= 0xFFF {
                let inst = base | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32);
                code.extend_from_slice(&inst.to_le_bytes());
                return Ok(code);
            }
        }
    }
    if offset < -256 {
        let abs_offset = (-offset) as u32;
        if abs_offset > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "scalar STR offset {} out of range",
                offset
            )));
        }
        let sub_inst = SUB_IMM_BASE | ((abs_offset & 0xFFF) << 10) | ((rn as u32) << 5) | TEMP_RN;
        code.extend_from_slice(&sub_inst.to_le_bytes());
        let stur = stur_simm9_base(kind) | (TEMP_RN << 5) | (rt as u32);
        code.extend_from_slice(&stur.to_le_bytes());
        return Ok(code);
    }
    Err(RasError::EncodingError(format!(
        "scalar STR offset {} out of range for {:?}",
        offset, kind
    )))
}

/// STUR/ST scaled/SUB+STUR for 64-bit stores. `rt`/`rn` are AArch64 GPR encodings 0–31.
pub fn encode_str_imm64(rt: u8, rn: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    if (-256..=255).contains(&offset) {
        let imm9 = (offset as u32) & 0x1FF;
        let inst = STUR_BASE | (imm9 << 12) | ((rn as u32) << 5) | (rt as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }
    if (0..=(0xFFF * 8)).contains(&offset) && (offset % 8 == 0) {
        let imm12 = (offset as u32) / 8;
        let inst = STR_SCALED_BASE | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }
    if offset < -256 {
        let abs_offset = (-offset) as u32;
        if abs_offset > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "STR offset {} out of range",
                offset
            )));
        }
        let sub_inst = SUB_IMM_BASE | ((abs_offset & 0xFFF) << 10) | ((rn as u32) << 5) | TEMP_RN;
        code.extend_from_slice(&sub_inst.to_le_bytes());
        let stur = STUR_BASE | (TEMP_RN << 5) | (rt as u32);
        code.extend_from_slice(&stur.to_le_bytes());
        return Ok(code);
    }
    Err(RasError::EncodingError(format!(
        "STR offset {} out of range",
        offset
    )))
}

/// LDUR/LDR scaled/SUB+LDUR for 64-bit loads. `rt`/`rn` are AArch64 GPR encodings 0–31.
pub fn encode_ldr_imm64(rt: u8, rn: u8, offset: i32) -> Result<Vec<u8>, RasError> {
    let mut code = Vec::new();
    if (-256..=255).contains(&offset) {
        let imm9 = (offset as u32) & 0x1FF;
        let inst = LDUR_BASE | (imm9 << 12) | ((rn as u32) << 5) | (rt as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }
    if (0..=(0xFFF * 8)).contains(&offset) && (offset % 8 == 0) {
        let imm12 = (offset as u32) / 8;
        let inst = LDR_SCALED_BASE | (imm12 << 10) | ((rn as u32) << 5) | (rt as u32);
        code.extend_from_slice(&inst.to_le_bytes());
        return Ok(code);
    }
    if offset < -256 {
        let abs_offset = (-offset) as u32;
        if abs_offset > 0xFFF {
            return Err(RasError::EncodingError(format!(
                "LDR offset {} out of range",
                offset
            )));
        }
        let sub_inst = SUB_IMM_BASE | ((abs_offset & 0xFFF) << 10) | ((rn as u32) << 5) | TEMP_RN;
        code.extend_from_slice(&sub_inst.to_le_bytes());
        let ldur = LDUR_BASE | (TEMP_RN << 5) | (rt as u32);
        code.extend_from_slice(&ldur.to_le_bytes());
        return Ok(code);
    }
    Err(RasError::EncodingError(format!(
        "LDR offset {} out of range",
        offset
    )))
}

#[cfg(test)]
mod aarch64_ldst_imm64_tests {
    use super::{
        AArch64ScalarLdKind, AArch64ScalarStKind, encode_ldr_imm64, encode_ldr_scalar,
        encode_str_imm64, encode_str_scalar,
    };

    #[test]
    fn ldrsb_ldur_unscaled_matches_as() {
        let b = encode_ldr_scalar(0, 1, 0, AArch64ScalarLdKind::I8S).expect("encode");
        assert_eq!(b, [0x20, 0x00, 0x80, 0x38]);
    }

    #[test]
    fn ldrsw_unscaled_small_offset_encodes_like_ldursw() {
        let b = encode_ldr_scalar(0, 1, 8, AArch64ScalarLdKind::I32S).expect("encode");
        assert_eq!(b, [0x20, 0x80, 0x80, 0xB9]);
    }

    #[test]
    fn ldrsw_scaled_positive_offset_uses_ldr_form() {
        let b = encode_ldr_scalar(0, 1, 0x1000, AArch64ScalarLdKind::I32S).expect("encode");
        assert_eq!(b, [0x20, 0x00, 0x90, 0xB9]);
    }

    #[test]
    fn strb_unscaled_small_offset_encodes_like_sturb() {
        let b = encode_str_scalar(0, 1, 8, AArch64ScalarStKind::I8).expect("encode");
        assert_eq!(b, [0x20, 0x80, 0x00, 0x39]);
    }

    #[test]
    fn ldr_imm64_scaled_1024() {
        let b = encode_ldr_imm64(0, 1, 1024).expect("encode");
        assert_eq!(b, vec![0x20, 0x00, 0x42, 0xF9]);
    }

    #[test]
    fn str_imm64_scaled_512() {
        let b = encode_str_imm64(2, 3, 512).expect("encode");
        assert_eq!(b, vec![0x62, 0x00, 0x01, 0xF9]);
    }

    #[test]
    fn ldr_imm64_rejects_unaligned_260() {
        let err = encode_ldr_imm64(0, 1, 260).expect_err("260");
        match err {
            crate::error::RasError::EncodingError(msg) => assert!(msg.contains("260"), "{msg}"),
            _ => panic!("expected EncodingError"),
        }
    }
}
