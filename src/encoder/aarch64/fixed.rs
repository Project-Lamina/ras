use crate::error::RasError;

pub(crate) fn enc_stp_pre_64(rt: u8, rt2: u8, rn: u8, imm_bytes: i32) -> Result<u32, RasError> {
    if imm_bytes % 8 != 0 {
        return Err(RasError::EncodingError(
            "STP imm must be multiple of 8".into(),
        ));
    }
    let imm7 = imm_bytes / 8;
    if !(-64..=63).contains(&imm7) {
        return Err(RasError::EncodingError(format!(
            "STP imm7 out of range: {}",
            imm7
        )));
    }
    let imm7_bits = (imm7 as u32) & 0x7F;

    // STP 64-bit, pre-index base opcode: 0xA980_0000
    // imm7 at [21:15], Rt2 at [14:10], Rn at [9:5], Rt at [4:0]
    Ok(0xA980_0000 | (imm7_bits << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32))
}

pub(crate) fn enc_ldp_post_64(rt: u8, rt2: u8, rn: u8, imm_bytes: i32) -> Result<u32, RasError> {
    if imm_bytes % 8 != 0 {
        return Err(RasError::EncodingError(
            "LDP imm must be multiple of 8".into(),
        ));
    }
    let imm7 = imm_bytes / 8;
    if !(-64..=63).contains(&imm7) {
        return Err(RasError::EncodingError(format!(
            "LDP imm7 out of range: {}",
            imm7
        )));
    }
    let imm7_bits = (imm7 as u32) & 0x7F;

    // LDP 64-bit, post-index base opcode: 0xA8C0_0000
    // imm7 at [21:15], Rt2 at [14:10], Rn at [9:5], Rt at [4:0]
    Ok(0xA8C0_0000 | (imm7_bits << 15) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32))
}

pub(crate) fn encode_msub(rd: u8, rn: u8, rm: u8, ra: u8) -> Vec<u8> {
    let w = 0x9B008000u32
        | ((rm as u32) << 16)
        | ((ra as u32) << 10)
        | ((rn as u32) << 5)
        | (rd as u32);
    w.to_le_bytes().to_vec()
}

#[cfg(feature = "encoder")]
pub(crate) fn encode_add_rrr(rd: u8, rn: u8, rm: u8) -> Vec<u8> {
    let w = 0x8B000000u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32);
    w.to_le_bytes().to_vec()
}

#[cfg(feature = "encoder")]
pub(crate) fn encode_and_rrr(rd: u8, rn: u8, rm: u8) -> Vec<u8> {
    let w = 0x8A000000u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32);
    w.to_le_bytes().to_vec()
}

pub(crate) fn encode_msub_w(rd: u8, rn: u8, rm: u8, ra: u8) -> Vec<u8> {
    let w = 0x1B00_8000u32
        | ((rm as u32) << 16)
        | ((ra as u32) << 10)
        | ((rn as u32) << 5)
        | (rd as u32);
    w.to_le_bytes().to_vec()
}

#[cfg(feature = "encoder")]
pub(crate) fn encode_and_w_rrr(rd: u8, rn: u8, rm: u8) -> Vec<u8> {
    let w = 0x0A00_0000u32 | ((rm as u32) << 16) | ((rn as u32) << 5) | (rd as u32);
    w.to_le_bytes().to_vec()
}
