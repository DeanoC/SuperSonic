//! Neutral wire-codec identifiers shared by the public GQH reader and the
//! internal FLM codec foundation.
//!
//! Keep this module independent of model/runtime descriptors.  The numeric
//! IDs are part of the artifact wire contracts and must not be redefined in a
//! model-specific loader.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GqhCodecIds {
    pub gguf_qtype: u32,
    pub flm_codec: u16,
}

pub const GGML_TYPE_GQH3: u32 = 108;
pub const GGML_TYPE_GQH2_H: u32 = 109;
pub const GGML_TYPE_GQH2_C: u32 = 110;
pub const GGML_TYPE_GQH4: u32 = 111;

pub const CODEC_GQH3: u16 = 13;
pub const CODEC_GQH2_H: u16 = 14;
pub const CODEC_GQH2_C: u16 = 15;
pub const CODEC_GQH4: u16 = 16;

pub const GQH_CODEC_IDS: [GqhCodecIds; 4] = [
    GqhCodecIds {
        gguf_qtype: GGML_TYPE_GQH3,
        flm_codec: CODEC_GQH3,
    },
    GqhCodecIds {
        gguf_qtype: GGML_TYPE_GQH2_H,
        flm_codec: CODEC_GQH2_H,
    },
    GqhCodecIds {
        gguf_qtype: GGML_TYPE_GQH2_C,
        flm_codec: CODEC_GQH2_C,
    },
    GqhCodecIds {
        gguf_qtype: GGML_TYPE_GQH4,
        flm_codec: CODEC_GQH4,
    },
];

pub fn flm_codec_for_ggml_type(gguf_qtype: u32) -> Option<u16> {
    match gguf_qtype {
        GGML_TYPE_GQH3 => Some(CODEC_GQH3),
        GGML_TYPE_GQH2_H => Some(CODEC_GQH2_H),
        GGML_TYPE_GQH2_C => Some(CODEC_GQH2_C),
        GGML_TYPE_GQH4 => Some(CODEC_GQH4),
        _ => None,
    }
}

pub fn ggml_type_for_flm_codec(flm_codec: u16) -> Option<u32> {
    match flm_codec {
        CODEC_GQH3 => Some(GGML_TYPE_GQH3),
        CODEC_GQH2_H => Some(GGML_TYPE_GQH2_H),
        CODEC_GQH2_C => Some(GGML_TYPE_GQH2_C),
        CODEC_GQH4 => Some(GGML_TYPE_GQH4),
        _ => None,
    }
}
