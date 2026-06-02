use std::collections::HashMap;

use model_store::manifest::{LayoutTag, Manifest, TensorMeta};
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct Qwen36Q4KmAuditSpec {
    pub weight_prefix: String,
    pub layer_is_full: Vec<bool>,
    pub tied_lm_head: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen36Q4KmProjectionFamily {
    FullAttention,
    LinearAttention,
    SharedExpert,
    RoutedExpert,
    LmHead,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen36Q4KmProjectionStatus {
    Missing,
    Bf16OrUnquantized,
    NativeInt4Sidecars,
    RawGgmlKBlock,
    UnsupportedLayout,
}

#[derive(Debug, Clone, Serialize)]
pub struct Qwen36Q4KmProjectionAudit {
    pub name: String,
    pub family: Qwen36Q4KmProjectionFamily,
    pub layout: Option<String>,
    pub dtype: Option<String>,
    pub shape: Option<Vec<usize>>,
    pub status: Qwen36Q4KmProjectionStatus,
    pub supported_by_current_metal: bool,
    pub blocker: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Qwen36Q4KmAuditSummary {
    pub projections: usize,
    pub missing: usize,
    pub native_int4_sidecars: usize,
    pub raw_ggml_k_blocks: usize,
    pub bf16_or_unquantized: usize,
    pub unsupported_layout: usize,
    pub current_metal_blockers: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Qwen36Q4KmAuditReport {
    pub weight_prefix: String,
    pub num_layers: usize,
    pub full_attention_layers: usize,
    pub linear_attention_layers: usize,
    pub tied_lm_head: bool,
    pub summary: Qwen36Q4KmAuditSummary,
    pub projections: Vec<Qwen36Q4KmProjectionAudit>,
}

pub fn audit_qwen36_q4km_manifest(
    manifest: &Manifest,
    spec: &Qwen36Q4KmAuditSpec,
) -> Qwen36Q4KmAuditReport {
    let index: HashMap<&str, &TensorMeta> = manifest
        .tensors
        .iter()
        .map(|meta| (meta.name.as_str(), meta))
        .collect();
    let mut projections = Vec::new();

    for (layer_idx, is_full) in spec.layer_is_full.iter().copied().enumerate() {
        let lp = format!("{}.layers.{layer_idx}", spec.weight_prefix);
        if is_full {
            let fa = format!("{lp}.self_attn");
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                projections.push(audit_projection(
                    &index,
                    format!("{fa}.{proj}.weight"),
                    Qwen36Q4KmProjectionFamily::FullAttention,
                    SidecarStyle::DenseWeight,
                ));
            }
        } else {
            let la = format!("{lp}.linear_attn");
            for proj in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                projections.push(audit_projection(
                    &index,
                    format!("{la}.{proj}.weight"),
                    Qwen36Q4KmProjectionFamily::LinearAttention,
                    SidecarStyle::DenseWeight,
                ));
            }
        }

        let mp = format!("{lp}.mlp");
        for proj in [
            "shared_expert.gate_proj",
            "shared_expert.up_proj",
            "shared_expert.down_proj",
        ] {
            projections.push(audit_projection(
                &index,
                format!("{mp}.{proj}.weight"),
                Qwen36Q4KmProjectionFamily::SharedExpert,
                SidecarStyle::DenseWeight,
            ));
        }
        for proj in ["experts.gate_up_proj", "experts.down_proj"] {
            projections.push(audit_projection(
                &index,
                format!("{mp}.{proj}"),
                Qwen36Q4KmProjectionFamily::RoutedExpert,
                SidecarStyle::FusedExpert,
            ));
        }
    }

    let lm_head_name = if spec.tied_lm_head {
        format!("{}.embed_tokens.weight", spec.weight_prefix)
    } else {
        "lm_head.weight".to_string()
    };
    projections.push(audit_projection(
        &index,
        lm_head_name,
        Qwen36Q4KmProjectionFamily::LmHead,
        SidecarStyle::DenseWeight,
    ));

    let mut summary = Qwen36Q4KmAuditSummary::default();
    summary.projections = projections.len();
    for projection in &projections {
        match projection.status {
            Qwen36Q4KmProjectionStatus::Missing => summary.missing += 1,
            Qwen36Q4KmProjectionStatus::Bf16OrUnquantized => summary.bf16_or_unquantized += 1,
            Qwen36Q4KmProjectionStatus::NativeInt4Sidecars => summary.native_int4_sidecars += 1,
            Qwen36Q4KmProjectionStatus::RawGgmlKBlock => summary.raw_ggml_k_blocks += 1,
            Qwen36Q4KmProjectionStatus::UnsupportedLayout => summary.unsupported_layout += 1,
        }
        if !projection.supported_by_current_metal {
            summary.current_metal_blockers += 1;
        }
    }

    Qwen36Q4KmAuditReport {
        weight_prefix: spec.weight_prefix.clone(),
        num_layers: spec.layer_is_full.len(),
        full_attention_layers: spec.layer_is_full.iter().filter(|&&v| v).count(),
        linear_attention_layers: spec.layer_is_full.iter().filter(|&&v| !v).count(),
        tied_lm_head: spec.tied_lm_head,
        summary,
        projections,
    }
}

#[derive(Debug, Clone, Copy)]
enum SidecarStyle {
    DenseWeight,
    FusedExpert,
}

fn audit_projection(
    index: &HashMap<&str, &TensorMeta>,
    name: String,
    family: Qwen36Q4KmProjectionFamily,
    sidecar_style: SidecarStyle,
) -> Qwen36Q4KmProjectionAudit {
    let Some(meta) = index.get(name.as_str()).copied() else {
        return Qwen36Q4KmProjectionAudit {
            name,
            family,
            layout: None,
            dtype: None,
            shape: None,
            status: Qwen36Q4KmProjectionStatus::Missing,
            supported_by_current_metal: false,
            blocker: Some("required tensor is missing from the manifest".to_string()),
        };
    };

    let status = if has_int4_sidecars(index, &name, sidecar_style) {
        Qwen36Q4KmProjectionStatus::NativeInt4Sidecars
    } else if is_ggml_k_layout(&meta.layout) {
        Qwen36Q4KmProjectionStatus::RawGgmlKBlock
    } else if is_unquantized_layout(&meta.layout) {
        Qwen36Q4KmProjectionStatus::Bf16OrUnquantized
    } else {
        Qwen36Q4KmProjectionStatus::UnsupportedLayout
    };
    let (supported_by_current_metal, blocker) = support_for(family, &status, &name);

    Qwen36Q4KmProjectionAudit {
        name,
        family,
        layout: Some(format!("{:?}", meta.layout)),
        dtype: Some(meta.dtype.clone()),
        shape: Some(meta.shape.clone()),
        status,
        supported_by_current_metal,
        blocker,
    }
}

fn has_int4_sidecars(
    index: &HashMap<&str, &TensorMeta>,
    name: &str,
    _sidecar_style: SidecarStyle,
) -> bool {
    let scale = format!("{name}_int4_scale");
    let zero = format!("{name}_int4_zero");
    index.contains_key(scale.as_str()) && index.contains_key(zero.as_str())
}

fn support_for(
    family: Qwen36Q4KmProjectionFamily,
    status: &Qwen36Q4KmProjectionStatus,
    name: &str,
) -> (bool, Option<String>) {
    match (family, status) {
        (_, Qwen36Q4KmProjectionStatus::Missing) => (
            false,
            Some("required tensor is missing from the manifest".to_string()),
        ),
        (_, Qwen36Q4KmProjectionStatus::UnsupportedLayout) => (
            false,
            Some(format!("{name}: layout is not supported by the current Qwen36 Metal path")),
        ),
        (_, Qwen36Q4KmProjectionStatus::NativeInt4Sidecars)
        | (_, Qwen36Q4KmProjectionStatus::Bf16OrUnquantized) => (true, None),
        (Qwen36Q4KmProjectionFamily::RoutedExpert, Qwen36Q4KmProjectionStatus::RawGgmlKBlock) => {
            (true, None)
        }
        (_, Qwen36Q4KmProjectionStatus::RawGgmlKBlock) => (
            false,
            Some(format!(
                "{name}: raw GGML K-block projection needs Metal loader/kernel support that consumes per-projection qtype metadata"
            )),
        ),
    }
}

fn is_ggml_k_layout(layout: &LayoutTag) -> bool {
    matches!(
        layout,
        LayoutTag::GgmlQ4K | LayoutTag::GgmlQ5K | LayoutTag::GgmlQ6K
    )
}

fn is_unquantized_layout(layout: &LayoutTag) -> bool {
    matches!(
        layout,
        LayoutTag::Raw
            | LayoutTag::DepthwiseConvSqueezed
            | LayoutTag::HeadBiasReshaped
            | LayoutTag::HeadExpReshaped
            | LayoutTag::Fp8Dequantized
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_store::manifest::{TensorMeta, CONVERTER_VERSION, FORMAT_VERSION};

    fn meta(name: &str, layout: LayoutTag) -> TensorMeta {
        TensorMeta {
            name: name.to_string(),
            shape: vec![128, 64],
            dtype: if matches!(
                layout,
                LayoutTag::GgmlQ4K | LayoutTag::GgmlQ5K | LayoutTag::GgmlQ6K
            ) {
                "u8".to_string()
            } else {
                "bf16".to_string()
            },
            layout,
            offset: 0,
            byte_len: 128,
        }
    }

    fn manifest(tensors: Vec<TensorMeta>) -> Manifest {
        Manifest {
            format_version: FORMAT_VERSION,
            converter_version: CONVERTER_VERSION,
            model_family: "qwen36-moe".to_string(),
            quant_profile: Some("q4km-ggml-v1".to_string()),
            source_format: Some("gguf".to_string()),
            source_quant: Some("Q4_K_M".to_string()),
            quant_method: None,
            tensors,
        }
    }

    fn spec() -> Qwen36Q4KmAuditSpec {
        Qwen36Q4KmAuditSpec {
            weight_prefix: "model.language_model".to_string(),
            layer_is_full: vec![false, false, false, true],
            tied_lm_head: false,
        }
    }

    #[test]
    fn raw_dense_and_shared_q4k_are_blockers_but_routed_experts_are_supported() {
        let mut tensors = Vec::new();
        for li in 0..4 {
            let lp = format!("model.language_model.layers.{li}");
            if li == 3 {
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    tensors.push(meta(
                        &format!("{lp}.self_attn.{proj}.weight"),
                        LayoutTag::GgmlQ4K,
                    ));
                }
            } else {
                for proj in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                    tensors.push(meta(
                        &format!("{lp}.linear_attn.{proj}.weight"),
                        LayoutTag::GgmlQ4K,
                    ));
                }
            }
            for proj in [
                "shared_expert.gate_proj",
                "shared_expert.up_proj",
                "shared_expert.down_proj",
            ] {
                tensors.push(meta(&format!("{lp}.mlp.{proj}.weight"), LayoutTag::GgmlQ4K));
            }
            for proj in ["experts.gate_up_proj", "experts.down_proj"] {
                tensors.push(meta(&format!("{lp}.mlp.{proj}"), LayoutTag::GgmlQ4K));
            }
        }
        tensors.push(meta("lm_head.weight", LayoutTag::Raw));

        let report = audit_qwen36_q4km_manifest(&manifest(tensors), &spec());

        assert_eq!(report.summary.raw_ggml_k_blocks, 33);
        assert_eq!(report.summary.current_metal_blockers, 25);
        assert!(report
            .projections
            .iter()
            .any(|p| { p.name.ends_with("experts.gate_up_proj") && p.supported_by_current_metal }));
        assert!(report.projections.iter().any(|p| {
            p.name.ends_with("linear_attn.in_proj_qkv.weight") && !p.supported_by_current_metal
        }));
    }

    #[test]
    fn native_int4_sidecars_make_dense_projection_supported() {
        let base = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let report = audit_qwen36_q4km_manifest(
            &manifest(vec![
                meta(base, LayoutTag::Int4Quantized),
                meta(&format!("{base}_int4_scale"), LayoutTag::Raw),
                meta(&format!("{base}_int4_zero"), LayoutTag::Raw),
            ]),
            &Qwen36Q4KmAuditSpec {
                weight_prefix: "model.language_model".to_string(),
                layer_is_full: vec![false],
                tied_lm_head: true,
            },
        );

        let audited = report
            .projections
            .iter()
            .find(|p| p.name == base)
            .expect("audited dense projection");
        assert_eq!(
            audited.status,
            Qwen36Q4KmProjectionStatus::NativeInt4Sidecars
        );
        assert!(audited.supported_by_current_metal);
    }
}
