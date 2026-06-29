use std::path::PathBuf;

use model_store::{BakedStore, FlmLoadOptions};

#[test]
fn flm_native_qwen_bpe_tokenizer_matches_embedded_hf_json_for_basic_prompt() {
    let Some(path) = std::env::var_os("SUPERSONIC_QWEN36_27B_RUNNABLE_FLM") else {
        eprintln!("skipping: SUPERSONIC_QWEN36_27B_RUNNABLE_FLM is unset");
        return;
    };
    let path = PathBuf::from(path);
    if !path.exists() {
        eprintln!(
            "skipping: SUPERSONIC_QWEN36_27B_RUNNABLE_FLM path does not exist: {}",
            path.display()
        );
        return;
    }
    let store = BakedStore::open_flm_with_options(
        &path,
        FlmLoadOptions {
            compressed_tensors_int4_aliases: true,
        },
    )
    .expect("open runnable FLM fixture");
    let runtime = match store.flm_runtime() {
        Some(runtime) => runtime,
        None => {
            eprintln!("skipping: FLM fixture has no runtime directory");
            return;
        }
    };
    let oracle_json = match runtime.asset_by_kind("hf_tokenizer_json") {
        Some(asset) => &asset.payload,
        None => {
            eprintln!("skipping: FLM fixture has no hf_tokenizer_json compatibility asset");
            return;
        }
    };

    let native = runner::flm_tokenizer::load_qwen_bpe_from_flm(runtime)
        .expect("load native Qwen BPE tokenizer from FLM assets");
    let oracle = tokenizers::Tokenizer::from_bytes(oracle_json)
        .expect("load embedded Hugging Face tokenizer oracle");

    for prompt in ["Hello", "The quick brown fox", "GPU quantization"] {
        let native_ids = native.encode(prompt, true).unwrap().get_ids().to_vec();
        let oracle_ids = oracle.encode(prompt, true).unwrap().get_ids().to_vec();
        assert_eq!(native_ids, oracle_ids, "prompt={prompt:?}");
    }
}
