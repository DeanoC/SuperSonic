use std::collections::HashSet;

use anyhow::{anyhow, bail, Context, Result};
use model_store::flm::TOKENIZER_QWEN_BPE_V1;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::tokenizer::SplitDelimiterBehavior;
use tokenizers::{AddedToken, Tokenizer};

const VOCAB_KIND: &str = "tokenizer_vocab";
const MERGES_KIND: &str = "tokenizer_merges";
const ADDED_TOKENS_KIND: &str = "tokenizer_added_tokens";
const REGEX_KIND: &str = "tokenizer_regex";

#[derive(Debug)]
struct QwenBpeAssets {
    vocab: Vocab,
    merges: Vec<(String, String)>,
    added_tokens: Vec<QwenAddedToken>,
    regex: String,
}

#[derive(Debug)]
struct QwenAddedToken {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

impl QwenBpeAssets {
    fn parse(
        algorithm_id: u32,
        vocab: &[u8],
        merges: &[u8],
        added_tokens: &[u8],
        regex: &[u8],
    ) -> Result<Self> {
        if algorithm_id != TOKENIZER_QWEN_BPE_V1 {
            bail!(
                "unsupported FLM tokenizer algorithm_id {algorithm_id}; expected TOKENIZER_QWEN_BPE_V1 ({TOKENIZER_QWEN_BPE_V1})"
            );
        }
        Ok(Self {
            vocab: parse_vocab(vocab)?,
            merges: parse_merges(merges)?,
            added_tokens: parse_added_tokens(added_tokens)?,
            regex: parse_regex(regex)?,
        })
    }
}

pub fn load_qwen_bpe_from_flm(
    runtime: &model_store::FlmRuntimeDirectory,
) -> Result<tokenizers::Tokenizer> {
    let descriptor = runtime
        .tokenizer()
        .ok_or_else(|| anyhow!("FLM runtime has no tokenizer descriptor"))?;
    let vocab = asset_payload(runtime, descriptor.vocab_asset_id, VOCAB_KIND)?;
    let merges = asset_payload(runtime, descriptor.merges_asset_id, MERGES_KIND)?;
    let added_tokens = asset_payload(runtime, descriptor.added_tokens_asset_id, ADDED_TOKENS_KIND)?;
    let regex = asset_payload(runtime, descriptor.regex_asset_id, REGEX_KIND)?;

    let assets = QwenBpeAssets::parse(descriptor.algorithm_id, vocab, merges, added_tokens, regex)?;
    if assets.vocab.len() != descriptor.vocab_size as usize {
        bail!(
            "FLM tokenizer vocab_size descriptor mismatch: descriptor={} asset={}",
            descriptor.vocab_size,
            assets.vocab.len()
        );
    }
    build_qwen_bpe_tokenizer(assets)
}

fn asset_payload<'a>(
    runtime: &'a model_store::FlmRuntimeDirectory,
    asset_id: u32,
    expected_kind: &str,
) -> Result<&'a [u8]> {
    let asset = runtime
        .asset(asset_id)
        .ok_or_else(|| anyhow!("FLM tokenizer missing {expected_kind} asset id {asset_id}"))?;
    if asset.kind != expected_kind {
        bail!(
            "FLM tokenizer asset id {asset_id} has kind {}; expected {expected_kind}",
            asset.kind
        );
    }
    Ok(&asset.payload)
}

fn build_qwen_bpe_tokenizer(assets: QwenBpeAssets) -> Result<Tokenizer> {
    let added_tokens = assets.added_tokens;
    let bpe = BPE::builder()
        .vocab_and_merges(assets.vocab, assets.merges)
        .byte_fallback(false)
        .ignore_merges(false)
        .fuse_unk(false)
        .build()
        .map_err(|e| anyhow!("build Qwen BPE model from FLM tokenizer assets: {e}"))?;

    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_normalizer(Some(NFC));
    let split = Split::new(
        SplitPattern::Regex(assets.regex),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("compile FLM tokenizer regex: {e}"))?;
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::from(split),
        PreTokenizerWrapper::from(ByteLevel::new(false, false, false)),
    ])));
    tokenizer.with_post_processor(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));

    for token in added_tokens {
        match tokenizer.token_to_id(&token.content) {
            Some(actual) if actual == token.id => {}
            Some(actual) => bail!(
                "FLM tokenizer added token id {} for {:?} does not match model id {actual}",
                token.id,
                token.content
            ),
            None => bail!(
                "FLM tokenizer added token {:?} with id {} is not present in BPE vocab",
                token.content,
                token.id
            ),
        }

        let added = AddedToken::from(token.content, token.special)
            .single_word(token.single_word)
            .lstrip(token.lstrip)
            .rstrip(token.rstrip)
            .normalized(token.normalized)
            .special(token.special);
        if added.special {
            tokenizer.add_special_tokens(&[added]);
        } else {
            tokenizer.add_tokens(&[added]);
        }
    }

    Ok(tokenizer)
}

fn parse_vocab(buf: &[u8]) -> Result<Vocab> {
    let mut reader = BinaryReader::new(VOCAB_KIND, buf);
    let count = reader.read_u32("count")? as usize;
    let mut vocab = Vocab::with_capacity(count);
    let mut ids = HashSet::with_capacity(count);
    for index in 0..count {
        let id = reader.read_u32(&format!("record {index} id"))?;
        if !ids.insert(id) {
            bail!("{VOCAB_KIND} has duplicate vocab id {id}");
        }
        let token = reader.read_string(&format!("record {index} token"))?;
        if vocab.insert(token.clone(), id).is_some() {
            bail!("{VOCAB_KIND} has duplicate vocab token {token:?}");
        }
    }
    reader.finish()?;
    Ok(vocab)
}

fn parse_merges(buf: &[u8]) -> Result<Vec<(String, String)>> {
    let mut reader = BinaryReader::new(MERGES_KIND, buf);
    let count = reader.read_u32("count")? as usize;
    let mut merges = Vec::with_capacity(count);
    let mut seen = HashSet::with_capacity(count);
    for index in 0..count {
        let left = reader.read_string(&format!("record {index} left"))?;
        let right = reader.read_string(&format!("record {index} right"))?;
        if !seen.insert((left.clone(), right.clone())) {
            bail!("{MERGES_KIND} has duplicate merge ({left:?}, {right:?})");
        }
        merges.push((left, right));
    }
    reader.finish()?;
    Ok(merges)
}

fn parse_added_tokens(buf: &[u8]) -> Result<Vec<QwenAddedToken>> {
    let mut reader = BinaryReader::new(ADDED_TOKENS_KIND, buf);
    let count = reader.read_u32("count")? as usize;
    let mut tokens = Vec::with_capacity(count);
    let mut ids = HashSet::with_capacity(count);
    let mut contents = HashSet::with_capacity(count);
    for index in 0..count {
        let id = reader.read_u32(&format!("record {index} id"))?;
        if !ids.insert(id) {
            bail!("{ADDED_TOKENS_KIND} has duplicate added token id {id}");
        }
        let content = reader.read_string(&format!("record {index} content"))?;
        if !contents.insert(content.clone()) {
            bail!("{ADDED_TOKENS_KIND} has duplicate added token content {content:?}");
        }
        tokens.push(QwenAddedToken {
            id,
            content,
            single_word: reader.read_bool_flag(index, "single_word")?,
            lstrip: reader.read_bool_flag(index, "lstrip")?,
            rstrip: reader.read_bool_flag(index, "rstrip")?,
            normalized: reader.read_bool_flag(index, "normalized")?,
            special: reader.read_bool_flag(index, "special")?,
        });
    }
    reader.finish()?;
    Ok(tokens)
}

fn parse_regex(buf: &[u8]) -> Result<String> {
    std::str::from_utf8(buf)
        .with_context(|| format!("{REGEX_KIND} is not valid UTF-8"))
        .map(str::to_owned)
}

struct BinaryReader<'a> {
    label: &'static str,
    buf: &'a [u8],
    offset: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(label: &'static str, buf: &'a [u8]) -> Self {
        Self {
            label,
            buf,
            offset: 0,
        }
    }

    fn read_u32(&mut self, field: &str) -> Result<u32> {
        let bytes = self.read_bytes(4, field)?;
        let mut value = [0u8; 4];
        value.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(value))
    }

    fn read_string(&mut self, field: &str) -> Result<String> {
        let len = self.read_u32(&format!("{field} len"))? as usize;
        let bytes = self.read_bytes(len, field)?;
        std::str::from_utf8(bytes)
            .with_context(|| format!("{} {field} is not valid UTF-8", self.label))
            .map(str::to_owned)
    }

    fn read_bool_flag(&mut self, record_index: usize, flag: &str) -> Result<bool> {
        let byte =
            self.read_bytes(1, &format!("record {record_index} added token flag {flag}"))?[0];
        match byte {
            0 => Ok(false),
            1 => Ok(true),
            _ => bail!(
                "{} record {record_index} added token flag {flag} has invalid value {byte}; expected 0 or 1",
                self.label
            ),
        }
    }

    fn read_bytes(&mut self, len: usize, field: &str) -> Result<&'a [u8]> {
        let end = self.offset.checked_add(len).ok_or_else(|| {
            anyhow!(
                "{} {field} byte range overflows (offset={}, len={len})",
                self.label,
                self.offset
            )
        })?;
        if end > self.buf.len() {
            bail!(
                "{} {field} truncated: need bytes [{}..{}), len={}",
                self.label,
                self.offset,
                end,
                self.buf.len()
            );
        }
        let bytes = &self.buf[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }

    fn finish(&self) -> Result<()> {
        if self.offset != self.buf.len() {
            bail!(
                "{} has {} trailing bytes",
                self.label,
                self.buf.len() - self.offset
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use model_store::flm::TOKENIZER_QWEN_BPE_V1;

    use super::*;

    fn u32_le(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn vocab_asset(entries: &[(u32, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        u32_le(&mut out, entries.len() as u32);
        for (id, token) in entries {
            u32_le(&mut out, *id);
            u32_le(&mut out, token.len() as u32);
            out.extend_from_slice(token.as_bytes());
        }
        out
    }

    fn merges_asset(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        u32_le(&mut out, entries.len() as u32);
        for (left, right) in entries {
            u32_le(&mut out, left.len() as u32);
            out.extend_from_slice(left.as_bytes());
            u32_le(&mut out, right.len() as u32);
            out.extend_from_slice(right.as_bytes());
        }
        out
    }

    fn added_tokens_asset(entries: &[(u32, &str, [u8; 5])]) -> Vec<u8> {
        let mut out = Vec::new();
        u32_le(&mut out, entries.len() as u32);
        for (id, content, flags) in entries {
            u32_le(&mut out, *id);
            u32_le(&mut out, content.len() as u32);
            out.extend_from_slice(content.as_bytes());
            out.extend_from_slice(flags);
        }
        out
    }

    #[test]
    fn parses_synthetic_assets_and_builds_qwen_bpe_tokenizer() {
        let assets = QwenBpeAssets::parse(
            TOKENIZER_QWEN_BPE_V1,
            &vocab_asset(&[
                (0, "H"),
                (1, "e"),
                (2, "l"),
                (3, "o"),
                (4, "He"),
                (5, "Hel"),
                (6, "Hell"),
                (7, "Hello"),
                (8, "<|endoftext|>"),
            ]),
            &merges_asset(&[("H", "e"), ("He", "l"), ("Hel", "l"), ("Hell", "o")]),
            &added_tokens_asset(&[(8, "<|endoftext|>", [0, 0, 0, 0, 1])]),
            br"\S+",
        )
        .expect("parse synthetic assets");

        let tokenizer = build_qwen_bpe_tokenizer(assets).expect("build tokenizer");
        let ids = tokenizer.encode("Hello", false).unwrap().get_ids().to_vec();

        assert_eq!(ids, vec![7]);
        assert_eq!(tokenizer.token_to_id("<|endoftext|>"), Some(8));
    }

    #[test]
    fn rejects_duplicate_vocab_ids() {
        let err = QwenBpeAssets::parse(
            TOKENIZER_QWEN_BPE_V1,
            &vocab_asset(&[(0, "a"), (0, "b")]),
            &merges_asset(&[]),
            &added_tokens_asset(&[]),
            br"\S+",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("duplicate vocab id"), "{err}");
    }

    #[test]
    fn rejects_truncated_merges_asset() {
        let mut merges = merges_asset(&[("a", "b")]);
        merges.pop();

        let err = QwenBpeAssets::parse(
            TOKENIZER_QWEN_BPE_V1,
            &vocab_asset(&[(0, "a"), (1, "b")]),
            &merges,
            &added_tokens_asset(&[]),
            br"\S+",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("tokenizer_merges"), "{err}");
        assert!(
            err.contains("truncated") || err.contains("past end"),
            "{err}"
        );
    }

    #[test]
    fn rejects_bad_added_token_flags() {
        let err = QwenBpeAssets::parse(
            TOKENIZER_QWEN_BPE_V1,
            &vocab_asset(&[(0, "a")]),
            &merges_asset(&[]),
            &added_tokens_asset(&[(0, "a", [0, 2, 0, 1, 0])]),
            br"\S+",
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("added token flag"), "{err}");
    }

    #[test]
    fn rejects_added_token_id_mismatch_for_existing_vocab_token() {
        let assets = QwenBpeAssets::parse(
            TOKENIZER_QWEN_BPE_V1,
            &vocab_asset(&[(0, "a")]),
            &merges_asset(&[]),
            &added_tokens_asset(&[(1, "a", [0, 0, 0, 1, 0])]),
            br"\S+",
        )
        .expect("parse assets");

        let err = build_qwen_bpe_tokenizer(assets).unwrap_err().to_string();

        assert!(err.contains("added token id"), "{err}");
        assert!(err.contains("does not match"), "{err}");
    }
}
