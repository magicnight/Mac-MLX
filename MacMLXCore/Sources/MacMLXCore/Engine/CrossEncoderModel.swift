import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXNN

// MARK: - CrossEncoderModel

/// A TRUE cross-encoder reranker: a BERT encoder + a single-logit sequence
/// classification head that JOINTLY scores a `[query, document]` pair (both
/// segments in one forward pass, attending across the `[SEP]` boundary) and
/// emits one relevance logit per pair. This is the architecture behind
/// `cross-encoder/ms-marco-MiniLM-L-6-v2` and every HF
/// `BertForSequenceClassification` reranker.
///
/// Contrast with the bi-encoder cosine MVP (`rerankByCosine`), which embeds
/// the query and each document INDEPENDENTLY and compares vectors — it never
/// lets a document's tokens attend to the query's, so it can't model the
/// fine-grained query↔document interactions a cross-encoder captures. The
/// cross-encoder is the accuracy-first path; the bi-encoder stays as the
/// documented fallback for `.embedder` models.
///
/// ## Why this is a CONTAINED feature, not a new model port
/// `MLXEmbedders` already ships the entire BERT encoder + pooler
/// (`BertModel`, built with `lmHead: false` so it installs the pooler and
/// returns `pooledOutput = tanh(pooler(hidden[:, 0]))` — exactly the tensor
/// HF feeds its classification head). All this type adds is the final
/// `Linear(hidden, 1)` head and the weight-key plumbing to load a
/// `bert.* + classifier.*` checkpoint. It reuses the encoder verbatim rather
/// than re-porting it.
///
/// ## Why `Module, BaseLanguageModel` (and nothing more)
/// `BaseLanguageModel` requires ONLY `sanitize(weights:)` — it does NOT drag
/// in the token-generation surface (`prepare`, `newCache`, the `LMOutput`
/// forward) that `LanguageModel` does. Conforming lets this model flow
/// through the SAME strict loader the LLMs and embedders use,
/// `loadWeights(modelDirectory:model:)`, whose final step is
/// `model.update(parameters:, verify: [.all])`. `verify: [.all]` demands an
/// EXACT bijection between the module's parameter tree and the (sanitized)
/// checkpoint keys — no missing params, no extras. That strictness is the
/// whole reason `sanitize` below and the submodule keys must mirror the
/// checkpoint 1:1: a single stray or missing key aborts the load loudly
/// instead of silently loading garbage.
///
/// ## First-cut scope (deferred-validation notes)
/// - fp16 / fp32 only — the quantization branch is skipped, so a quantized
///   reranker checkpoint (carrying `.scales` / `.biases`) is NOT yet
///   supported (its extra keys would fail `verify: [.all]`).
/// - Standard BERT weight naming only (as `ms-marco-MiniLM-L-6-v2` uses).
///   A DistilBERT-based cross-encoder needs the `DistilBertModel` rename
///   table instead; an XLM-R / RoBERTa cross-encoder needs a different
///   encoder entirely. Both are deferred.
/// - Architecture-faithful but NOT numerically validated against a real
///   checkpoint in this environment (no model download / Metal here); a
///   parity fixture + real-checkpoint smoke are deferred to when test
///   conditions exist.
final class CrossEncoderModel: Module, BaseLanguageModel {

    /// The BERT encoder + pooler, keyed `bert` so its parameters live under
    /// `bert.*` — matching an HF `BertForSequenceClassification` checkpoint,
    /// whose backbone is exactly `self.bert = BertModel(config)`. Built with
    /// `lmHead: false` so it carries a pooler (and thus a non-nil
    /// `pooledOutput`), never the masked-LM head.
    @ModuleInfo(key: "bert") var bert: BertModel

    /// The sequence-classification head: `Linear(hidden, 1)` producing one
    /// relevance logit. Keyed `classifier` to match HF's
    /// `BertForSequenceClassification.classifier` (`num_labels == 1` for a
    /// reranker). Carries a bias, as HF's does.
    @ModuleInfo(key: "classifier") var classifier: Linear

    /// - Parameters:
    ///   - bertConfiguration: Drives the encoder shape. Decoded from the
    ///     checkpoint's `config.json`.
    ///   - hiddenSize: The classifier's input dimension (`config.hidden_size`).
    ///     Passed EXPLICITLY rather than read off `bertConfiguration` because
    ///     `BertConfiguration.embedDim` is internal to `MLXEmbedders` and not
    ///     reachable from here — callers read `hidden_size` from the same
    ///     `config.json` so the two stay consistent. It MUST equal the BERT
    ///     hidden size, or the constructed `classifier.weight` shape
    ///     (`[1, hiddenSize]`) won't match the checkpoint's and
    ///     `verify: [.all]` will reject the load.
    init(bertConfiguration: BertConfiguration, hiddenSize: Int) {
        self._bert.wrappedValue = BertModel(bertConfiguration, lmHead: false)
        self._classifier.wrappedValue = Linear(hiddenSize, 1)
    }

    /// Joint forward pass over a batch of pre-tokenized `[query, doc]` pairs.
    ///
    /// - Parameters:
    ///   - inputIds: `[batch, seqLen]` token ids (`[CLS] q [SEP] d [SEP]`,
    ///     padded to the batch's longest).
    ///   - tokenTypeIds: `[batch, seqLen]` segment ids — 0 over
    ///     `[CLS] query [SEP]`, 1 over `doc [SEP]`. This is what makes it a
    ///     PAIR scorer: the encoder learns which span is the query and which
    ///     is the document from these ids.
    ///   - attentionMask: `[batch, seqLen]` 1 for real tokens, 0 for padding.
    /// - Returns: `[batch]` raw relevance logits (higher = more relevant),
    ///   one per pair. The caller (`RerankEngine`) ranks by these and exposes
    ///   `sigmoid(logit)` as the bounded `relevance_score`.
    func callAsFunction(
        _ inputIds: MLXArray,
        tokenTypeIds: MLXArray,
        attentionMask: MLXArray
    ) -> MLXArray {
        let out = bert(
            inputIds, positionIds: nil, tokenTypeIds: tokenTypeIds,
            attentionMask: attentionMask)
        guard let pooled = out.pooledOutput else {
            // Unreachable: `bert` is built with `lmHead: false`, which always
            // installs a pooler, so `pooledOutput` is always non-nil. A
            // `guard let` (not `!`) honors the project's no-force-unwrap rule;
            // a degenerate zero-vector beats a crash if the invariant ever
            // breaks. Shape `[batch]` mirrors the real return.
            return MLXArray.zeros(like: attentionMask.sum(axis: -1)).asType(.float32)
        }
        // pooled: [batch, hidden] → classifier → [batch, 1]. Column 0 is the
        // single relevance logit; `[0..., 0]` squeezes it to [batch] (the same
        // CLS-slice idiom `BertModel` uses for `outputs[0..., 0]`).
        let logits = classifier(pooled)
        return logits[0..., 0]
    }

    /// Rewrite raw HF checkpoint keys into this module's parameter layout so
    /// `loadWeights`' strict `verify: [.all]` update accepts them.
    ///
    /// This is `BertModel.sanitize`'s rename table applied to the
    /// `bert.`-prefixed keys — with ONE deliberate omission: `BertModel`,
    /// when loaded standalone, strips its own `bert.` prefix (its params live
    /// at `embeddings.*`, `encoder.*`, `pooler.*`). Here `BertModel` is a
    /// SUBMODULE at key `bert`, so its params live at `bert.embeddings.*`,
    /// `bert.encoder.*`, `bert.pooler.*` — we must KEEP the `bert.` prefix,
    /// hence we drop the `"bert." → ""` step. Every other substring the table
    /// rewrites (`.layer.` → `.layers.`, `.self.query.` → `.query_proj.`,
    /// `pooler.dense.` → `pooler.`, …) is independent of that prefix, so the
    /// bert-subtree keys land exactly where the submodule expects them while
    /// `classifier.weight` / `classifier.bias` — matching no rewrite substring
    /// — pass through verbatim.
    ///
    /// The `cls.predictions.* → lm_head.*` rules are inert for a
    /// `*ForSequenceClassification` checkpoint (which has no masked-LM head)
    /// but are kept so this stays a faithful 1:1 mirror of `BertModel.sanitize`
    /// — if upstream BERT weight naming shifts, both move together.
    ///
    /// Finally two persisted, non-learned buffers HF sometimes ships that the
    /// MLX `BertModel` doesn't declare as parameters —
    /// `bert.embeddings.position_ids` (this encoder generates positions
    /// dynamically) and `bert.embeddings.token_type_ids` (a default-segment
    /// buffer; real segment ids are always supplied explicitly per pair) —
    /// are dropped. Leaving either in would be an unexpected key under
    /// `verify: [.all]`. Deliberately narrow: only these two known embedding
    /// buffers are dropped, not any non-float or unrecognized key.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.reduce(into: [:]) { result, item in
            if let key = Self.sanitizedKey(item.key) {
                result[key] = item.value
            }
        }
    }

    /// Rewrite ONE checkpoint weight key into this module's parameter path, or
    /// return `nil` if the key must be dropped (`bert.embeddings.position_ids`
    /// or `bert.embeddings.token_type_ids` — see `sanitize(weights:)`).
    ///
    /// Split out as a PURE `static` string function (no `MLXArray`, no model
    /// instance, no MLX runtime) so the rename table — the load-bearing,
    /// most breakage-prone part of the port — is unit-testable ungated (no
    /// Metal). See `sanitize(weights:)` for why the `bert.` prefix is kept and
    /// why the `cls.predictions.*` rules stay.
    static func sanitizedKey(_ key: String) -> String? {
        if key == "bert.embeddings.position_ids" { return nil }
        if key == "bert.embeddings.token_type_ids" { return nil }
        return key
            .replacingOccurrences(of: ".layer.", with: ".layers.")
            .replacingOccurrences(of: ".self.key.", with: ".key_proj.")
            .replacingOccurrences(of: ".self.query.", with: ".query_proj.")
            .replacingOccurrences(of: ".self.value.", with: ".value_proj.")
            .replacingOccurrences(of: ".attention.output.dense.", with: ".attention.out_proj.")
            .replacingOccurrences(of: ".attention.output.LayerNorm.", with: ".ln1.")
            .replacingOccurrences(of: ".output.LayerNorm.", with: ".ln2.")
            .replacingOccurrences(of: ".intermediate.dense.", with: ".linear1.")
            .replacingOccurrences(of: ".output.dense.", with: ".linear2.")
            .replacingOccurrences(of: ".LayerNorm.", with: ".norm.")
            .replacingOccurrences(of: "pooler.dense.", with: "pooler.")
            .replacingOccurrences(of: "cls.predictions.transform.dense.", with: "lm_head.dense.")
            .replacingOccurrences(of: "cls.predictions.transform.LayerNorm.", with: "lm_head.ln.")
            .replacingOccurrences(of: "cls.predictions.decoder", with: "lm_head.decoder")
            .replacingOccurrences(of: "cls.predictions.transform.norm.weight", with: "lm_head.ln.weight")
            .replacingOccurrences(of: "cls.predictions.transform.norm.bias", with: "lm_head.ln.bias")
            .replacingOccurrences(of: "cls.predictions.bias", with: "lm_head.decoder.bias")
    }
}
