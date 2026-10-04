import Testing
import Foundation
@testable import MacMLXCore

/// Reranker detection. An encoder cross-encoder shares its `model_type`
/// (`bert` / `xlm-roberta`) with the embedders `ModelLibraryManager` tags
/// `.embedder`, so that family hinges on the single-logit
/// `*ForSequenceClassification` head in `config.json`'s `architectures`; a
/// Qwen3 reranker is config-identical to a chat model, so its family hinges
/// on the `Qwen3-Reranker` name and its logit-score declaration; Jina
/// reranker v3 declares `JinaForRanking`. These are pure filesystem tests — a
/// temp dir with a hand-crafted `config.json`, no Metal, no model download.
///
/// Serialised for the same reason as the embedder/VLM suites (parallel tmpdir
/// thrash + actor scans trip a Swift-stdlib flake).
@Suite("ModelLibraryManager reranker detection", .serialized)
struct ModelLibraryManagerRerankerTests {

    @Test
    func bertSequenceClassificationDetectedAsReranker() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "bert-rerank", modelType: "bert",
            architectures: ["BertForSequenceClassification"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models.count == 1)
        // Reranker wins over `.embedder` even though `bert` is an embedder type.
        #expect(models[0].format == .reranker)
    }

    @Test
    func xlmRobertaSequenceClassificationDetectedAsReranker() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "xlmr-rerank", modelType: "xlm-roberta",
            architectures: ["XLMRobertaForSequenceClassification"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .reranker)
    }

    @Test
    func bertWithoutSequenceClassificationStaysEmbedder() async throws {
        // Same `model_type` bert, but no classification head → still an embedder.
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "bert-embed", modelType: "bert",
            architectures: ["BertModel"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .embedder)
    }

    @Test
    func bertWithNoArchitecturesStaysEmbedder() async throws {
        let temp = try RerankerTempDir()
        try writeModel(in: temp.url, name: "bert-plain", modelType: "bert", architectures: nil)
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .embedder)
    }

    @Test
    func causalLMSequenceHeadlessStaysMLX() async throws {
        // A causal LM's `…ForCausalLM` architecture is NOT a reranker, and
        // `qwen3` isn't an embedder type → plain `.mlx`.
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "qwen3-chat", modelType: "qwen3",
            architectures: ["Qwen3ForCausalLM"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .mlx)
    }

    // MARK: - Qwen3 causal rerankers (name rule) and Jina reranker v3

    /// `Qwen3-Reranker-*` ships a `config.json` identical to a Qwen3 chat
    /// model's, so the name is the signal: `qwen3` + `Qwen3ForCausalLM` + a
    /// repo name that reads `Qwen3-Reranker` (separators and case ignored).
    @Test
    func qwen3CausalLMNamedRerankerDetectedAsReranker() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "Qwen3-Reranker-0.6B-4bit", modelType: "qwen3",
            architectures: ["Qwen3ForCausalLM"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .reranker)
    }

    @Test
    func qwen3RerankerNameRuleIgnoresCaseAndSeparators() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "qwen3_RERANKER-8b-mlx", modelType: "qwen3",
            architectures: ["Qwen3ForCausalLM"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .reranker)
    }

    /// Other Qwen3-based rerankers use a different scoring protocol than the
    /// yes/no margin `MLXRerankers` implements, so they must NOT be routed:
    /// `zeroentropy/zerank-2-reranker` and `ContextualAI/ctxl-rerank-v2-…`
    /// both carry `qwen3` + `Qwen3ForCausalLM` and "rerank" in the name — the
    /// rule upstream applies would take them — but their names do not read
    /// `Qwen3-Reranker`. They stay `.mlx`, where `/v1/rerank` answers 400.
    @Test
    func otherQwen3BasedRerankersStayMLX() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "zerank-2-reranker", modelType: "qwen3",
            architectures: ["Qwen3ForCausalLM"],
            logitScore: ["true_token_id": 9454, "false_token_id": NSNull()])
        try writeModel(
            in: temp.url, name: "ctxl-rerank-v2-instruct-multilingual-1b", modelType: "qwen3",
            architectures: ["Qwen3ForCausalLM"], id2label: ["0": "LABEL_0"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models.count == 2)
        #expect(models.allSatisfy { $0.format == .mlx })
    }

    /// A checkpoint that passes the name rule but ships a
    /// `1_LogitScore/config.json` with no `false_token_id` uses the
    /// single-token protocol, not yes/no: not a reranker for us. One that
    /// declares both tokens — the official `Qwen/Qwen3-Reranker-*` layout — is.
    @Test
    func qwen3LogitScoreMustDeclareBothTokens() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "Qwen3-Reranker-single-logit", modelType: "qwen3",
            architectures: ["Qwen3ForCausalLM"],
            logitScore: ["true_token_id": 9454, "false_token_id": NSNull()])
        try writeModel(
            in: temp.url, name: "Qwen3-Reranker-4B", modelType: "qwen3",
            architectures: ["Qwen3ForCausalLM"],
            logitScore: ["true_token_id": 9693, "false_token_id": 2152])
        let models = try await ModelLibraryManager().scan(temp.url)
        let byName = Dictionary(uniqueKeysWithValues: models.map { ($0.id, $0.format) })
        #expect(byName["Qwen3-Reranker-single-logit"] == .mlx)
        #expect(byName["Qwen3-Reranker-4B"] == .reranker)
    }

    /// The name rule is bound to the `qwen3` model type: the same
    /// architecture under another type is not this family.
    @Test
    func qwen3NameRuleRequiresTheQwen3ModelType() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "Qwen3-Reranker-0.6B", modelType: "llama",
            architectures: ["Qwen3ForCausalLM"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .mlx)
    }

    /// The name rule is scoped to the Qwen3 causal shape: "rerank" in the
    /// name of a plain BERT encoder changes nothing (it needs the
    /// classification head), and a Qwen3 config without `Qwen3ForCausalLM`
    /// stays a chat model even when named like a reranker.
    @Test
    func rerankInTheNameAloneDoesNotReclassifyOtherShapes() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "bge-reranker-lookalike", modelType: "bert",
            architectures: ["BertModel"])
        try writeModel(
            in: temp.url, name: "qwen3-reranker-no-arch", modelType: "qwen3",
            architectures: nil)
        let models = try await ModelLibraryManager().scan(temp.url)
        let byName = Dictionary(uniqueKeysWithValues: models.map { ($0.id, $0.format) })
        #expect(byName["bge-reranker-lookalike"] == .embedder)
        #expect(byName["qwen3-reranker-no-arch"] == .mlx)
    }

    /// Jina reranker v3 declares `JinaForRanking`; its `model_type` is `qwen3`,
    /// and nothing in its name is required.
    @Test
    func jinaForRankingDetectedAsRerankerRegardlessOfName() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "jina-v3", modelType: "qwen3",
            architectures: ["JinaForRanking"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .reranker)
    }

    /// Jina reranker v3.5 interleaves sliding-window layers that upstream's
    /// Qwen3 model ignores; scoring it would be silently wrong past 1,024
    /// tokens, so it is not routed — by `layer_types` or by `use_sliding_window`.
    @Test
    func jinaWithSlidingWindowLayersIsNotRouted() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "jina-reranker-v3.5", modelType: "qwen3",
            architectures: ["JinaForRanking"],
            extra: ["layer_types": ["sliding_attention", "full_attention"], "sliding_window": 1024])
        try writeModel(
            in: temp.url, name: "jina-sliding-flag", modelType: "qwen3",
            architectures: ["JinaForRanking"],
            extra: ["use_sliding_window": true])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models.count == 2)
        #expect(models.allSatisfy { $0.format == .mlx })
    }

    /// Only the repo part of a Hub id is matched: an org whose name reads
    /// `Qwen3-Reranker` does not make its Qwen3 chat model a reranker.
    @Test
    func qwen3RerankerInTheOrgNameDoesNotCount() async throws {
        let temp = try RerankerTempDir()
        let snapshot = temp.url
            .appendingPathComponent("models--qwen3-reranker-lab--Qwen3-8B")
            .appendingPathComponent("snapshots")
            .appendingPathComponent("abc123")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: snapshot.appendingPathComponent("tokenizer.json"))
        try Data("\u{00}".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
        let config: [String: Any] = ["model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: snapshot.appendingPathComponent("config.json"))
        let models = await ModelLibraryManager().scanHuggingFaceCache(directories: [temp.url])
        #expect(models.first?.id == "qwen3-reranker-lab/Qwen3-8B")
        #expect(models.first?.format == .mlx)
    }

    /// In the HuggingFace cache the directory is `snapshots/<sha>`, which
    /// carries no name; the repo id (`models--mlx-community--Qwen3-Reranker-…`)
    /// must drive the Qwen3 rule there.
    @Test
    func qwen3RerankerInHuggingFaceCacheDetectedByRepoID() async throws {
        let temp = try RerankerTempDir()
        let snapshot = temp.url
            .appendingPathComponent("models--mlx-community--Qwen3-Reranker-0.6B-4bit")
            .appendingPathComponent("snapshots")
            .appendingPathComponent("5f32454")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: snapshot.appendingPathComponent("tokenizer.json"))
        try Data("\u{00}".utf8).write(to: snapshot.appendingPathComponent("model.safetensors"))
        let config: [String: Any] = ["model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: snapshot.appendingPathComponent("config.json"))
        let models = await ModelLibraryManager().scanHuggingFaceCache(directories: [temp.url])
        #expect(models.count == 1)
        #expect(models.first?.id == "mlx-community/Qwen3-Reranker-0.6B-4bit")
        #expect(models.first?.format == .reranker)
    }

    @Test
    func lastArchitectureDecidesReranker() async throws {
        // HF lists the concrete task head last; a trailing
        // `…ForSequenceClassification` classifies as reranker.
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "multi-arch", modelType: "bert",
            architectures: ["BertModel", "BertForSequenceClassification"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .reranker)
    }

    // MARK: - num_labels / id2label gating (multi-class disqualifier)

    @Test
    func explicitNumLabelsOneDetectedAsReranker() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "bert-rerank-explicit", modelType: "bert",
            architectures: ["BertForSequenceClassification"], numLabels: 1)
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .reranker)
    }

    /// A GENUINE multi-class classifier (a 5-label sentiment BERT, labels
    /// unnamed) carries the same `*ForSequenceClassification` architecture as
    /// a reranker. `MLXRerankers` rejects such a head as ambiguous, and a
    /// classifier is not a sentence embedder, so it is neither: plain `.mlx`,
    /// never `.embedder` — `/v1/embeddings` used to pool its hidden states
    /// into 19,968-dimensional "vectors" (#131).
    @Test
    func multiLabelHeadWithoutAPositiveClassIsNeitherRerankerNorEmbedder() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "bert-multiclass", modelType: "bert",
            architectures: ["BertForSequenceClassification"], numLabels: 5)
        try writeModel(
            in: temp.url, name: "bert-sentiment", modelType: "bert",
            architectures: ["BertForSequenceClassification"],
            id2label: ["0": "very negative", "1": "negative", "2": "neutral", "3": "happy", "4": "very happy"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models.count == 2)
        #expect(models.allSatisfy { $0.format == .mlx })
    }

    /// A classification head upstream has no encoder for — `electra`
    /// (`cross-encoder/ms-marco-electra-base`), a `Qwen3ForSequenceClassification`
    /// conversion (`tomaarsen/Qwen3-Reranker-0.6B-seq-cls`), or a config with no
    /// `model_type` (`jinaai/jina-reranker-v2-base-multilingual`) — used to get
    /// the Rerank badge and then a 500 from the factory (#131). Now `.mlx`.
    @Test
    func classificationHeadOutsideTheThreeEncoderTypesIsNotARerankerBadge() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "ms-marco-electra-base", modelType: "electra",
            architectures: ["ElectraForSequenceClassification"], id2label: ["0": "LABEL_0"])
        try writeModel(
            in: temp.url, name: "Qwen3-Reranker-0.6B-seq-cls", modelType: "qwen3",
            architectures: ["Qwen3ForSequenceClassification"], id2label: ["0": "LABEL_0"])
        try writeModel(
            in: temp.url, name: "jina-reranker-v2-base-multilingual", modelType: nil,
            architectures: ["XLMRobertaForSequenceClassification"], numLabels: 1)
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models.count == 3)
        #expect(models.allSatisfy { $0.format == .mlx })
    }

    /// `roberta` is the third encoder type upstream builds the head for.
    @Test
    func robertaSequenceClassificationDetectedAsReranker() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "roberta-rerank", modelType: "roberta",
            architectures: ["RobertaForSequenceClassification"], numLabels: 1)
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .reranker)
    }

    /// `distilbert` and `nomic_bert` are embedder types but upstream gives them
    /// no classification head, so a `*ForSequenceClassification` checkpoint of
    /// theirs is neither a reranker nor an embedder.
    @Test
    func distilbertClassificationHeadIsNeither() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "distilbert-rerank", modelType: "distilbert",
            architectures: ["DistilBertForSequenceClassification"], numLabels: 1)
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .mlx)
    }

    /// `id2label`'s entry count is the second, independent single-label
    /// signal — a checkpoint that omits `num_labels` but populates
    /// `id2label` with exactly one entry is still a reranker.
    @Test
    func id2labelSingleEntryDetectedAsReranker() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "bert-rerank-id2label", modelType: "bert",
            architectures: ["BertForSequenceClassification"],
            id2label: ["0": "relevant"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .reranker)
    }

    /// A 3-way NLI head (`cross-encoder/nli-MiniLM2-L6-H768`: contradiction /
    /// entailment / neutral) names a positive class upstream recognizes, so
    /// `MLXRerankers` scores it as the softmax probability of `entailment`:
    /// `.reranker`, with `num_labels` absent and the count coming from the
    /// `id2label` keys as upstream derives it. Until #131 this fell through to
    /// `.embedder` and `/v1/embeddings` served flattened hidden states.
    @Test
    func multiLabelHeadWithAPositiveClassIsAReranker() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "nli-MiniLM2-L6-H768", modelType: "roberta",
            architectures: ["RobertaForSequenceClassification"],
            id2label: ["0": "contradiction", "1": "entailment", "2": "neutral"])
        try writeModel(
            in: temp.url, name: "bert-binary-relevance", modelType: "bert",
            architectures: ["BertForSequenceClassification"], numLabels: 2,
            extra: ["label2id": ["irrelevant": 0, "Relevant": 1]])
        try writeModel(
            in: temp.url, name: "bert-label1", modelType: "bert",
            architectures: ["BertForSequenceClassification"],
            id2label: ["0": "LABEL_0", "1": "LABEL_1"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models.count == 3)
        #expect(models.allSatisfy { $0.format == .reranker })
    }

    // MARK: - Helpers

    /// Lay down a `.mlx`-shaped directory (tokenizer.json + `.safetensors` +
    /// config.json) whose `config.json` carries `model_type` and, optionally,
    /// `architectures`/`num_labels`/`id2label` — so `upgradeFormat` has
    /// something to classify.
    private func writeModel(
        in root: URL, name: String, modelType: String?, architectures: [String]?,
        numLabels: Int? = nil, id2label: [String: String]? = nil,
        extra: [String: Any] = [:], logitScore: [String: Any]? = nil
    ) throws {
        let dir = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data("\u{00}".utf8).write(to: dir.appendingPathComponent("model.safetensors"))

        var config: [String: Any] = [:]
        if let modelType {
            config["model_type"] = modelType
        }
        if let architectures {
            config["architectures"] = architectures
        }
        if let numLabels {
            config["num_labels"] = numLabels
        }
        if let id2label {
            config["id2label"] = id2label
        }
        for (key, value) in extra {
            config[key] = value
        }
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: dir.appendingPathComponent("config.json"))

        // Sentence-Transformers' scoring-head declaration, as the official
        // Qwen3-Reranker checkpoints ship it.
        if let logitScore {
            let scoreDir = dir.appendingPathComponent("1_LogitScore")
            try FileManager.default.createDirectory(at: scoreDir, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: logitScore)
                .write(to: scoreDir.appendingPathComponent("config.json"))
        }
    }
}

/// Auto-cleaning temp directory for the reranker detection tests.
private struct RerankerTempDir {
    let url: URL

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("macmlx-reranker-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        self.url = base
    }
}
