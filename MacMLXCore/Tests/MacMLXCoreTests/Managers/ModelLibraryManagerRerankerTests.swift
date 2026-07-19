import Testing
import Foundation
@testable import MacMLXCore

/// Reranker detection: a cross-encoder shares its `model_type` (`bert` /
/// `xlm-roberta`) with the embedders `ModelLibraryManager` tags `.embedder`,
/// so classification hinges on the `*ForSequenceClassification` head in
/// `config.json`'s `architectures`. These are pure filesystem tests — a temp
/// dir with a hand-crafted `config.json`, no Metal, no model download.
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

    /// The core hardening: a GENUINE multi-class classifier (e.g. a 5-label
    /// sentiment BERT) carries the same `*ForSequenceClassification`
    /// architecture as a reranker, but `num_labels: 5` must disqualify it —
    /// `RerankEngine` always builds a single-logit `Linear(hidden, 1)` head,
    /// so loading a 5-label checkpoint through it would fail `verify: [.all]`
    /// with a cryptic error instead of being cleanly routed elsewhere. `bert`
    /// is a known embedder `model_type`, so disqualification falls through to
    /// `.embedder`.
    @Test
    func explicitNumLabelsFiveStaysEmbedderNotReranker() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "bert-multiclass", modelType: "bert",
            architectures: ["BertForSequenceClassification"], numLabels: 5)
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .embedder)
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

    /// Bugbot #103: a checkpoint that OMITS `num_labels` but declares a
    /// multi-entry `id2label` — a genuine multi-class head such as the 3-way
    /// NLI cross-encoder `nli-deberta-v3-base` — must NOT be taken for a
    /// reranker. The absent `num_labels` must not override the contradicting
    /// `id2label` count (effective label count is 3), so this stays an
    /// `.embedder` (bert model_type) rather than misrouting to `RerankEngine`,
    /// where a `[3, hidden]` classifier head would fail `verify: [.all]`.
    @Test
    func absentNumLabelsWithMultiEntryId2labelStaysEmbedder() async throws {
        let temp = try RerankerTempDir()
        try writeModel(
            in: temp.url, name: "bert-nli", modelType: "bert",
            architectures: ["BertForSequenceClassification"],
            id2label: ["0": "contradiction", "1": "entailment", "2": "neutral"])
        let models = try await ModelLibraryManager().scan(temp.url)
        #expect(models[0].format == .embedder)
    }

    // MARK: - Helpers

    /// Lay down a `.mlx`-shaped directory (tokenizer.json + `.safetensors` +
    /// config.json) whose `config.json` carries `model_type` and, optionally,
    /// `architectures`/`num_labels`/`id2label` — so `upgradeFormat` has
    /// something to classify.
    private func writeModel(
        in root: URL, name: String, modelType: String, architectures: [String]?,
        numLabels: Int? = nil, id2label: [String: String]? = nil
    ) throws {
        let dir = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data("\u{00}".utf8).write(to: dir.appendingPathComponent("model.safetensors"))

        var config: [String: Any] = ["model_type": modelType]
        if let architectures {
            config["architectures"] = architectures
        }
        if let numLabels {
            config["num_labels"] = numLabels
        }
        if let id2label {
            config["id2label"] = id2label
        }
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: dir.appendingPathComponent("config.json"))
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
