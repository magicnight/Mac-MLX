import Foundation
import MLX
import XCTest

@testable import MacMLXCore

/// Real-checkpoint smokes for ``RerankEngine`` (mlx-swift-lm's `MLXRerankers`).
///
/// GATED — never run in CI. Each test self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_RERANK_SMOKE=1`, and
///   3. the checkpoint is on disk. Discovery order per checkpoint:
///        • env `MACMLX_RERANK_ENCODER_MODEL_DIR` / `MACMLX_RERANK_QWEN3_MODEL_DIR`
///          (a directory holding `config.json`), else
///        • the HuggingFace cache snapshot under `~/.cache/huggingface/hub`, else
///        • `~/.mac-mlx/models/<name>`.
///
/// Checkpoints (about 750 MB together):
///   • encoder — `cross-encoder/ms-marco-MiniLM-L-6-v2` (BERT, 91 MB). Scores are
///     compared with a PyTorch fp32 reference captured by
///     `docs/reference/capture_ms_marco_reranker.py` into
///     `Fixtures/rerank_ms_marco_reference.json`, on the logit scale.
///   • qwen3 — `mlx-community/Qwen3-Reranker-0.6B-4bit` (347 MB). Ordering and
///     range only: no reference for the 4-bit conversion is captured in this
///     repository (upstream's integration tests carry bf16 reference margins).
///   • nli — `cross-encoder/nli-MiniLM2-L6-H768` (RoBERTa tokenizer, 3 labels,
///     313 MB; env `MACMLX_RERANK_NLI_MODEL_DIR`). The entailment probabilities
///     are compared with `Fixtures/rerank_nli_reference.json`, captured by
///     `docs/reference/capture_nli_reranker.py`. This is the checkpoint that
///     caught the RoBERTa `addSpecialTokens` bug in `TokenizerBridge`.
///
/// Precision: MLX runs fp32 matmuls as TF32 on M5 by default
/// (`MLX_ENABLE_TF32`, default 1). Under TF32 the ms-marco logits land within
/// 0.025 of PyTorch; with TF32 off they match to 2e-6. The parity test reads
/// the variable and applies the matching gate, so run it with TF32 off for the
/// strict comparison:
///   TEST_RUNNER_MACMLX_RUN_RERANK_SMOKE=1 TEST_RUNNER_MLX_ENABLE_TF32=0 \
///     xcodebuild test-without-building -scheme MacMLXCore \
///     -destination 'platform=macOS' \
///     -only-testing:MacMLXCoreTests/RerankEngineSmokeTests
/// (after `xcodebuild build-for-testing -scheme MacMLXCore -destination 'platform=macOS'`).
final class RerankEngineSmokeTests: XCTestCase {

    private struct Checkpoint {
        let envKey: String
        let hfRepo: String
        let localName: String
    }

    private static let encoder = Checkpoint(
        envKey: "MACMLX_RERANK_ENCODER_MODEL_DIR",
        hfRepo: "cross-encoder/ms-marco-MiniLM-L-6-v2",
        localName: "ms-marco-MiniLM-L-6-v2")

    private static let qwen3 = Checkpoint(
        envKey: "MACMLX_RERANK_QWEN3_MODEL_DIR",
        hfRepo: "mlx-community/Qwen3-Reranker-0.6B-4bit",
        localName: "Qwen3-Reranker-0.6B-4bit")

    /// A 3-way NLI cross-encoder (`roberta`, labels contradiction / entailment /
    /// neutral, 313 MB): the multi-label head upstream scores through its
    /// positive class. Exercises the #131 detection change end to end.
    private static let nli = Checkpoint(
        envKey: "MACMLX_RERANK_NLI_MODEL_DIR",
        hfRepo: "cross-encoder/nli-MiniLM2-L6-H768",
        localName: "nli-MiniLM2-L6-H768")

    /// The query and documents of the PyTorch reference, reused for the Qwen3
    /// ordering check: documents 0 and 2 answer the question, 1 and 3 do not.
    private struct Reference: Decodable {
        struct Pair: Decodable {
            let document: String
            let logit: Double
            let sigmoid: Double
        }
        let model: String
        let query: String
        let pairs: [Pair]
    }

    /// `rerank_nli_reference.json`: three class logits per pair and the
    /// probability of the positive (entailment) class.
    private struct NLIReference: Decodable {
        struct Pair: Decodable {
            let document: String
            let logits: [Double]
            let positive_probability: Double
        }
        let model: String
        let query: String
        let positive_class: Int
        let pairs: [Pair]
    }

    // MARK: - Gating

    private func requireGate() throws {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_RERANK_SMOKE"] == "1" else {
            throw XCTSkip("set MACMLX_RUN_RERANK_SMOKE=1 to run the reranker checkpoint smokes")
        }
    }

    /// Resolve a checkpoint directory: explicit env path → HF cache snapshot →
    /// `~/.mac-mlx/models`. Skips the test when nothing usable is present.
    private func resolve(_ checkpoint: Checkpoint) throws -> URL {
        let fm = FileManager.default
        func hasConfig(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appending(path: "config.json").path)
        }
        if let explicit = ProcessInfo.processInfo.environment[checkpoint.envKey] {
            let dir = URL(fileURLWithPath: explicit, isDirectory: true)
            if hasConfig(dir) { return dir }
            throw XCTSkip("\(checkpoint.envKey) does not point at a directory with config.json")
        }
        let cacheName = "models--" + checkpoint.hfRepo.replacingOccurrences(of: "/", with: "--")
        let snapshots = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".cache/huggingface/hub/\(cacheName)/snapshots", directoryHint: .isDirectory)
        if let entries = try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil),
            let snapshot = entries.sorted(by: { $0.path < $1.path }).first(where: hasConfig)
        {
            return snapshot
        }
        let local = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(checkpoint.localName)", directoryHint: .isDirectory)
        if hasConfig(local) { return local }
        throw XCTSkip("\(checkpoint.hfRepo) is not on disk (HF cache or ~/.mac-mlx/models)")
    }

    private func localModel(id: String, directory: URL) -> LocalModel {
        LocalModel(
            id: id, displayName: id, directory: directory, sizeBytes: 0,
            format: .reranker, quantization: nil, parameterCount: nil, architecture: nil)
    }

    private func loadReference() throws -> Reference {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "rerank_ms_marco_reference", withExtension: "json",
                subdirectory: "Fixtures"),
            "missing fixture rerank_ms_marco_reference.json")
        return try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }

    private func loadNLIReference() throws -> NLIReference {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "rerank_nli_reference", withExtension: "json",
                subdirectory: "Fixtures"),
            "missing fixture rerank_nli_reference.json")
        return try JSONDecoder().decode(NLIReference.self, from: Data(contentsOf: url))
    }

    /// Inverse of the sigmoid `MLXRerankers` applies to a single-logit encoder
    /// head. The sigmoid saturates on this model (0.99986 and 0.000013 for the
    /// reference pairs), so a tolerance on the probability would hide a drift
    /// of whole logits; the comparison is made on the logit scale instead.
    private func logit(_ probability: Double) -> Double {
        log(probability / (1 - probability))
    }

    // MARK: - Encoder cross-encoder (BERT) against the PyTorch reference

    func testMsMarcoCrossEncoderMatchesThePyTorchReference() async throws {
        try requireGate()
        try requireTrustworthyMetalOrSkip()
        let directory = try resolve(Self.encoder)
        let reference = try loadReference()
        XCTAssertEqual(reference.model, Self.encoder.hfRepo)

        let engine = RerankEngine()
        try await engine.load(localModel(id: Self.encoder.localName, directory: directory))
        let kind = await engine.scoreKind
        XCTAssertEqual(kind, .normalizedRelevance, "a single-logit BERT head scores as sigmoid(logit)")

        let scores = try await engine.score(
            query: reference.query, documents: reference.pairs.map(\.document))
        XCTAssertEqual(scores.count, reference.pairs.count)

        // Measured on 2026-10-03 (M5 Max, mlx-swift 0.32.3, mlx-swift-lm 3.32.3):
        // with TF32 off the worst |Δ| is 1.9e-6 — the integration is exact to
        // fp32 rounding; with M5's default TF32 the fp32 matmuls lose mantissa
        // and the worst |Δ| is 0.025 on the +8.85 logit. Gate accordingly: the
        // project's 1e-4 parity bar when TF32 is off, twice the measured TF32
        // drift otherwise. A wrong segment id, pooler or head would miss by
        // whole logits under either.
        let tf32Enabled = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] != "0"
        let tolerance = tf32Enabled ? 0.05 : 1e-4
        var worst = 0.0
        for (score, pair) in zip(scores, reference.pairs) {
            XCTAssert((0.0 ... 1.0).contains(score), "score out of range: \(score)")
            let difference = abs(logit(score) - pair.logit)
            worst = max(worst, difference)
            XCTAssertEqual(
                logit(score), pair.logit, accuracy: tolerance,
                "logit drift on: \(pair.document)")
        }
        print("[rerank-smoke] ms-marco logits (mlx vs torch, TF32 \(tf32Enabled ? "on" : "off")):",
              zip(scores, reference.pairs).map { "\(logit($0.0)) vs \($0.1.logit)" },
              "worst |Δ| = \(worst)")

        // The two documents that answer the question outrank the two that do not.
        let ranked = rankAndTruncate(scores: scores).map(\.index)
        XCTAssertEqual(Set(ranked.prefix(2)), [0, 2])
    }

    // MARK: - Qwen3 causal reranker (yes/no margin): ordering and range

    func testQwen3RerankerRanksTheRelevantDocumentsFirst() async throws {
        try requireGate()
        let directory = try resolve(Self.qwen3)
        let reference = try loadReference()

        let engine = RerankEngine()
        try await engine.load(localModel(id: Self.qwen3.localName, directory: directory))
        let kind = await engine.scoreKind
        XCTAssertEqual(kind, .normalizedRelevance)

        let scores = try await engine.score(
            query: reference.query, documents: reference.pairs.map(\.document))
        XCTAssertEqual(scores.count, reference.pairs.count)
        print("[rerank-smoke] Qwen3-Reranker-0.6B-4bit scores:", scores)

        for score in scores {
            XCTAssert((0.0 ... 1.0).contains(score), "score out of range: \(score)")
        }
        let relevant = [scores[0], scores[2]]
        let unrelated = [scores[1], scores[3]]
        XCTAssertGreaterThan(
            try XCTUnwrap(relevant.min()), try XCTUnwrap(unrelated.max()),
            "every relevant document must outscore every unrelated one")
        XCTAssertGreaterThan(try XCTUnwrap(relevant.min()), 0.5, "a yes answer sits above one half")
        XCTAssertLessThan(try XCTUnwrap(unrelated.max()), 0.5, "a no answer sits below one half")
    }

    // MARK: - Swap order and cache drain (#130)

    /// Two ids on the same ms-marco weights, then Qwen3, then ms-marco again,
    /// through the real server. The old load-then-assign order held two
    /// copies during a swap (peak about twice one copy); releasing first keeps
    /// the swap's peak near one copy. And the swap drains MLX's cache: without
    /// the drain the buffers of the dropped 0.6B model stay parked (around
    /// 600 MB), with it a few MB remain. The same-weights swap cannot show the
    /// drain, since the new load reuses cached buffers of the same sizes, so
    /// the drain is checked on the swap back to ms-marco.
    func testSwapReleasesTheOldEngineBeforeLoadingAndDrainsTheCache() async throws {
        try requireGate()
        let encoderDir = try resolve(Self.encoder)
        let qwenDir = try resolve(Self.qwen3)
        let msA = localModel(id: "ms-a", directory: encoderDir)
        let msB = localModel(id: "ms-b", directory: encoderDir)
        let qwen = localModel(id: "qwen", directory: qwenDir)
        let table = [msA.id: msA, msB.id: msB, qwen.id: qwen]
        let server = HummingbirdServer(
            engine: StubInferenceEngine(engineID: .mlxSwift), modelResolver: { table[$0] })
        let port = try await server.start(preferredPort: 19_780)
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/rerank"))
        func rerank(_ id: String) async throws -> Int {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "model": id,
                "query": "How many people live in Berlin?",
                "documents": ["Berlin has about 3.5 million inhabitants.", "The museum opens at nine."],
            ])
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode ?? -1
        }
        do {
            Memory.clearCache()
            let firstStatus = try await rerank(msA.id)
            XCTAssertEqual(firstStatus, 200)
            let oneCopy = Memory.activeMemory
            Memory.peakMemory = 0   // the setter resets MLX's peak counter
            let swapStatus = try await rerank(msB.id)
            XCTAssertEqual(swapStatus, 200)
            let swapPeak = Memory.peakMemory
            print("[rerank-smoke] swap ms-a→ms-b: one copy active=\(oneCopy) B, peak during swap=\(swapPeak) B")
            XCTAssertLessThan(
                swapPeak, oneCopy * 3 / 2,
                "a swap must not hold two copies of the model (load-then-assign peaks near twice one copy)")

            let qwenStatus = try await rerank(qwen.id)
            XCTAssertEqual(qwenStatus, 200)
            let backStatus = try await rerank(msA.id)
            XCTAssertEqual(backStatus, 200)
            let cached = Memory.cacheMemory
            print("[rerank-smoke] after qwen→ms-a: cache=\(cached) B active=\(Memory.activeMemory) B")
            XCTAssertLessThan(
                cached, 64 << 20,
                "the dropped model's buffers must leave MLX's cache (around 600 MB stay without the drain)")
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()
    // MARK: - Multi-label head scored through its positive class (#131)

    /// The real scanner must classify the NLI checkpoint as a reranker, and the
    /// engine must then score it as the probability of `entailment`, matching
    /// the PyTorch reference. Before #131 this checkpoint fell through to
    /// `.embedder`; and until `TokenizerBridge` started honouring
    /// `addSpecialTokens: false`, the RoBERTa tokenizer wrapped each segment in
    /// `<s> … </s>` twice and the scores were those of a different input
    /// (0.0028 / 0.0008 / 0.0033 / 0.0013 against the reference's
    /// 0.0046 / 0.0004 / 0.0036 / 0.0012 — top-1 flipped). The ordering check
    /// alone passed on the wrong input; the probabilities are what catch it.
    func testMultiLabelNLIHeadIsServedAsAReranker() async throws {
        try requireGate()
        try requireTrustworthyMetalOrSkip()
        let directory = try resolve(Self.nli)
        let reference = try loadNLIReference()
        XCTAssertEqual(reference.model, Self.nli.hfRepo)

        // Detection on the real config.json, through a managed-directory scan
        // of a root that holds just this checkpoint (an APFS clone, so no copy).
        let root = FileManager.default.temporaryDirectory
            .appending(path: "macmlx-nli-smoke-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let clone = Process()
        clone.executableURL = URL(fileURLWithPath: "/bin/cp")
        clone.arguments = ["-RcL", directory.path, root.appending(path: Self.nli.localName).path]
        try clone.run()
        clone.waitUntilExit()
        XCTAssertEqual(clone.terminationStatus, 0, "cloning the checkpoint into the temp root")
        let scanned = try await ModelLibraryManager().scan(root)
        let model = try XCTUnwrap(scanned.first { $0.id == Self.nli.localName })
        XCTAssertEqual(model.format, .reranker, "a multi-label head with an entailment class is a reranker")

        let engine = RerankEngine()
        try await engine.load(model)
        let kind = await engine.scoreKind
        XCTAssertEqual(kind, .normalizedRelevance)
        let scores = try await engine.score(
            query: reference.query, documents: reference.pairs.map(\.document))
        XCTAssertEqual(scores.count, reference.pairs.count)
        // Probabilities here are around 1e-3 (an NLI head measures entailment
        // of a statement by a question, not relevance), so the comparison is
        // relative. Measured on 2026-10-04 (M5 Max): worst relative error
        // 7.4e-6 with TF32 off and 1.8% under M5's default TF32 matmuls, so
        // the gates are the project's 1e-4 bar and 5%. The doubly wrapped
        // input this test exists for was off by 127% on one passage.
        let tf32Enabled = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] != "0"
        let relativeTolerance = tf32Enabled ? 0.05 : 1e-4
        var worst = 0.0
        for (score, pair) in zip(scores, reference.pairs) {
            XCTAssert((0.0 ... 1.0).contains(score), "score out of range: \(score)")
            let relative = abs(score - pair.positive_probability) / pair.positive_probability
            worst = max(worst, relative)
            XCTAssertLessThanOrEqual(
                relative, relativeTolerance,
                "P(entailment) \(score) vs reference \(pair.positive_probability) on: \(pair.document)")
        }
        print("[rerank-smoke] nli-MiniLM2-L6-H768 P(entailment) (mlx vs torch, TF32 \(tf32Enabled ? "on" : "off")):",
              zip(scores, reference.pairs).map { "\($0.0) vs \($0.1.positive_probability)" },
              "worst relative = \(worst)")
        let relevant = [scores[0], scores[2]]
        let unrelated = [scores[1], scores[3]]
        XCTAssertGreaterThan(
            try XCTUnwrap(relevant.min()), try XCTUnwrap(unrelated.max()),
            "entailment probability must rank the answering passages first")
    }

    // MARK: - End to end through POST /v1/rerank

    func testRerankEndpointServesTheEncoderCheckpoint() async throws {
        try requireGate()
        let directory = try resolve(Self.encoder)
        let reference = try loadReference()
        let model = localModel(id: "ms-marco", directory: directory)
        let resolver: HummingbirdServer.ModelResolver = { $0 == "ms-marco" ? model : nil }
        let server = HummingbirdServer(
            engine: StubInferenceEngine(engineID: .mlxSwift), modelResolver: resolver)
        let port = try await server.start(preferredPort: 19_690)

        var request = URLRequest(url: try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/rerank")))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "ms-marco",
            "query": reference.query,
            "documents": reference.pairs.map(\.document),
            "top_n": 2,
            "return_documents": true,
        ] as [String: Any])
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()

        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200, String(decoding: data, as: UTF8.self))
        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["model"] as? String, "ms-marco")
        let results = try XCTUnwrap(json["results"] as? [[String: Any]])
        XCTAssertEqual(results.count, 2, "top_n truncates")
        let indices = results.compactMap { $0["index"] as? Int }
        XCTAssertEqual(Set(indices), [0, 2], "the two Berlin-population documents rank first")
        for result in results {
            let score = try XCTUnwrap(result["relevance_score"] as? Double)
            XCTAssertGreaterThan(score, 0.99, "sigmoid of a logit near +9")
            XCTAssertLessThanOrEqual(score, 1.0)
            let index = try XCTUnwrap(result["index"] as? Int)
            XCTAssertEqual(result["document"] as? String, reference.pairs[index].document)
        }
        let first = try XCTUnwrap(results.first?["relevance_score"] as? Double)
        let second = try XCTUnwrap(results.last?["relevance_score"] as? Double)
        XCTAssertGreaterThanOrEqual(first, second, "results are ordered by descending relevance")
    }
}
