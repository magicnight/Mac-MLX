import Foundation
import MLX
import XCTest
import os

@testable import MacMLXCore

/// A pool stand-in for the residency smokes: records what the server asks,
/// keeps the releases it is handed, and can run them the way the pool would
/// when it reclaims an entry.
private actor ReleasingResidency: AuxiliaryResidency {
    private(set) var events: [String] = []
    private var releases: [String: @Sendable () async -> Void] = [:]
    /// When set, the next admission runs its release before it returns: a
    /// pool that let the model go in the window between recording the entry
    /// and the server installing the engine (on a real pool an `unload` of
    /// the key, since an admission in use is neither evicted nor swept).
    private var releaseNextAdmissionAtOnce = false
    /// When set, the next `beginUse` answers false: a pool that evicted or
    /// swept the model and whose release has not reached the server yet.
    private var refuseNextUse = false
    /// When set, the next `beginUse` runs the model's release first, then
    /// answers false: the release reached the server while the request was
    /// suspended, before the refusal did.
    private var releaseBeforeRefusingNextUse = false

    func releaseTheNextAdmissionAtOnce() { releaseNextAdmissionAtOnce = true }

    func letGoOfTheNextUse() { refuseNextUse = true }

    func letGoOfTheNextUseReleasingFirst() { releaseBeforeRefusingNextUse = true }

    func admit(
        _ kind: AuxiliaryKind,
        _ model: LocalModel,
        load: @Sendable () async throws -> Void,
        release: @escaping @Sendable () async -> Void
    ) async throws {
        let key = ModelPool.auxiliaryKey(kind, model.id)
        events.append("admit \(key)")
        try await load()
        if releaseNextAdmissionAtOnce {
            releaseNextAdmissionAtOnce = false
            events.append("release \(key) before install")
            await release()
            return
        }
        releases[key] = release
    }

    func beginUse(_ kind: AuxiliaryKind, _ modelID: String) async -> Bool {
        let key = ModelPool.auxiliaryKey(kind, modelID)
        if releaseBeforeRefusingNextUse {
            releaseBeforeRefusingNextUse = false
            if let release = releases.removeValue(forKey: key) {
                events.append("release \(key)")
                await release()
            }
            events.append("use refused \(key)")
            return false
        }
        if refuseNextUse {
            refuseNextUse = false
            events.append("use refused \(key)")
            return false
        }
        events.append("use \(key)")
        return true
    }

    func endUse(_ kind: AuxiliaryKind, _ modelID: String) async {
        events.append("end \(ModelPool.auxiliaryKey(kind, modelID))")
    }

    func forget(_ kind: AuxiliaryKind, _ modelID: String) async {
        let key = ModelPool.auxiliaryKey(kind, modelID)
        events.append("forget \(key)")
        releases.removeValue(forKey: key)
    }

    /// What the pool does when it evicts or sweeps: run the release.
    func reclaim(_ kind: AuxiliaryKind, _ modelID: String) async {
        let key = ModelPool.auxiliaryKey(kind, modelID)
        guard let release = releases.removeValue(forKey: key) else { return }
        events.append("release \(key)")
        await release()
    }

    /// The release handed over for a model, kept by the caller: a pool
    /// running a release after the server has already swapped the model out.
    func release(for kind: AuxiliaryKind, _ modelID: String) -> (@Sendable () async -> Void)? {
        releases[ModelPool.auxiliaryKey(kind, modelID)]
    }
}

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
            return try assembled(snapshot, weightsFrom: entries)
        }
        let local = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(checkpoint.localName)", directoryHint: .isDirectory)
        if hasConfig(local) { return local }
        throw XCTSkip("\(checkpoint.hfRepo) is not on disk (HF cache or ~/.mac-mlx/models)")
    }

    /// The HF cache files each download under the snapshot of the revision
    /// it was fetched at, so a repository whose weights were fetched by a
    /// separate call can have `config.json` in one snapshot and
    /// `model.safetensors` in another (two of the embedders on the machine
    /// this was written on). When the config snapshot has no weights and a
    /// sibling has them, link both into one directory under a temporary
    /// root, removed when the test ends.
    private func assembled(_ snapshot: URL, weightsFrom siblings: [URL]) throws -> URL {
        let fm = FileManager.default
        let weights = "model.safetensors"
        func hasWeights(_ dir: URL) -> Bool { fm.fileExists(atPath: dir.appending(path: weights).path) }
        if hasWeights(snapshot) { return snapshot }
        guard let donor = siblings.first(where: { $0 != snapshot && hasWeights($0) }) else { return snapshot }
        let root = fm.temporaryDirectory
            .appending(path: "macmlx-snapshot-\(UUID().uuidString)", directoryHint: .isDirectory)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        for entry in try fm.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: nil)
        where entry.lastPathComponent != weights {
            try fm.createSymbolicLink(
                at: root.appending(path: entry.lastPathComponent), withDestinationURL: entry)
        }
        try fm.createSymbolicLink(
            at: root.appending(path: weights), withDestinationURL: donor.appending(path: weights))
        return root
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
    }

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

    // MARK: - Residency reported to the pool (#130)

    private static let embedder = Checkpoint(
        envKey: "MACMLX_EMBEDDER_MODEL_DIR",
        hfRepo: "BAAI/bge-small-zh-v1.5",
        localName: "bge-small-zh-v1.5")

    private static let embedderB = Checkpoint(
        envKey: "MACMLX_EMBEDDER_B_MODEL_DIR",
        hfRepo: "BAAI/bge-base-zh-v1.5",
        localName: "bge-base-zh-v1.5")

    private struct ResidencyServer {
        let server: HummingbirdServer
        let residency: ReleasingResidency
        let port: Int
        /// What the server resolves ids to; a test may swap an entry out.
        let models: OSAllocatedUnfairLock<[String: LocalModel]>

        func post(_ path: String, _ body: [String: Any]) async throws -> (Int, [String: Any]) {
            var request = URLRequest(url: try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)\(path)")))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, response) = try await URLSession.shared.data(for: request)
            let json = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            return ((response as? HTTPURLResponse)?.statusCode ?? -1, json)
        }

        func status() async throws -> [String: Any] {
            let (data, _) = try await URLSession.shared.data(
                from: try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/x/status")))
            return (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        }

        func rerank(_ id: String) async throws -> Int {
            try await post("/v1/rerank", [
                "model": id,
                "query": "How many people live in Berlin?",
                "documents": ["Berlin has about 3.5 million inhabitants.", "The museum opens at nine."],
            ]).0
        }

        func embed(_ id: String) async throws -> Int {
            try await post("/v1/embeddings", ["model": id, "input": "Berlin has about 3.5 million inhabitants."]).0
        }
    }

    private func residencyServer(_ table: [String: LocalModel], port: Int) async throws -> ResidencyServer {
        let residency = ReleasingResidency()
        let stub = StubInferenceEngine(engineID: .mlxSwift)
        let models = OSAllocatedUnfairLock(initialState: table)
        let server = HummingbirdServer(
            engineProvider: { stub }, modelResolver: { id in models.withLock { $0[id] } },
            auxiliaryResidency: residency)
        let bound = try await server.start(preferredPort: port)
        return ResidencyServer(server: server, residency: residency, port: bound, models: models)
    }

    /// The real server with a pool stand-in: a cold request admits the
    /// reranker before loading it, a repeat marks it in use in one step, a hit
    /// the pool refuses (it let the model go) loads afresh, the use is ended
    /// after the score, the pool's release empties the slot (and the
    /// drain that follows leaves MLX's cache near empty) so the next request
    /// admits again, a swap forgets the old id and admits the new one with the
    /// old weights out of the cache, a release for the model already swapped
    /// out is ignored, a swap into a model that fails to load still forgets
    /// the old one and leaves the slot empty, and the unload door drops a
    /// reranker by id, tells the pool, and drains.
    func testServerReportsRerankerResidencyToThePool() async throws {
        try requireGate()
        let encoderDir = try resolve(Self.encoder)
        let qwenDir = try resolve(Self.qwen3)
        let msA = localModel(id: "ms-a", directory: encoderDir)
        let qwen = localModel(id: "qwen", directory: qwenDir)
        let broken = localModel(
            id: "broken", directory: URL(fileURLWithPath: "/nonexistent/broken-reranker", isDirectory: true))
        let table = [msA.id: msA, qwen.id: qwen, broken.id: broken]
        let cacheBound = 64 << 20
        try requireWeights(in: encoderDir, toOutweigh: cacheBound, "ms-marco")
        let s = try await residencyServer(table, port: 19_781)
        let residency = s.residency
        do {
            Memory.clearCache()
            let cold = try await s.rerank(msA.id)
            XCTAssertEqual(cold, 200)
            var events = await residency.events
            XCTAssertEqual(
                events, ["admit reranker:ms-a", "end reranker:ms-a"],
                "a cold request admits before it loads, in use from the admission until it has scored")
            let hit = try await s.rerank(msA.id)
            XCTAssertEqual(hit, 200)
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(2)),
                ["use reranker:ms-a", "end reranker:ms-a"],
                "a repeat marks the resident model in use, in one step")
            var resident = try await s.status()["reranker"] as? String
            XCTAssertEqual(resident, "ms-a")

            await residency.reclaim(.reranker, msA.id)
            resident = try await s.status()["reranker"] as? String
            XCTAssertNil(resident, "the pool's release empties the slot")
            let cachedAfterRelease = Memory.cacheMemory
            XCTAssertLessThan(
                cachedAfterRelease, cacheBound,
                "the release drops the engine and drains: \(cachedAfterRelease) B left in MLX's cache")
            let again = try await s.rerank(msA.id)
            XCTAssertEqual(again, 200)
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(2)),
                ["admit reranker:ms-a", "end reranker:ms-a"],
                "the next request admits again")

            // A hit after the pool let the model go (its release still on its
            // way): the request loads afresh instead of computing on a model
            // the pool no longer counts, and that release then does nothing.
            let releaseOfTheLetGoHandle = await residency.release(for: .reranker, msA.id)
            let releaseOfTheLetGo = try XCTUnwrap(releaseOfTheLetGoHandle, "the pool still holds the release for ms-a")
            await residency.letGoOfTheNextUse()
            let readmitted = try await s.rerank(msA.id)
            XCTAssertEqual(readmitted, 200)
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(4)),
                ["use refused reranker:ms-a", "forget reranker:ms-a", "admit reranker:ms-a", "end reranker:ms-a"],
                "a refused use swaps: the slot is cleared, the pool told, the model admitted afresh")
            await releaseOfTheLetGo()
            resident = try await s.status()["reranker"] as? String
            XCTAssertEqual(resident, "ms-a", "the release of the admission the pool let go finds the token moved on")
            let cachedAfterRefusal = Memory.cacheMemory
            XCTAssertLessThan(cachedAfterRefusal, cacheBound, "MLX's cache is near empty after the readmission: \(cachedAfterRefusal) B left")

            // The other ordering: the release reached the server while the
            // request was suspended, so the slot is already empty when the
            // hit is refused. The engine the request held is drained before
            // it admits afresh; here that readmission fails (the id now
            // resolves to a directory that does not exist), so nothing
            // recycles the old buffers and the bound shows the drain.
            let brokenMsA = localModel(
                id: msA.id, directory: URL(fileURLWithPath: "/nonexistent/ms-a-gone", isDirectory: true))
            s.models.withLock { $0[msA.id] = brokenMsA }
            await residency.letGoOfTheNextUseReleasingFirst()
            let refusedAfterRelease = try await s.rerank(msA.id)
            XCTAssertEqual(refusedAfterRelease, 500, "the readmission fails on the missing directory")
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(3)),
                ["release reranker:ms-a", "use refused reranker:ms-a", "admit reranker:ms-a"],
                "the release ran first, the use was refused, the readmission attempted (and failed)")
            resident = try await s.status()["reranker"] as? String
            XCTAssertNil(resident, "nothing is resident after the failed readmission")
            let cachedAfterReleaseFirst = Memory.cacheMemory
            XCTAssertLessThan(
                cachedAfterReleaseFirst, cacheBound,
                "the engine the refused request held was drained before the readmission: \(cachedAfterReleaseFirst) B left")
            s.models.withLock { $0[msA.id] = msA }
            let recovered = try await s.rerank(msA.id)
            XCTAssertEqual(recovered, 200, "the next request admits the real checkpoint again")

            let staleReleaseHandle = await residency.release(for: .reranker, msA.id)
            let staleRelease = try XCTUnwrap(staleReleaseHandle, "the pool still holds the release for ms-a")
            let swapped = try await s.rerank(qwen.id)
            XCTAssertEqual(swapped, 200)
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(3)),
                ["forget reranker:ms-a", "admit reranker:qwen", "end reranker:qwen"],
                "a swap forgets the old id first")
            await staleRelease()
            resident = try await s.status()["reranker"] as? String
            XCTAssertEqual(resident, "qwen", "a release for the model already swapped out leaves the new one alone")

            // Back to the small model: the dropped Qwen3 reranker's buffers must
            // have left MLX's cache with the swap (the #133 measurement: around
            // 300 MB stay when the pool's closure keeps the engine alive past
            // the drain, a few MB when the slot held the last reference).
            let back = try await s.rerank(msA.id)
            XCTAssertEqual(back, 200)
            events = await residency.events
            XCTAssertEqual(Array(events.suffix(3)).prefix(2), ["forget reranker:qwen", "admit reranker:ms-a"])
            let cachedAfterSwapBack = Memory.cacheMemory
            XCTAssertLessThan(
                cachedAfterSwapBack, cacheBound,
                "the swap's drain must free the dropped model's buffers: \(cachedAfterSwapBack) B left in MLX's cache")

            let failed = try await s.rerank(broken.id)
            XCTAssertEqual(failed, 500)
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(2)), ["forget reranker:ms-a", "admit reranker:broken"],
                "a swap into a model that fails still forgets the old one")
            resident = try await s.status()["reranker"] as? String
            XCTAssertNil(resident, "and leaves the slot empty")
            let reloaded = try await s.rerank(qwen.id)
            XCTAssertEqual(reloaded, 200)

            let (unloadStatus, unloadBody) = try await s.post("/x/models/unload", ["model": "qwen"])
            XCTAssertEqual(unloadStatus, 200)
            XCTAssertEqual(unloadBody["model"] as? String, "qwen")
            events = await residency.events
            XCTAssertEqual(events.last, "forget reranker:qwen", "the unload door tells the pool")
            resident = try await s.status()["reranker"] as? String
            XCTAssertNil(resident)
            let cachedAfterDoor = Memory.cacheMemory
            XCTAssertLessThan(
                cachedAfterDoor, cacheBound,
                "the door drops the engine and drains: \(cachedAfterDoor) B left in MLX's cache")
            let (goneStatus, _) = try await s.post("/x/models/unload", ["model": "qwen"])
            XCTAssertEqual(goneStatus, 404, "unloading it twice finds nothing")
        } catch {
            await s.server.stop()
            throw error
        }
        await s.server.stop()
        let afterStop = await residency.events
        XCTAssertEqual(afterStop.last, "forget reranker:qwen", "stopping a server that holds nothing forgets nothing more")
    }

    /// A release that beats the install: the pool lets the admission go
    /// before the server has put the engine in its slot (an `unload` of the
    /// key, since an admission in use is not evicted). The request is served on
    /// the engine it loaded, nothing is installed (the pool holds no entry for
    /// it), and the next request admits afresh.
    func testAReleaseThatBeatsTheInstallIsHonoured() async throws {
        try requireGate()
        let encoderDir = try resolve(Self.encoder)
        let msA = localModel(id: "ms-a", directory: encoderDir)
        try requireWeights(in: encoderDir, toOutweigh: 64 << 20, "ms-marco")
        let s = try await residencyServer([msA.id: msA], port: 19_783)
        let residency = s.residency
        do {
            Memory.clearCache()
            await residency.releaseTheNextAdmissionAtOnce()
            let served = try await s.rerank(msA.id)
            XCTAssertEqual(served, 200, "the request runs on the engine it loaded")
            var events = await residency.events
            XCTAssertEqual(
                events,
                ["admit reranker:ms-a", "release reranker:ms-a before install", "forget reranker:ms-a", "end reranker:ms-a"],
                "the slot moved on, so the admission is forgotten, and the request ends the use the admission began")
            let resident = try await s.status()["reranker"] as? String
            XCTAssertNil(resident, "an admission the pool released is not installed")
            // The request was the last holder of that engine: it drained after
            // itself, so the weights it freed are not parked in MLX's cache.
            let cachedAfterOrphan = Memory.cacheMemory
            XCTAssertLessThan(
                cachedAfterOrphan, 64 << 20,
                "an orphaned engine's buffers must be drained by the request that held it: \(cachedAfterOrphan) B left")
            let next = try await s.rerank(msA.id)
            XCTAssertEqual(next, 200)
            events = await residency.events
            XCTAssertEqual(Array(events.suffix(2)).first, "admit reranker:ms-a", "the next request admits afresh")
        } catch {
            await s.server.stop()
            throw error
        }
        await s.server.stop()
    }

    /// The embedder half of the same contract, on `/v1/embeddings`: admit,
    /// the use begun and ended, a hit's use, the pool's release emptying the
    /// slot, and the server's stop forgetting what it still held.
    func testServerReportsEmbedderResidencyToThePool() async throws {
        try requireGate()
        let embedderDir = try resolve(Self.embedder)
        let embedderBDir = try resolve(Self.embedderB)
        let bge = LocalModel(
            id: "bge", displayName: "bge", directory: embedderDir, sizeBytes: 0,
            format: .embedder, quantization: nil, parameterCount: nil, architecture: nil)
        let bgeB = LocalModel(
            id: "bge-b", displayName: "bge-b", directory: embedderBDir, sizeBytes: 0,
            format: .embedder, quantization: nil, parameterCount: nil, architecture: nil)
        let cacheBound = 64 << 20
        try requireWeights(in: embedderDir, toOutweigh: cacheBound, "bge-small")
        let s = try await residencyServer([bge.id: bge, bgeB.id: bgeB], port: 19_782)
        let residency = s.residency
        do {
            Memory.clearCache()
            let cold = try await s.embed(bge.id)
            XCTAssertEqual(cold, 200)
            var events = await residency.events
            XCTAssertEqual(events, ["admit embedder:bge", "end embedder:bge"])
            let hit = try await s.embed(bge.id)
            XCTAssertEqual(hit, 200)
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(2)),
                ["use embedder:bge", "end embedder:bge"])
            var resident = try await s.status()["embedder"] as? String
            XCTAssertEqual(resident, "bge")

            await residency.reclaim(.embedder, bge.id)
            resident = try await s.status()["embedder"] as? String
            XCTAssertNil(resident, "the pool's release empties the embedder slot")
            let cached = Memory.cacheMemory
            XCTAssertLessThan(cached, cacheBound, "the release drops the engine and drains: \(cached) B left in MLX's cache")
            let again = try await s.embed(bge.id)
            XCTAssertEqual(again, 200)
            resident = try await s.status()["embedder"] as? String
            XCTAssertEqual(resident, "bge")

            // A swap forgets the old id first; a release for it afterwards is
            // ignored; the swap back finds the dropped model's buffers gone.
            let staleReleaseHandle = await residency.release(for: .embedder, bge.id)
            let staleRelease = try XCTUnwrap(staleReleaseHandle)
            let swapped = try await s.embed(bgeB.id)
            XCTAssertEqual(swapped, 200)
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(3)),
                ["forget embedder:bge", "admit embedder:bge-b", "end embedder:bge-b"],
                "a swap forgets the old id first")
            await staleRelease()
            resident = try await s.status()["embedder"] as? String
            XCTAssertEqual(resident, "bge-b", "a release for the model already swapped out leaves the new one alone")
            let back = try await s.embed(bge.id)
            XCTAssertEqual(back, 200)
            let cachedAfterSwapBack = Memory.cacheMemory
            XCTAssertLessThan(cachedAfterSwapBack, cacheBound, "the swap's drain must free the dropped model's buffers: \(cachedAfterSwapBack) B left")

            // The door by id tells the pool and drains.
            let (unloadStatus, unloadBody) = try await s.post("/x/models/unload", ["model": "bge"])
            XCTAssertEqual(unloadStatus, 200)
            XCTAssertEqual(unloadBody["model"] as? String, "bge")
            events = await residency.events
            XCTAssertEqual(events.last, "forget embedder:bge")
            resident = try await s.status()["embedder"] as? String
            XCTAssertNil(resident)
            let cachedAfterDoor = Memory.cacheMemory
            XCTAssertLessThan(cachedAfterDoor, cacheBound, "the door drops the engine and drains: \(cachedAfterDoor) B left")

            // A release that beats the install: served, not installed, drained.
            await residency.releaseTheNextAdmissionAtOnce()
            let orphaned = try await s.embed(bge.id)
            XCTAssertEqual(orphaned, 200)
            resident = try await s.status()["embedder"] as? String
            XCTAssertNil(resident, "an admission the pool released is not installed")
            let cachedAfterOrphan = Memory.cacheMemory
            XCTAssertLessThan(cachedAfterOrphan, cacheBound, "the orphaned engine is drained by the request that held it: \(cachedAfterOrphan) B left")

            // Resident again (the orphan left the slot empty), for the two
            // refusals below.
            let reloaded = try await s.embed(bge.id)
            XCTAssertEqual(reloaded, 200)
            // A hit the pool refuses (it let the model go) admits afresh, and
            // the release of the admission it let go then does nothing.
            let bgeReleaseHandle = await residency.release(for: .embedder, bge.id)
            let bgeRelease = try XCTUnwrap(bgeReleaseHandle, "the pool still holds the release for bge")
            await residency.letGoOfTheNextUse()
            let readmitted = try await s.embed(bge.id)
            XCTAssertEqual(readmitted, 200)
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(4)),
                ["use refused embedder:bge", "forget embedder:bge", "admit embedder:bge", "end embedder:bge"],
                "a refused use swaps: the slot is cleared, the pool told, the model admitted afresh")
            await bgeRelease()
            resident = try await s.status()["embedder"] as? String
            XCTAssertEqual(resident, "bge", "the release of the admission the pool let go finds the token moved on")

            // The other ordering, as for the reranker: the release reached the
            // server first, so the slot is already empty when the hit is
            // refused; the engine the request held is drained before the
            // readmission, which fails here (the id now resolves to a
            // directory that does not exist), so nothing recycles the old
            // buffers and the bound shows the drain.
            let brokenBge = LocalModel(
                id: bge.id, displayName: bge.id,
                directory: URL(fileURLWithPath: "/nonexistent/bge-gone", isDirectory: true), sizeBytes: 0,
                format: .embedder, quantization: nil, parameterCount: nil, architecture: nil)
            s.models.withLock { $0[bge.id] = brokenBge }
            await residency.letGoOfTheNextUseReleasingFirst()
            let refusedAfterRelease = try await s.embed(bge.id)
            XCTAssertEqual(refusedAfterRelease, 500, "the readmission fails on the missing directory")
            events = await residency.events
            XCTAssertEqual(
                Array(events.suffix(3)),
                ["release embedder:bge", "use refused embedder:bge", "admit embedder:bge"],
                "the release ran first, the use was refused, the readmission attempted (and failed)")
            resident = try await s.status()["embedder"] as? String
            XCTAssertNil(resident, "nothing is resident after the failed readmission")
            let cachedAfterReleaseFirst = Memory.cacheMemory
            XCTAssertLessThan(
                cachedAfterReleaseFirst, cacheBound,
                "the engine the refused request held was drained before the readmission: \(cachedAfterReleaseFirst) B left")
            s.models.withLock { $0[bge.id] = bge }
            let recovered = try await s.embed(bge.id)
            XCTAssertEqual(recovered, 200, "the next request admits the real checkpoint again")

            // A hit on the resident embedder: its use begins in one step and
            // ends after the work, here an input the model truncates to its
            // 512 positions.
            let overlong = String(repeating: "柏林有三百五十万居民。", count: 4_000)
            let (overlongStatus, _) = try await s.post("/v1/embeddings", ["model": bge.id, "input": overlong])
            XCTAssertEqual(overlongStatus, 200, "the model truncates the input")
            events = await residency.events
            XCTAssertEqual(Array(events.suffix(2)), ["use embedder:bge", "end embedder:bge"], "the use begun on the hit is ended")
        } catch {
            await s.server.stop()
            throw error
        }
        await s.server.stop()
        let afterStop = await residency.events
        XCTAssertEqual(afterStop.last, "forget embedder:bge", "stopping the server forgets the embedder it still held")
        let cachedAfterStop = Memory.cacheMemory
        XCTAssertLessThan(cachedAfterStop, cacheBound, "stopping drains what it let go of: \(cachedAfterStop) B left")
    }

    /// The server lets its auxiliary engines go only once it has stopped
    /// serving: a request parked on the generation lock when `stop()` is
    /// called still runs, admits and installs, and the stop then forgets what
    /// that request installed, rather than the other way round.
    func testStopLetsGoOfWhatARequestInFlightInstalled() async throws {
        try requireGate()
        let encoderDir = try resolve(Self.encoder)
        let msA = localModel(id: "ms-a", directory: encoderDir)
        let s = try await residencyServer([msA.id: msA], port: 19_784)
        let residency = s.residency
        try await s.server.acquireGenerationLock()
        let parked = Task { try await s.rerank(msA.id) }
        // The request must be inside the server, parked on the lock, before
        // the stop is triggered: a stop that lands first closes the listener
        // and the request never connects. Polled, since it must happen.
        let deadline = ContinuousClock.now + .seconds(60)
        while await s.server.generationLockWaiterCount == 0, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        let waiters = await s.server.generationLockWaiterCount
        XCTAssertEqual(waiters, 1, "the request is parked on the lock before the stop")
        let stopping = Task { await s.server.stop() }
        // The stop must have begun (it stops listening before it waits for
        // the parked request) before the lock is released, or the request
        // would install first whatever the stop's order. Polled, since it
        // must happen.
        let stopDeadline = ContinuousClock.now + .seconds(60)
        while await s.server.isRunning, ContinuousClock.now < stopDeadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        let running = await s.server.isRunning
        XCTAssertFalse(running, "the stop has begun")
        // A fixed wait, followed only by "has not happened yet" assertions:
        // the request is parked on the lock, so nothing can have been admitted.
        try await Task.sleep(nanoseconds: 300_000_000)
        let beforeRelease = await residency.events
        XCTAssertEqual(beforeRelease, [], "nothing is admitted while the lock is held")
        await s.server.releaseGenerationLock()
        let status = try await parked.value
        await stopping.value
        XCTAssertEqual(status, 200, "the parked request is served")
        let events = await residency.events
        XCTAssertEqual(events.first, "admit reranker:ms-a")
        XCTAssertEqual(events.last, "forget reranker:ms-a", "the stop forgets what the request installed, after it: \(events)")
        let stillHeld = await residency.release(for: .reranker, msA.id)
        XCTAssertNil(stillHeld, "the pool holds no release for an engine the server let go")
    }

    /// The cache bound the residency smokes assert after a release or a drain
    /// (64 MB) can show a missing drain only when the checkpoint whose
    /// buffers would be left behind outweighs it: fp32 ms-marco is 91 MB and
    /// bge-small 96 MB, and a half-precision copy of either would pass,
    /// vacuously, every bound measured after it is dropped.
    private func requireWeights(in directory: URL, toOutweigh bound: Int, _ what: String) throws {
        let weights = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .reduce(Int64(0)) { total, file in
                let size = (try? file.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                return total + Int64(size)
            }
        XCTAssertGreaterThan(
            weights, Int64(bound),
            "\(what) weighs \(weights) B against the \(bound) B cache bound, which could then not show a missing drain")
    }

    /// The bytes under `directory`, every file, following the HF cache's
    /// symlinks to the blobs: enough for a budget that fits exactly one of two
    /// models (the library itself sums only the top-level weights).
    private func directorySize(_ directory: URL) -> Int64 {
        guard let entries = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil) else {
            return 0
        }
        var total: Int64 = 0
        for case let url as URL in entries {
            let resolved = url.resolvingSymlinksInPath()
            if let size = (try? resolved.resourceValues(forKeys: [.fileSizeKey]))?.fileSize {
                total += Int64(size)
            }
        }
        return total
    }

    /// The server over a real pool whose budget holds one of the two models,
    /// through the residency the app hands in. Admitting the reranker evicts
    /// the embedder: the pool runs the embedder's release, which re-enters
    /// the server while the rerank handler holds the generation lock and
    /// awaits the admission. The release must therefore never take the lock;
    /// this is the test that would not return if it did, hence the bound.
    func testAdmittingOneSlotEvictsTheOtherThroughARealPool() async throws {
        try requireGate()
        let encoderDir = try resolve(Self.encoder)
        let embedderDir = try resolve(Self.embedder)
        let msA = LocalModel(
            id: "ms-a", displayName: "ms-a", directory: encoderDir, sizeBytes: directorySize(encoderDir),
            format: .reranker, quantization: nil, parameterCount: nil, architecture: nil)
        let bge = LocalModel(
            id: "bge", displayName: "bge", directory: embedderDir, sizeBytes: directorySize(embedderDir),
            format: .embedder, quantization: nil, parameterCount: nil, architecture: nil)
        XCTAssertGreaterThan(msA.sizeBytes, 0)
        XCTAssertGreaterThan(bge.sizeBytes, 0)
        try requireWeights(in: embedderDir, toOutweigh: 64 << 20, "bge-small")
        let pool = ModelPool(maxBytes: max(msA.sizeBytes, bge.sizeBytes) + 1) { _ in
            StubInferenceEngine(engineID: .mlxSwift)
        }
        let parametersDir = FileManager.default.temporaryDirectory
            .appending(path: "aux-pool-smoke-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: parametersDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: parametersDir) }
        let residency = PoolAuxiliaryResidency(
            pool: pool, parameters: ModelParametersStore(directory: parametersDir))
        let table = [msA.id: msA, bge.id: bge]
        let stub = StubInferenceEngine(engineID: .mlxSwift)
        let server = HummingbirdServer(
            engineProvider: { stub }, modelResolver: { table[$0] }, auxiliaryResidency: residency)
        let port = try await server.start(preferredPort: 19_785)
        // For the request helpers only; this server reports to the pool.
        let s = ResidencyServer(
            server: server, residency: ReleasingResidency(), port: port,
            models: OSAllocatedUnfairLock(initialState: table))
        do {
            Memory.clearCache()
            let embedded = try await s.embed(bge.id)
            XCTAssertEqual(embedded, 200)
            var resident = await pool.residentModelIDs()
            XCTAssertEqual(resident, ["embedder:bge"])
            // Neither child throws: a request that fails or times out (the
            // client gives up after 60 s) comes back as -2, the clock as -1,
            // so a stuck handler is recorded rather than thrown into the
            // catch below, whose `stop()` would wait for it forever.
            let scored: Int = await withTaskGroup(of: Int.self) { group in
                group.addTask { (try? await s.rerank(msA.id)) ?? -2 }
                group.addTask {
                    try? await Task.sleep(for: .seconds(120))
                    return -1
                }
                let first = await group.next() ?? -1
                group.cancelAll()
                return first
            }
            XCTAssertEqual(
                scored, 200,
                "the rerank is served; -2 or -1 means it never completed, which is what a release waiting for the lock looks like")
            // A handler that never completed (-2 or -1) would also hang
            // `stop()` (the graceful shutdown waits for it): record the
            // failure and leave the server behind. An ordinary error status
            // goes on to the stop.
            guard scored >= 0 else { return }
            resident = await pool.residentModelIDs()
            XCTAssertEqual(resident, ["reranker:ms-a"], "the embedder made room for the reranker")
            let status = try await s.status()
            XCTAssertNil(status["embedder"] as? String, "the pool's release emptied the embedder slot")
            XCTAssertEqual(status["reranker"] as? String, "ms-a")
            let cached = Memory.cacheMemory
            XCTAssertLessThan(
                cached, 64 << 20,
                "MLX's cache is near empty after the swap (the reranker's load may have reused the embedder's buffers; the embedder smoke checks the drain itself): \(cached) B")
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()
        let afterStop = await pool.residentModelIDs()
        XCTAssertEqual(afterStop, [], "the stop forgot the reranker")
    }
}
