// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// End-to-end proof that the A2c continuous-batching numerical core decodes a
/// RAGGED (different-length) cohort correctly through a REAL dense model — the
/// A2c payoff over A2b.
///
/// A2b's `BatchKVCacheModelTests` batched the PREFILL (multi-row, multi-token),
/// which hits the not-yet-landed batched-prefill RoPE fix (mlx-swift#441), so it
/// could only *report* the per-row divergence, not assert parity. A2c prefills
/// each row B=1 (the proven scalar-offset path) and merges it into the cohort, so
/// only the L=1 DECODE is batched — the A1-proven correct kernel. That lets this
/// test ASSERT the strongest signal: each row's ragged batched trajectory is
/// TOKEN-FOR-TOKEN identical to its standalone B=1 greedy decode.
///
/// It drives ``ModelBatchInferenceCore`` directly (admit → decode), the same
/// "verify the seam by hand" style as `BatchKVCacheModelTests` — the scheduler
/// actor's admission/eviction/streaming logic is covered MLX-free in
/// `BatchSchedulerLogicTests`; this isolates the MLX numerics.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_BATCH_SPIKE=1`,
///   3. a DENSE, allowlisted model is on disk (env `MACMLX_BATCH_RAGGED_MODEL`
///      names a dir under `~/.mac-mlx/models`, else the built-in candidates —
///      e.g. `Qwen3-4B-4bit` — are tried), and
///   4. that model passes the coverage gate (dense caches + verified `ropeOffset`
///      architecture); an uncovered model makes `admit` throw and the test skips.
final class BatchSchedulerModelTests: XCTestCase {

    private enum ModelTestError: Error { case emptyPrompt }

    /// Only `Sendable` `[[Int]]` rides back out of `container.perform`; every
    /// `MLXArray` / cache stays inside the isolation domain.
    private struct RaggedRun: Sendable {
        var trajectories: [[Int]]
        var stockRefs: [[Int]]
        var isolationRow0: [Int]
        var leftPadding: [Int]
    }

    private func denseModelDirectory() -> URL? {
        let root = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models", directoryHint: .isDirectory)
        var candidates: [String] = []
        if let override = ProcessInfo.processInfo.environment["MACMLX_BATCH_RAGGED_MODEL"] {
            candidates.append(override)
        }
        candidates.append(contentsOf: [
            "Qwen3-4B-4bit",
            "Qwen3-4B-8bit",
            "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
            "Qwen3.5-35B-A3B-4bit",
        ])
        for name in candidates {
            let dir = root.appending(path: name, directoryHint: .isDirectory)
            if FileManager.default.fileExists(atPath: dir.path) { return dir }
        }
        return nil
    }

    func testRaggedCohortEachRowMatchesStandaloneB1Greedy() async throws {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_BATCH_SPIKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_BATCH_SPIKE=1 to run the A2c ragged real-model test")
        }
        guard let modelDir = denseModelDirectory() else {
            throw XCTSkip("No dense model found (set MACMLX_BATCH_RAGGED_MODEL to a dir name)")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelDir, using: HuggingFaceTokenizerLoader())

        let steps = 24

        let run: RaggedRun? = try await container.perform { context in
            let model = context.model
            let tokenizer = context.tokenizer

            let promptShort = tokenizer.encode(
                text: "The capital of France is", addSpecialTokens: true)
            let promptLong = tokenizer.encode(
                text: "Once upon a time, in a distant kingdom by the sea, there lived a",
                addSpecialTokens: true)
            let promptOther = tokenizer.encode(
                text: "In the beginning the compiler emitted a", addSpecialTokens: true)
            guard !promptShort.isEmpty, !promptLong.isEmpty, !promptOther.isEmpty else {
                throw ModelTestError.emptyPrompt
            }

            /// Ragged greedy decode through the A2c core: B=1 prefill per row
            /// (inside `admit`), then batched L=1 lockstep. Returns per-row
            /// trajectories, or `nil` if the model fails the coverage gate.
            func raggedGreedy(_ prompts: [[Int]]) throws -> [[Int]]? {
                let core = ModelBatchInferenceCore(model: model)
                let configs = prompts.map {
                    BatchSlotConfig(
                        promptTokens: $0,
                        parameters: GenerateParameters(maxTokens: steps, temperature: 0))
                }
                let firstTokens: [Int]
                do {
                    firstTokens = try core.admit(configs)
                } catch BatchUnsupportedError.cacheNotBatchable {
                    return nil  // uncovered model → caller skips
                }
                var trajectories = firstTokens.map { [$0] }
                var feedback = firstTokens
                for _ in 1..<steps {
                    let next = try core.decode(feedback)
                    for row in prompts.indices { trajectories[row].append(next[row]) }
                    feedback = next
                }
                return trajectories
            }

            /// Stock scalar-offset B=1 greedy — the production single-stream path.
            func stockGreedy(_ prompt: [Int]) -> [Int] {
                let cache = model.newCache(parameters: nil)
                func lastArgmax(_ logits: MLXArray) -> Int {
                    let sequenceLength = logits.dim(1)
                    return logits[0..., (sequenceLength - 1)..., 0...]
                        .argMax(axis: -1).asArray(Int.self)[0]
                }
                var next = lastArgmax(
                    model(MLXArray(prompt.map { Int32($0) }, [1, prompt.count]), cache: cache))
                var out = [next]
                for _ in 1..<steps {
                    next = lastArgmax(model(MLXArray([Int32(next)], [1, 1]), cache: cache))
                    out.append(next)
                }
                return out
            }

            guard let trajectories = try raggedGreedy([promptShort, promptLong, promptOther]) else {
                return nil
            }
            // Isolation: swap row 1/2 for a different batchmate; row 0 must not move.
            guard let isolation = try raggedGreedy([promptShort, promptOther]) else { return nil }

            let (_, leftPadding) = BatchPrefillAssembly.leftPad(
                prompts: [promptShort, promptLong, promptOther], padToken: 0)
            return RaggedRun(
                trajectories: trajectories,
                stockRefs: [stockGreedy(promptShort), stockGreedy(promptLong), stockGreedy(promptOther)],
                isolationRow0: isolation[0],
                leftPadding: leftPadding)
        }

        guard let run else {
            throw XCTSkip("Model failed the batch coverage gate (non-dense or unlisted arch)")
        }

        // The cohort is genuinely ragged (at least one row left-padded).
        XCTAssertTrue(
            run.leftPadding.contains { $0 > 0 }, "prompts must differ in length (ragged)")

        // Full-length trajectory per row.
        for (row, trajectory) in run.trajectories.enumerated() {
            XCTAssertEqual(
                trajectory.count, steps, "row \(row) must decode a full-length trajectory")
        }

        // THE A2c PAYOFF: each row's ragged batched trajectory is token-for-token
        // identical to its standalone B=1 greedy decode. This is assertable
        // (unlike A2b) because A2c prefills B=1 and only batches the L=1 decode.
        for row in run.trajectories.indices {
            XCTAssertEqual(
                run.trajectories[row], run.stockRefs[row],
                "row \(row): ragged batched decode must match standalone B=1 greedy "
                    + "(B=1 prefill + batched L=1 decode is numerically exact)")
        }

        // Per-row isolation: row 0 is unchanged when its batchmates are swapped —
        // the left-padding mask confines each row to its own tokens.
        XCTAssertEqual(
            run.isolationRow0, run.trajectories[0],
            "row 0's trajectory must be independent of its batchmates")
    }

    /// Regression lock for the coverage-gate retry bypass (adversarial-review
    /// HIGH): a model that fails the coverage gate must be refused on EVERY
    /// admit, not only the first. A cached failed verdict must never let a
    /// later submit slip an uncovered model into a batched decode, where a
    /// scalar-offset architecture would silently return wrong tokens.
    func testUncoveredModelIsRefusedOnEveryAdmit() async throws {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_BATCH_SPIKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_BATCH_SPIKE=1 to run the A2c coverage-gate test")
        }
        guard let modelDir = uncoveredModelDirectory() else {
            throw XCTSkip(
                "No uncovered (hybrid) model found (set MACMLX_BATCH_UNCOVERED_MODEL)")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelDir, using: HuggingFaceTokenizerLoader())

        let refusals: [Bool] = try await container.perform { context in
            let core = ModelBatchInferenceCore(model: context.model)
            let prompt = context.tokenizer.encode(text: "Hello", addSpecialTokens: true)
            let config = BatchSlotConfig(
                promptTokens: prompt,
                parameters: GenerateParameters(maxTokens: 4, temperature: 0))
            var outcomes: [Bool] = []
            for _ in 0 ..< 2 {
                do {
                    _ = try core.admit([config])
                    outcomes.append(false)
                } catch is BatchUnsupportedError {
                    outcomes.append(true)
                }
            }
            return outcomes
        }

        XCTAssertEqual(
            refusals, [true, true],
            "an uncovered model must be refused on the FIRST and EVERY LATER admit — "
                + "a failed coverage verdict must re-throw, never silently admit on retry")
    }

    /// A model that FAILS the coverage gate on purpose: Qwen3.5 is hybrid
    /// (GatedDeltaNet linear layers → `MambaCache`), so it fails BOTH gates —
    /// non-dense caches AND `Qwen35Model` absent from the allowlist.
    private func uncoveredModelDirectory() -> URL? {
        let root = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models", directoryHint: .isDirectory)
        var candidates: [String] = []
        if let override = ProcessInfo.processInfo.environment["MACMLX_BATCH_UNCOVERED_MODEL"] {
            candidates.append(override)
        }
        candidates.append(contentsOf: [
            "Qwen3.5-4B-MLX-4bit",
            "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
            "Qwen3.5-35B-A3B-4bit",
        ])
        for name in candidates {
            let dir = root.appending(path: name, directoryHint: .isDirectory)
            if FileManager.default.fileExists(atPath: dir.path) { return dir }
        }
        return nil
    }
}
