// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// End-to-end proof that a ``BatchKVCache`` cohort of RAGGED (different-length)
/// prompts decodes correctly through a REAL dense model — the A2b payoff that
/// A2a's equal-length path could not reach.
///
/// The seam into ``BatchDecodeRunner`` is intentionally NOT wired here (that is
/// A2c; see ``BatchKVCache``'s "Deferred to a follow-up wave" note). This test
/// instead drives a hand-written batched
/// forward loop over caches built by ``BatchCacheConverter`` — the "verify the
/// cache directly" path — so the cache is proven end-to-end independent of the
/// scheduler.
///
/// GATED — never runs in CI. It self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_BATCH_SPIKE=1`,
///   3. a DENSE model is on disk (env `MACMLX_BATCH_RAGGED_MODEL` names a dir
///      under `~/.mac-mlx/models`, else the built-in candidates are tried), and
///   4. that model's caches are all plain `KVCacheSimple` — the converter
///      refuses sliding-window models (Gemma etc.), which need the deferred
///      `BatchRotatingKVCache`, and the test skips them.
///
/// ## What it asserts (RoPE-version-robust)
///  - **Runs + full length:** the ragged batched forward produces a complete,
///    finite trajectory per row (a broken left-pad mask would NaN or truncate).
///  - **Per-row isolation:** row 0's trajectory is unchanged when its batchmate
///    is swapped — the left-padding mask must keep each row's attention confined
///    to its own real tokens. This is the strongest cache-correctness signal
///    that does NOT depend on the mlx-swift RoPE version.
///
/// ## What it reports (RoPE-version-dependent, informational)
/// Each row's batched trajectory vs its STOCK B=1 greedy decode. A divergence on
/// the left-PADDED row exercises per-row NEGATIVE-offset RoPE during the batched
/// prefill; on mlx-swift's vendored mlx-core 0.31.1 this is the not-yet-landed
/// batched-RoPE fix (ml-explore/mlx-swift#441), NOT a ``BatchKVCache`` defect —
/// the cache's array math is proven RoPE-free by `BatchKVCacheParityTests`. It is
/// reported, not asserted, so this gated probe surfaces the upstream gap without
/// red-flagging a correct cache.
final class BatchKVCacheModelTests: XCTestCase {

    private enum ModelTestError: Error { case emptyPrompt }

    private struct RaggedResult: Sendable {
        var trajectories: [[Int]]
        var isolationRow0: [Int]
        var stockRefs: [[Int]]
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
            "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
            "Qwen3.5-35B-A3B-4bit",
        ])
        for name in candidates {
            let dir = root.appending(path: name, directoryHint: .isDirectory)
            if FileManager.default.fileExists(atPath: dir.path) { return dir }
        }
        return nil
    }

    func testRaggedCohortDecodesAndIsolatesPerRow() async throws {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_BATCH_SPIKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_BATCH_SPIKE=1 to run the A2b ragged real-model test")
        }
        guard let modelDir = denseModelDirectory() else {
            throw XCTSkip("No dense model found (set MACMLX_BATCH_RAGGED_MODEL to a dir name)")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelDir, using: HuggingFaceTokenizerLoader())

        let decodeCount = 24

        let result: RaggedResult? = try await container.perform { context in
            let model = context.model
            let tokenizer = context.tokenizer

            // Ragged prompts of deliberately different lengths.
            let promptShort = tokenizer.encode(text: "The capital of France is", addSpecialTokens: true)
            let promptLong = tokenizer.encode(
                text: "Once upon a time, in a distant kingdom by the sea, there lived a",
                addSpecialTokens: true)
            let promptOther = tokenizer.encode(
                text: "In the beginning the compiler emitted a", addSpecialTokens: true)
            guard !promptShort.isEmpty, !promptLong.isEmpty, !promptOther.isEmpty else {
                throw ModelTestError.emptyPrompt
            }

            /// Ragged batched greedy decode via a hand-written forward loop over
            /// `BatchKVCache`s. Returns per-row trajectories, or `nil` if this
            /// model's caches are not all dense (converter refused).
            func raggedDecode(_ prompts: [[Int]]) -> [[Int]]? {
                let padToken = prompts.first?.first ?? 0  // any valid, in-vocab id (masked out)
                let (padded, leftPadding) = BatchPrefillAssembly.leftPad(
                    prompts: prompts, padToken: padToken)
                guard
                    let caches = BatchCacheConverter.makeBatchCaches(
                        from: model.newCache(parameters: nil), leftPadding: leftPadding)
                else { return nil }

                let batch = prompts.count
                let lmax = padded.first?.count ?? 0
                let inputs = MLXArray(padded.flatMap { $0.map { Int32($0) } }, [batch, lmax])

                func lastArgmaxPerRow(_ logits: MLXArray) -> [Int] {
                    let seqLen = logits.dim(1)
                    let last = logits[0..., (seqLen - 1)..., 0...].reshaped(batch, -1)  // [B, vocab]
                    let tokens = argMax(last, axis: -1)
                    tokens.eval()
                    return tokens.asArray(Int.self)
                }

                var next = lastArgmaxPerRow(model(inputs, cache: caches))
                var trajectories = next.map { [$0] }
                for _ in 1..<decodeCount {
                    let step = MLXArray(next.map { Int32($0) }, [batch, 1])
                    next = lastArgmaxPerRow(model(step, cache: caches))
                    for row in 0..<batch { trajectories[row].append(next[row]) }
                }
                return trajectories
            }

            /// Stock (scalar-offset) B=1 greedy reference — the production path.
            func stockGreedy(_ prompt: [Int]) -> [Int] {
                let cache = model.newCache(parameters: nil)
                func lastArgmax(_ logits: MLXArray) -> Int {
                    let seqLen = logits.dim(1)
                    return logits[0..., (seqLen - 1)..., 0...].argMax(axis: -1).asArray(Int.self)[0]
                }
                var next = lastArgmax(model(MLXArray(prompt.map { Int32($0) }, [1, prompt.count]), cache: cache))
                var out = [next]
                for _ in 1..<decodeCount {
                    next = lastArgmax(model(MLXArray([Int32(next)], [1, 1]), cache: cache))
                    out.append(next)
                }
                return out
            }

            let (_, leftPadding) = BatchPrefillAssembly.leftPad(
                prompts: [promptShort, promptLong], padToken: 0)
            guard let trajectories = raggedDecode([promptShort, promptLong]) else {
                return nil  // non-dense model → caller skips
            }
            // Isolation: swap the batchmate; row 0 (promptShort) must be identical.
            guard let isolation = raggedDecode([promptShort, promptOther]) else { return nil }

            return RaggedResult(
                trajectories: trajectories,
                isolationRow0: isolation[0],
                stockRefs: [stockGreedy(promptShort), stockGreedy(promptLong)],
                leftPadding: leftPadding)
        }

        guard let result else {
            throw XCTSkip("Model caches are not all dense — needs BatchRotatingKVCache (deferred)")
        }

        // Sanity: the cohort is actually ragged (at least one row left-padded).
        XCTAssertTrue(
            result.leftPadding.contains { $0 > 0 }, "prompts must differ in length (ragged)")

        // Full-length trajectories per row.
        for (row, trajectory) in result.trajectories.enumerated() {
            XCTAssertEqual(
                trajectory.count, decodeCount, "row \(row) must decode a full-length trajectory")
        }

        // Per-row isolation (RoPE-version-robust cache-correctness gate).
        XCTAssertEqual(
            result.isolationRow0, result.trajectories[0],
            "row 0's trajectory must be independent of its batchmate — the left-padding mask "
                + "must confine each row to its own tokens")

        // Informational: batched-vs-stock-B1 divergence per row (RoPE-version).
        for (row, trajectory) in result.trajectories.enumerated() {
            let divergence = zip(trajectory, result.stockRefs[row])
                .enumerated().first { $0.element.0 != $0.element.1 }?.offset
            print(
                "BATCH_A2B row=\(row) leftPad=\(result.leftPadding[row]) "
                    + "batched-vs-stockB1 firstDivergence="
                    + "\(divergence.map(String.init) ?? "none (identical)") "
                    + "(a divergence on a left-padded row = mlx-swift#441 batched-prefill RoPE, "
                    + "not a BatchKVCache defect)")
        }
    }
}
