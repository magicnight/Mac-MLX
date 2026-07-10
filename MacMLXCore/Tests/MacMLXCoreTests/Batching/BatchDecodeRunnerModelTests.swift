// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// End-to-end proof that the A2a ``BatchDecodeRunner`` decodes correctly on a
/// REAL model, exercising the full A2a path (batch-positioned cache → batched
/// forward → per-row greedy sampling → per-row stop → per-slot fan-out).
///
/// GATED — never runs in CI. It self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal backend, i.e. xcodebuild),
///   2. env `MACMLX_RUN_BATCH_SPIKE=1` is set, and
///   3. the gemma-4-e4b-it-8bit model dir exists on disk.
///
/// Run:
///   MACMLX_RUN_BATCH_SPIKE=1 TEST_RUNNER_MACMLX_RUN_BATCH_SPIKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/BatchDecodeRunnerModelTests
///
/// ## What it proves
///  - **Cross-row identity (the A1 gate, through A2a):** a B=4 cohort of the SAME
///    prompt yields four IDENTICAL greedy token trajectories. This re-proves the
///    ``BatchPositionedCacheWrapper`` RoPE fix survives A2a's full sampling + stop
///    + fan-out path (exact — same `.batch` kernel, identical rows).
///  - **Per-slot parity (through the same path):** a B=2 cohort of two DIFFERENT
///    equal-length prompts — each slot's trajectory equals that prompt's
///    STANDALONE B=1 decode through the same runner. `B == 1` is just a
///    single-row cohort, so this is an apples-to-apples same-kernel comparison.
///  - **Per-slot isolation (exact backstop):** row 0's trajectory is unchanged
///    when its batchmate is swapped — the batched forward computes each row
///    independently of the others.
///
/// A separate informational line compares against the STOCK scalar-path B=1
/// decode (the production single-stream path). A late divergence there is the
/// known, legal batch-size / kernel non-invariance (documented on
/// ``BatchPositionedCacheWrapper``), so it is reported, not asserted.
final class BatchDecodeRunnerModelTests: XCTestCase {

    private enum ModelTestError: Error { case emptyPrompt }

    /// All the trajectories collected inside `container.perform` (only `Sendable`
    /// `[[Int]]` rides back out; every `MLXArray` / `KVCache` stays in the actor).
    private struct Trajectories: Sendable {
        var b4Identical: [[Int]]
        var b4SamePathRef: [Int]
        var b2Different: [[Int]]
        var refA: [Int]
        var refB: [Int]
        var isolationRow0: [Int]
        var stockRefA: [Int]
    }

    func testBatchDecodeCoreRealModelParity() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_BATCH_SPIKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_BATCH_SPIKE=1 to run the A2a real-model parity test")
        }

        let modelDir = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/gemma-4-e4b-it-8bit", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw XCTSkip("Parity model dir not found: \(modelDir.path)")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelDir, using: HuggingFaceTokenizerLoader())

        let decodeCount = 24  // fixed-length trajectories (no EOS early-stop)

        let trajectories: Trajectories = try await container.perform { context in
            let model = context.model
            let tokenizer = context.tokenizer
            // Empty EOS set ⇒ rows never early-stop; every row decodes exactly
            // `decodeCount` tokens, so trajectories are directly comparable.
            let eos: Set<Int> = []

            /// Decode a cohort through the A2a runner and return per-slot tokens.
            func runCohort(_ promptRows: [[Int]]) throws -> [[Int]] {
                let cohort = promptRows.map {
                    BatchSlotConfig(
                        promptTokens: $0,
                        parameters: GenerateParameters(maxTokens: decodeCount, temperature: 0)
                    )
                }
                let (runner, _) = try BatchDecodeRunner.make(
                    model: model,
                    tokenizer: tokenizer,
                    eosTokenIds: eos,
                    cohort: cohort,
                    globalMaxTokens: decodeCount
                )
                try runner.run()
                return runner.slotTokens
            }

            /// Reference greedy decode via the STOCK (scalar-offset) B=1 path —
            /// the production single-stream path, NOT the batch-positioned one.
            func stockGreedy(_ promptTokens: [Int]) -> [Int] {
                let cache = model.newCache(parameters: nil)
                let promptArr = MLXArray(promptTokens, [1, promptTokens.count])
                func lastArgmax(_ logits: MLXArray) -> Int {
                    let seqLen = logits.dim(1)
                    return logits[0..., (seqLen - 1)..., 0...]
                        .argMax(axis: -1).asArray(Int.self)[0]
                }
                var logits = model(promptArr, cache: cache)
                var next = lastArgmax(logits)
                var out = [next]
                for _ in 1..<decodeCount {
                    logits = model(MLXArray([next], [1, 1]), cache: cache)
                    next = lastArgmax(logits)
                    out.append(next)
                }
                return out
            }

            // ---- Cohort A: B=4 identical prompt ----
            let identical = tokenizer.encode(
                text: "Planets in order: Mercury, Venus, Earth,", addSpecialTokens: true)
            guard !identical.isEmpty else { throw ModelTestError.emptyPrompt }
            let b4 = try runCohort(Array(repeating: identical, count: 4))
            let b4SamePathRef = try runCohort([identical])[0]

            // ---- Cohort B: B=2 different, equal-length prompts ----
            let tokensA = tokenizer.encode(
                text: "The capital of France is", addSpecialTokens: true)
            let tokensB = tokenizer.encode(
                text: "def add(a, b):\n    return", addSpecialTokens: true)
            let tokensC = tokenizer.encode(
                text: "Once upon a time there was", addSpecialTokens: true)
            let length = min(tokensA.count, min(tokensB.count, tokensC.count))
            guard length > 0 else { throw ModelTestError.emptyPrompt }
            let promptA = Array(tokensA.prefix(length))
            let promptB = Array(tokensB.prefix(length))
            let promptC = Array(tokensC.prefix(length))

            let b2 = try runCohort([promptA, promptB])
            let refA = try runCohort([promptA])[0]
            let refB = try runCohort([promptB])[0]
            let isolation = try runCohort([promptA, promptC])  // row 0 = A, batchmate swapped
            let stockRefA = stockGreedy(promptA)

            return Trajectories(
                b4Identical: b4,
                b4SamePathRef: b4SamePathRef,
                b2Different: b2,
                refA: refA,
                refB: refB,
                isolationRow0: isolation[0],
                stockRefA: stockRefA
            )
        }

        // ---- Cohort A assertions: cross-row identity (A1 re-proof through A2a) ----
        XCTAssertEqual(trajectories.b4Identical.count, 4)
        XCTAssertEqual(trajectories.b4Identical[0].count, decodeCount)
        for row in 1..<4 {
            XCTAssertEqual(
                trajectories.b4Identical[row], trajectories.b4Identical[0],
                "B=4 identical cohort: row \(row) must match row 0 (A1 row-identity through A2a)")
        }
        XCTAssertEqual(
            trajectories.b4Identical[0], trajectories.b4SamePathRef,
            "B=4 identical row must equal the standalone B=1 decode through the same path")

        // ---- Cohort B assertions: per-slot parity + isolation ----
        XCTAssertNotEqual(
            trajectories.refA, trajectories.refB,
            "the two prompts must actually decode to different token streams")
        XCTAssertEqual(
            trajectories.b2Different[0], trajectories.refA,
            "slot 0 must match its standalone B=1 greedy decode (same path)")
        XCTAssertEqual(
            trajectories.b2Different[1], trajectories.refB,
            "slot 1 must match its standalone B=1 greedy decode (same path)")
        XCTAssertEqual(
            trajectories.isolationRow0, trajectories.b2Different[0],
            "row 0's trajectory must be independent of its batchmate (per-slot isolation)")

        // ---- Informational: vs the STOCK scalar-path B=1 decode ----
        let stockDivergence = zip(trajectories.b2Different[0], trajectories.stockRefA)
            .enumerated().first { $0.element.0 != $0.element.1 }?.offset
        print(
            "BATCH_A2A slot0-vs-stockB1 firstDivergence="
                + "\(stockDivergence.map(String.init) ?? "none (identical)") "
                + "(late divergence = legal batch-size/kernel non-invariance)")
    }

    /// A mixed cohort — one greedy row, one temperature/top-p sampling row —
    /// forces `ModelBatchStepEvaluator`'s per-row processor/sampler LOOP for
    /// the WHOLE cohort (any row with `temperature != 0` disables the batched
    /// argmax fast path), unlike `testBatchDecodeCoreRealModelParity`'s
    /// greedy-only cohorts. Proves: (1) the greedy row's trajectory through
    /// that per-row loop still matches its standalone B=1 greedy decode
    /// exactly, and (2) the sampling row decodes a full, legal-length
    /// trajectory alongside it (values aren't asserted — sampling is
    /// non-deterministic without a fixed seed).
    func testMixedSamplingCohortGreedyRowMatchesStandaloneB1() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_BATCH_SPIKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_BATCH_SPIKE=1 to run the A2a real-model parity test")
        }

        let modelDir = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/gemma-4-e4b-it-8bit", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw XCTSkip("Parity model dir not found: \(modelDir.path)")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelDir, using: HuggingFaceTokenizerLoader())

        let decodeCount = 16

        let (mixedGreedyRow, standaloneGreedy, samplingRowCount): ([Int], [Int], Int) =
            try await container.perform { context in
                let model = context.model
                let tokenizer = context.tokenizer
                // Empty EOS set ⇒ rows never early-stop on EOS; the greedy row's
                // length is therefore directly comparable across cohorts.
                let eos: Set<Int> = []

                func runCohort(
                    _ rows: [(promptTokens: [Int], parameters: GenerateParameters)]
                ) throws -> [[Int]] {
                    let cohort = rows.map {
                        BatchSlotConfig(promptTokens: $0.promptTokens, parameters: $0.parameters)
                    }
                    let (runner, _) = try BatchDecodeRunner.make(
                        model: model,
                        tokenizer: tokenizer,
                        eosTokenIds: eos,
                        cohort: cohort,
                        globalMaxTokens: decodeCount
                    )
                    try runner.run()
                    return runner.slotTokens
                }

                let prompt = tokenizer.encode(
                    text: "Planets in order: Mercury, Venus, Earth,", addSpecialTokens: true)
                guard !prompt.isEmpty else { throw ModelTestError.emptyPrompt }

                let greedyParams = GenerateParameters(maxTokens: decodeCount, temperature: 0)
                let samplingParams = GenerateParameters(
                    maxTokens: decodeCount, temperature: 0.8, topP: 0.95)

                // Row 0 greedy, row 1 sampling — the mixed cohort that forces
                // the per-row loop for BOTH rows.
                let mixed = try runCohort([
                    (prompt, greedyParams), (prompt, samplingParams),
                ])

                // Reference: the same greedy row decoded ALONE (B=1), through
                // the same runner/path.
                let standalone = try runCohort([(prompt, greedyParams)])[0]

                return (mixed[0], standalone, mixed[1].count)
            }

        XCTAssertEqual(
            samplingRowCount, decodeCount,
            "sampling row must still produce a full, legal-length trajectory")
        XCTAssertEqual(
            mixedGreedyRow, standaloneGreedy,
            "the greedy row's trajectory (via the per-row loop) must be unaffected by a "
                + "sampling batchmate and must equal its standalone B=1 greedy decode")
    }

    /// Two rows share the SAME prompt but different `maxTokens` (4 vs 24), so
    /// row 0 finishes and freezes (MASK-not-shrink: fed a pad token) while row
    /// 1 keeps decoding live for the rest of the cohort. Proves a finished,
    /// padded batchmate never perturbs the still-running row's trajectory, and
    /// that the short row stops at exactly its cap with `finishReason == .length`.
    func testEarlyFinishBatchmateDoesNotPerturbStillRunningRow() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_BATCH_SPIKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_BATCH_SPIKE=1 to run the A2a real-model parity test")
        }

        let modelDir = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/gemma-4-e4b-it-8bit", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw XCTSkip("Parity model dir not found: \(modelDir.path)")
        }

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelDir, using: HuggingFaceTokenizerLoader())

        let shortCount = 4
        let longCount = 24

        let (shortTokens, shortFinishReason, longTokens, standaloneLongTokens):
            ([Int], FinishReason?, [Int], [Int]) = try await container.perform { context in
                let model = context.model
                let tokenizer = context.tokenizer
                let eos: Set<Int> = []

                let prompt = tokenizer.encode(
                    text: "Planets in order: Mercury, Venus, Earth,", addSpecialTokens: true)
                guard !prompt.isEmpty else { throw ModelTestError.emptyPrompt }

                let shortParams = GenerateParameters(maxTokens: shortCount, temperature: 0)
                let longParams = GenerateParameters(maxTokens: longCount, temperature: 0)

                // Same prompt on both rows: row 0 finishes at maxTokens=4 and
                // freezes; row 1 keeps decoding to 24.
                let (mixedRunner, mixedStreams) = try BatchDecodeRunner.make(
                    model: model,
                    tokenizer: tokenizer,
                    eosTokenIds: eos,
                    cohort: [
                        BatchSlotConfig(promptTokens: prompt, parameters: shortParams),
                        BatchSlotConfig(promptTokens: prompt, parameters: longParams),
                    ],
                    globalMaxTokens: longCount
                )
                try mixedRunner.run()

                // `run()` already buffered every chunk into the streams, so
                // draining now (post-completion) is safe and yields the
                // row's terminal finish-reason chunk.
                var shortReason: FinishReason?
                for try await chunk in mixedStreams[0] {
                    if let reason = chunk.finishReason { shortReason = reason }
                }

                // Reference: row 1's prompt/params decoded ALONE (B=1),
                // through the same runner/path.
                let (standaloneRunner, _) = try BatchDecodeRunner.make(
                    model: model,
                    tokenizer: tokenizer,
                    eosTokenIds: eos,
                    cohort: [BatchSlotConfig(promptTokens: prompt, parameters: longParams)],
                    globalMaxTokens: longCount
                )
                try standaloneRunner.run()

                return (
                    mixedRunner.slotTokens[0], shortReason, mixedRunner.slotTokens[1],
                    standaloneRunner.slotTokens[0]
                )
            }

        XCTAssertEqual(shortTokens.count, shortCount, "row 0 must stop at exactly its maxTokens")
        XCTAssertEqual(shortFinishReason, .length)
        XCTAssertEqual(
            longTokens, standaloneLongTokens,
            "row 1's full trajectory must match its standalone B=1 decode — a finished, "
                + "padded batchmate must not perturb the still-running row")
    }
}
