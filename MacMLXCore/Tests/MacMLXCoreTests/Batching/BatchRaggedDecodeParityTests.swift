// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MacMLXCore

/// Ragged batched decode against each prompt decoded on its own, for every
/// architecture ``BatchModelAllowlist`` names, on tiny random-weight models
/// (``TinyRandomLanguageModel``), so it runs without a checkpoint.
///
/// `BatchSchedulerModelTests` asserts the same parity on a real checkpoint but
/// only when one is on disk and the spike env is set; this one runs on every
/// xcodebuild pass. Two prompts of different lengths make the cohort ragged:
/// the shorter row is left-padded by the merge, so a model whose attention
/// mask does not come from the cache's `makeMask` attends to those zero-filled
/// pad columns at every decode step and its logits move, with no error. That
/// is how Gemma 2 is wired in mlx-swift-lm 3.32.3, and why it is off the
/// allowlist: ``testGemma2RaggedDecodeAttendsToLeftPadding`` pins the
/// divergence, so a release that fixes it fails here and the model can be
/// reconsidered.
///
/// The decode inputs are teacher-forced (the same tokens for both runs), so
/// one early argmax flip cannot cascade into a comparison of different
/// sequences.
final class BatchRaggedDecodeParityTests: XCTestCase {

    private let shortPrompt = [3, 17, 5]
    private let longPrompt = [9, 1, 22, 40, 7, 13, 2, 31, 11, 4]
    private let forcedTokens = [6, 19, 28, 12]

    /// MLX runs float32 matmuls as TF32 on M5 by default (`MLX_ENABLE_TF32`),
    /// and the `[2, 1]` batched forward and the `[1, 1]` standalone one do not
    /// lose the same mantissa bits. Measured on an M5 over these seven models:
    /// with TF32 off the worst row is 3.4e-6, with it on 0.018. The gate is the
    /// project's 1e-4 with TF32 off and about three times the TF32 worst case
    /// with it on; Gemma 2's left-padded row is off by 1.58 either way. For the
    /// tight gate:
    ///
    ///   TEST_RUNNER_MLX_ENABLE_TF32=0 xcodebuild test -scheme MacMLXCore \
    ///     -destination 'platform=macOS' -skipPackagePluginValidation \
    ///     -only-testing:MacMLXCoreTests/BatchRaggedDecodeParityTests
    private let tolerance: Float =
        ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" ? 1e-4 : 0.05

    private let allowlisted: [TinyRandomLanguageModel] = [
        .gemma, .gemma3Text, .gemma4Text, .qwen2, .qwen3, .llama,
    ]

    // MARK: - Allowlisted architectures

    func testAllowlistedArchitecturesMatchStandaloneDecodeOnARaggedCohort() throws {
        try requireTrustworthyMetalOrSkip()
        for architecture in allowlisted {
            let model = try architecture.make()
            XCTAssertTrue(
                BatchModelAllowlist.contains(model), "\(architecture) should be allowlisted")
            let errors = try raggedDecodeLogitErrors(model)
            print("[ragged-parity] \(architecture) max abs logit error per row: \(errors)")
            for (row, error) in errors.enumerated() {
                XCTAssertLessThan(
                    error, tolerance,
                    "\(architecture) row \(row): batched decode logits must match the "
                        + "standalone decode (max abs error \(error))")
            }
        }
    }

    /// The same parity through the production core: ``ModelBatchInferenceCore``
    /// admits both rows (B=1 prefill + merge) and decodes them batched; each
    /// row's tokens must equal the standalone argmax on the same inputs.
    func testModelBatchInferenceCoreMatchesStandaloneGreedyOnARaggedCohort() throws {
        try requireTrustworthyMetalOrSkip()
        for architecture in allowlisted {
            let model = try architecture.make()
            let core = ModelBatchInferenceCore(model: model)
            let configs = [shortPrompt, longPrompt].map {
                BatchSlotConfig(
                    promptTokens: $0,
                    parameters: GenerateParameters(maxTokens: 8, temperature: 0))
            }
            var batched = try core.admit(configs).map { [$0] }
            for token in forcedTokens {
                let next = try core.decode([token, token])
                for row in batched.indices { batched[row].append(next[row]) }
            }
            let standalone = try [shortPrompt, longPrompt].map { prompt in
                try standaloneLogits(model, prompt: prompt).map(argmax)
            }
            XCTAssertEqual(
                batched, standalone,
                "\(architecture): ragged batched greedy tokens must match standalone decode")
        }
    }

    // MARK: - Gemma 2

    func testModelBatchInferenceCoreRefusesGemma2() throws {
        try requireMLXRuntimeOrSkip()
        let core = ModelBatchInferenceCore(model: try TinyRandomLanguageModel.gemma2.make())
        let config = BatchSlotConfig(
            promptTokens: shortPrompt, parameters: GenerateParameters(maxTokens: 4, temperature: 0))
        XCTAssertThrowsError(try core.admit([config])) { error in
            XCTAssertEqual(error as? BatchUnsupportedError, .cacheNotBatchable)
        }
        XCTAssertEqual(core.rowCount, 0, "a refused admit must not leave a row behind")
    }

    /// Why Gemma 2 is off the allowlist. `Gemma2ModelInner` builds its mask
    /// with the `[KVCache]?` overload of `createAttentionMask`, which never
    /// asks the cache: `nil` at one token, a plain causal mask otherwise. The
    /// unpadded row matches; the left-padded row's decode attends to its pad
    /// columns and does not.
    func testGemma2RaggedDecodeAttendsToLeftPadding() throws {
        try requireTrustworthyMetalOrSkip()
        let errors = try raggedDecodeLogitErrors(try TinyRandomLanguageModel.gemma2.make())
        print("[ragged-parity] gemma2 max abs logit error per row: \(errors)")
        XCTAssertGreaterThan(
            errors[0], 0.5,
            "Gemma 2's left-padded row no longer diverges: if mlx-swift-lm now takes its "
                + "mask from the cache, Gemma2Model can go back on the allowlist")
        XCTAssertLessThan(errors[1], tolerance, "the unpadded row has no pad columns to leak")
    }

    // MARK: - Helpers

    /// Per row, the max absolute logit difference over the forced decode steps
    /// between the ragged batched decode and that prompt decoded alone. The
    /// batched side follows ``ModelBatchInferenceCore``'s admit and decode
    /// (B=1 prefill into stock caches, wrapped with `BatchKVCache.singleRow`
    /// and merged with `extend`, then `[B, 1]` forwards) but bypasses its
    /// coverage gate, so it can measure a model the allowlist refuses.
    private func raggedDecodeLogitErrors(_ model: any LanguageModel) throws -> [Float] {
        let prompts = [shortPrompt, longPrompt]
        var batchCaches: [BatchKVCache]?
        for prompt in prompts {
            let stock = try model.newCache(parameters: nil)
            _ = model(MLXArray(prompt.map { Int32($0) }, [1, prompt.count]), cache: stock)
            let row = stock.map { layer -> BatchKVCache in
                let state = layer.state
                return BatchKVCache.singleRow(keys: state[0], values: state[1])
            }
            if let caches = batchCaches {
                for index in caches.indices { caches[index].extend(other: row[index]) }
            } else {
                batchCaches = row
            }
        }
        let caches = try XCTUnwrap(batchCaches)
        XCTAssertTrue(
            BatchCacheConverter.makeBatchCaches(
                from: try model.newCache(parameters: nil), leftPadding: [0]) != nil,
            "the model's caches must pass the dense-cache gate for this to be the batched path")

        var batched: [[MLXArray]] = prompts.map { _ in [] }
        for token in forcedTokens {
            let logits = model(MLXArray([Int32(token), Int32(token)], [2, 1]), cache: caches)
            for row in prompts.indices {
                batched[row].append(logits[row, -1])
            }
        }

        return try prompts.indices.map { row in
            let reference = try standaloneLogits(model, prompt: prompts[row])
            // Step 0 is the prefill's own logits, which the batched side never
            // recomputes; compare the decode steps.
            return zip(batched[row], reference.dropFirst())
                .map { abs($0 - $1).max().item(Float.self) }
                .max() ?? .infinity
        }
    }

    /// Last-position logits `[vocab]` of `prompt` decoded alone through stock
    /// caches: the prefill, then one entry per forced decode token.
    private func standaloneLogits(_ model: any LanguageModel, prompt: [Int]) throws -> [MLXArray] {
        let cache = try model.newCache(parameters: nil)
        var out = [model(MLXArray(prompt.map { Int32($0) }, [1, prompt.count]), cache: cache)[0, -1]]
        for token in forcedTokens {
            out.append(model(MLXArray([Int32(token)], [1, 1]), cache: cache)[0, -1])
        }
        return out
    }

    private func argmax(_ logits: MLXArray) -> Int {
        logits.argMax().item(Int.self)
    }
}
