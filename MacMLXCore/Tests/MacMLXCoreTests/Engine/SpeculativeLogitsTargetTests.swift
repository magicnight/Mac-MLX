// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MacMLXCore

/// A speculative target whose `prepare` prefills the whole prompt and returns
/// logits, as VLM `Gemma4.prepare` does, on tiny random weights. mlx-swift-lm
/// 3.32.3's `SpeculativeTokenIterator` sampled such a target's first token
/// into the first verify pass and never emitted it, so the full path
/// (``FullPathVLM``) wrapped its target to leave the last prompt token for
/// that pass. 3.33.0 emits the token and the wrapper is gone; this holds the
/// line it stood on.
@Suite("Speculative decoding on a target that returns logits",
       .enabled(if: mlxMetallibIsAvailable, "needs the MLX Metal library"))
@MLXTestActor
struct SpeculativeLogitsTargetTests {

    private static var prompt: LMInput { LMInput(tokens: MLXArray((1...12).map(Int32.init))) }
    private static let parameters = GenerateParameters(maxTokens: 10, temperature: 0)

    private func plain(_ model: any LanguageModel) throws -> [Int] {
        var iterator = try TokenIterator(input: Self.prompt, model: model, parameters: Self.parameters)
        var out: [Int] = []
        while let token = iterator.next() { out.append(token) }
        return out
    }

    private func speculative(_ target: any LanguageModel, draft: any LanguageModel) throws -> [Int] {
        var iterator = try SpeculativeTokenIterator(
            input: Self.prompt, mainModel: target, draftModel: draft, parameters: Self.parameters,
            numDraftTokens: 3)
        var out: [Int] = []
        while let token = iterator.next() { out.append(token) }
        return out
    }

    /// Exact equality with greedy decoding holds on the seed-7 weights. Some
    /// seeds have a near-tie that the verify pass and single-token decode
    /// break differently, even on a plain `.tokens` target (seed 54 of the
    /// first 64), so a failure of the first expectation points at the weights
    /// or at MLX's numerics, not at the target's `prepare`.
    @Test
    func aTargetThatPrefillsThePromptSpeculatesItsWholeAnswer() throws {
        let target = try TinyRandomLanguageModel.qwen3.make()
        let draft = try TinyRandomLanguageModel.qwen3.make()
        let reference = try plain(target)
        try #require(reference.count == 10)
        #expect(try speculative(target, draft: draft) == reference, "a `.tokens` target speculates as it decodes")

        let logitsTarget = WholePromptPrepareModel(target)
        #expect(try plain(logitsTarget) == reference, "the stand-in decodes as the model it wraps")
        #expect(try speculative(logitsTarget, draft: draft) == reference, "the first token is emitted")
    }
}

/// A text model whose `prepare` runs the whole prompt and returns its last
/// logits, as VLM `Gemma4.prepare` does, instead of leaving the last token.
private final class WholePromptPrepareModel: Module, LanguageModel {
    let base: any LanguageModel

    init(_ base: any LanguageModel) {
        self.base = base
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let text = input.text.tokens.ndim == 1 ? input.text[text: .newAxis] : input.text
        return .logits(base(text, cache: cache, state: state))
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        base(input, cache: cache, state: state)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try base.newCache(parameters: parameters)
    }
}
