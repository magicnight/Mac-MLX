// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MacMLXCore

/// ``LastPromptTokenTarget`` on tiny random weights: a target whose `prepare`
/// prefills the whole prompt and returns logits, as VLM `Gemma4.prepare`
/// does, loses its first token under upstream's speculative iterator, and
/// wrapped it speculates to exactly the plain greedy output.
@Suite("LastPromptTokenTarget", .enabled(if: mlxMetallibIsAvailable, "needs the MLX Metal library"))
struct LastPromptTokenTargetTests {

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

    @Test
    func aTargetThatPrefillsThePromptSpeculatesItsWholeAnswer() throws {
        let target = try TinyRandomLanguageModel.qwen3.make()
        let draft = try TinyRandomLanguageModel.qwen3.make()
        let reference = try plain(target)
        try #require(reference.count == 10)
        #expect(try speculative(target, draft: draft) == reference, "a `.tokens` target speculates as it decodes")

        let logitsTarget = WholePromptPrepareModel(target)
        #expect(try plain(logitsTarget) == reference, "the stand-in decodes as the model it wraps")
        // Upstream's drop. When this fails, upstream has fixed it and
        // `LastPromptTokenTarget` can go.
        let dropped = try speculative(logitsTarget, draft: draft)
        #expect(Array(dropped.prefix(9)) == Array(reference.dropFirst()), "upstream emits the first token")

        #expect(try speculative(LastPromptTokenTarget(logitsTarget), draft: draft) == reference)
    }

    /// A short prompt or one with a mask is left to the wrapped model.
    @Test
    func anInputItDoesNotHandleGoesToTheWrappedModel() throws {
        let target = LastPromptTokenTarget(WholePromptPrepareModel(try TinyRandomLanguageModel.qwen3.make()))
        let cache = try target.newCache(parameters: nil)
        guard case .tokens(let single) = try target.prepare(
            LMInput(tokens: MLXArray([Int32(5)])), cache: cache, state: nil, prefill: .init())
        else {
            Issue.record("a one-token prompt is left whole for the first verify pass")
            return
        }
        #expect(single.tokens.asArray(Int.self) == [5])

        let masked = LMInput(text: .init(
            tokens: MLXArray([Int32(5), 6]).reshaped(1, 2), mask: MLXArray.ones([1, 2], dtype: .int8)))
        guard case .logits = try target.prepare(
            masked, cache: try target.newCache(parameters: nil), state: nil, prefill: .init())
        else {
            Issue.record("a masked prompt goes to the wrapped model, which returns logits")
            return
        }
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
