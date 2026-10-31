// Copyright © 2026 macMLX. English comments only.

import MLX
import MLXLMCommon
import MLXNN

/// A speculative-decoding target whose `prepare` leaves the prompt's last
/// token for the first verify pass, as an LLM's does (`.tokens`), whatever
/// the wrapped model's own `prepare` returns.
///
/// mlx-swift-lm 3.32.3's `SpeculativeTokenIterator` mishandles a target whose
/// `prepare` returns `.logits`: it samples the first token into the input of
/// the first verify pass and never emits it, and the draft model never sees
/// it. VLM `Gemma4.prepare` prefills the whole prompt and returns `.logits`,
/// so on the full path (``FullPathVLM``) every speculative answer lost its
/// first token and the draft ran a token behind. Wrapped, the iterator takes
/// the `.tokens` branch it takes on the text load.
///
/// Applies to a text-only `[N]` prompt with no mask and no state, which is
/// what the full path speculates on (an image request never speculates, and
/// Gemma 4 starts with no iterator state); any other input goes to the
/// wrapped model as it is. Everything but `prepare` is the wrapped model's.
final class LastPromptTokenTarget: Module, LanguageModel {

    let base: any LanguageModel

    init(_ base: any LanguageModel) {
        self.base = base
        super.init()
    }

    var toolCallFormat: ToolCallFormat? { base.toolCallFormat }
    var reasoningConfig: ReasoningConfig? { base.reasoningConfig }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        base.sanitize(weights: weights)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        guard !FullPathVLM.carriesMedia(input), input.text.mask == nil, tokens.ndim == 1, state == nil else {
            return try base.prepare(input, cache: cache, state: state, prefill: prefill)
        }
        let count = tokens.dim(0)
        guard count > 1 else { return .tokens(input.text) }
        let last = tokens[(count - 1)...]
        let head = LMInput(text: .init(tokens: tokens[..<(count - 1)]))
        switch try base.prepare(head, cache: cache, state: nil, prefill: prefill) {
        case .tokens(let remaining):
            return .tokens(.init(tokens: concatenated([remaining.tokens, last])))
        case .logits:
            // The head is in the cache; its logits predict the last prompt
            // token, which is known.
            return .tokens(.init(tokens: last))
        }
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        base(input, cache: cache, state: state)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        base(inputs, cache: cache)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try base.newCache(parameters: parameters)
    }

    func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus {
        try base.cacheStatus(parameters: parameters)
    }
}
