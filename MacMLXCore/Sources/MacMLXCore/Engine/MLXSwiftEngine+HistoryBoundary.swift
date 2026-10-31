// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import MLXLMCommon

/// The history-boundary prompt-cache snapshot (see ``HistoryBoundary``): the
/// render that finds the boundary and the prefill that stops at it. Both run
/// inside `runLLMGeneration`'s `container.perform`, against the resident
/// model's own context.
extension MLXSwiftEngine {

    /// Where to split this request's prefill, decided from the chat rendered
    /// with and without the generation prompt.
    ///
    /// The render mirrors the input processor that produced `promptTokens`:
    /// `messageGenerator`, the generator the engine took at load as the one
    /// that processor renders with (for an LLM the factory's choice: the
    /// configuration's, else the model's, else the default; for a full-path
    /// VLM the configuration's, else its processor's), over the same
    /// `UserInput` (messages, tools, template kwargs), then the resident
    /// tokenizer's chat template, chat-template override included, through the
    /// same call generation's render goes through, with `addGenerationPrompt`
    /// false. ``HistoryBoundary/decide`` accepts it only as an exact prefix of
    /// `promptTokens`, so a render that ever drifted from generation's would
    /// cost the snapshot, never correctness: any prefix of the prompt is a
    /// valid cache to store.
    ///
    /// Returns `.skip` without rendering when `cache` can always be trimmed
    /// (see ``cacheNeedsHistorySnapshot(_:)``) or no boundary could clear the
    /// minimum gain, and when the tokenizer is not the swift-transformers
    /// bridge, has no chat template, or the render throws.
    static func historyBoundary(
        for input: UserInput,
        messageGenerator: any MessageGenerator,
        cache: [any KVCache],
        context: ModelContext,
        promptTokens: [Int],
        reusedTokenCount: Int
    ) -> HistoryBoundary {
        guard cacheNeedsHistorySnapshot(cache) else {
            return .skip(reason: "every layer of the cache can be trimmed")
        }
        // B < prompt and B - reused >= minimum together need more than the
        // minimum still to prefill; short of that, skip the two renders.
        guard promptTokens.count - reusedTokenCount > HistoryBoundary.minimumBoundarySnapshotTokens else {
            return .skip(
                reason: "\(promptTokens.count - reusedTokenCount) tokens to prefill, "
                    + "not more than \(HistoryBoundary.minimumBoundarySnapshotTokens)")
        }
        guard let tokenizer = context.tokenizer as? TokenizerBridge else {
            return .skip(reason: "the resident tokenizer cannot render without the generation prompt")
        }
        let messages = messageGenerator.generate(from: input)
        do {
            let history = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext,
                addGenerationPrompt: false)
            return HistoryBoundary.decide(
                historyTokens: history, promptTokens: promptTokens, reusedTokenCount: reusedTokenCount)
        } catch {
            return .skip(reason: "rendering the history failed: \(error)")
        }
    }

    /// Whether `caches` contain a layer that refuses a trim now or can come to
    /// refuse one: a sliding-window `RotatingKVCache` (once past its window),
    /// an `ArraysCache` such as the `MambaCache` of a linear-attention layer,
    /// or any cache that reports itself untrimmable. `CacheList`s are searched
    /// to their leaves.
    ///
    /// A cache of only `KVCacheSimple` / `QuantizedKVCache` layers can always
    /// be trimmed, so the turn's own entry already serves every prefix of it,
    /// the next turn's history included, through a trim; a snapshot there
    /// would only duplicate it, and at agent sizes (about 2.2 GB for a dense
    /// 8B at 17k tokens) cost a second SSD write per turn.
    static func cacheNeedsHistorySnapshot(_ caches: [any KVCache]) -> Bool {
        var needs = false
        for cache in caches {
            if let list = cache as? CacheList {
                // `mapChildren` walks every leaf; each is returned unchanged.
                list.mapChildren { leaf in
                    if leafRefusesTrim(leaf) { needs = true }
                    return leaf
                }
            } else if leafRefusesTrim(cache) {
                needs = true
            }
        }
        return needs
    }

    private static func leafRefusesTrim(_ cache: any KVCache) -> Bool {
        cache is RotatingKVCache || cache is ArraysCache || !cache.isTrimmable
    }

    /// Prefill `tokens` into `cache` and evaluate it, generating nothing.
    ///
    /// mlx-swift-lm 3.32.3 has no public prefill-only entry point, so this
    /// builds a `TokenIterator` over the slice: its initializer runs the
    /// model's own chunked prefill with `prefill` (the chunking generation
    /// uses), then steps the last token, which puts every one of `tokens` into
    /// `cache`. The token it samples (greedy, no processor, so no random state
    /// is drawn) is dropped, and the iterator with it; `cache` is mutated in
    /// place. The cache state is evaluated before returning — the sampled
    /// token alone would not force a recurrent layer's next state — so a copy
    /// taken now holds computed arrays rather than a graph over the prefill.
    /// `state` seeds the iterator as it seeds generation's (see
    /// ``FullPathVLM/iteratorState()``); the state the iterator ends with is
    /// dropped with it, so the generation after this starts from the same
    /// seed on a now-warm cache.
    ///
    /// - Returns: the wall time the prefill took, which the caller adds to the
    ///   request's prompt time.
    static func prefillHistory(
        _ tokens: MLXArray,
        into cache: [any KVCache],
        context: ModelContext,
        state: LMOutput.State? = nil,
        prefill: PrefillParameters
    ) throws -> TimeInterval {
        let start = Date.timeIntervalSinceReferenceDate
        _ = try TokenIterator(
            input: LMInput(tokens: tokens),
            model: context.model,
            cache: cache,
            state: state,
            processor: nil,
            sampler: ArgMaxSampler(),
            prefill: prefill
        )
        eval(cache.flatMap { $0.state })
        return Date.timeIntervalSinceReferenceDate - start
    }

    /// Whether a finished turn records its own entry (the prompt, the
    /// generated tokens and the stop token) besides the history-boundary
    /// snapshot.
    ///
    /// Not when this turn stored a snapshot and `caches` cannot be trimmed.
    /// Such an entry serves only a prompt that extends it exactly, and the
    /// templates the snapshot is for (Qwen3.6, Mellum2) never re-render a
    /// turn that way; even an exact retry needs a one-token trim these caches
    /// refuse. It would cost as much as the snapshot (about 1.2 GB on
    /// Qwen3.6-27B at 17k tokens) and double the memory or SSD writes of every
    /// turn. A template that does re-render the turn verbatim (Gemma 4 E4B,
    /// once past its 512-token sliding window) loses only the reuse of the
    /// generated tokens; the snapshot still serves its history. A cache that
    /// can still be trimmed (a sliding window not yet filled) keeps recording
    /// the entry, which replaces the snapshot on insert.
    static func recordsTurnEntry(historySnapshotStored: Bool, caches: [any KVCache]) -> Bool {
        !historySnapshotStored || MLXLMCommon.canTrimPromptCache(caches)
    }

    /// `info` with the boundary prefill counted as prompt work: the iterator
    /// that reported it prefilled only the tokens after the boundary, so the
    /// tokens and time spent before it are added back. `cachedPromptTokens` is
    /// untouched; a boundary prefill is computed, not served from the cache.
    static func completionInfo(
        _ info: GenerateCompletionInfo, addingPrefillOf tokens: Int, seconds: TimeInterval
    ) -> GenerateCompletionInfo {
        GenerateCompletionInfo(
            promptTokenCount: info.promptTokenCount + tokens,
            cachedPromptTokenCount: info.cachedPromptTokenCount,
            generationTokenCount: info.generationTokenCount,
            promptTime: info.promptTime + seconds,
            generationTime: info.generateTime,
            stopReason: info.stopReason,
            proposedDraftTokens: info.proposedDraftTokens,
            acceptedDraftTokens: info.acceptedDraftTokens,
            passthroughReason: info.passthroughReason,
            speculativeDecodingTelemetry: info.speculativeDecodingTelemetry,
            rejectedToolCallCount: info.rejectedToolCallCount,
            recoveredToolCallCount: info.recoveredToolCallCount
        )
    }
}
