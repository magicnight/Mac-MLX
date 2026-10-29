// Copyright © 2026 macMLX. English comments only.

import MLXLMCommon
import Testing

@testable import MacMLXCore

/// The history-boundary decision (``HistoryBoundary/decide``): when the engine
/// splits a request's prefill at the end of its history and stores the cache
/// there. MLX-free; the engine-level behaviour on real weights is
/// `HistoryBoundaryPromptCacheSmokeTests`.
@Suite("History boundary")
struct HistoryBoundaryTests {

    private static let minimum = HistoryBoundary.minimumBoundarySnapshotTokens

    /// A prompt of `history` tokens followed by a generation prompt of
    /// `tail` tokens, the shape every measured template renders.
    private static func prompt(history: Int, tail: Int = 3) -> (history: [Int], prompt: [Int]) {
        let history = Array(0..<history)
        return (history, history + Array(100_000..<(100_000 + tail)))
    }

    @Test("A boundary that is a prefix and clears the minimum is snapshotted")
    func snapshotsAQualifyingBoundary() {
        let (history, prompt) = Self.prompt(history: 600)
        let decision = HistoryBoundary.decide(
            historyTokens: history, renderedPromptTokens: prompt,
            promptTokens: prompt, reusedTokenCount: 100)
        #expect(decision == .snapshot(boundary: 600))
    }

    @Test("A cold prompt counts the whole history as gain")
    func snapshotsOnAMiss() {
        let (history, prompt) = Self.prompt(history: Self.minimum)
        let decision = HistoryBoundary.decide(
            historyTokens: history, renderedPromptTokens: prompt,
            promptTokens: prompt, reusedTokenCount: 0)
        #expect(decision == .snapshot(boundary: Self.minimum))
    }

    @Test("A gain one token short of the minimum is skipped")
    func skipsBelowTheMinimumGain() {
        let (history, prompt) = Self.prompt(history: 1_000)
        let reused = 1_000 - Self.minimum + 1
        let decision = HistoryBoundary.decide(
            historyTokens: history, renderedPromptTokens: prompt,
            promptTokens: prompt, reusedTokenCount: reused)
        guard case .skip = decision else {
            Issue.record("expected a skip, got \(decision)")
            return
        }
        // Exactly the minimum qualifies.
        #expect(
            HistoryBoundary.decide(
                historyTokens: history, renderedPromptTokens: prompt,
                promptTokens: prompt, reusedTokenCount: reused - 1)
                == .snapshot(boundary: 1_000))
    }

    @Test("A history render that is not a prefix of the prompt is skipped")
    func skipsANonPrefixRender() {
        var (history, prompt) = Self.prompt(history: 600)
        // The render diverges one token before its end, the way Qwen3.6's
        // full turn-1 prompt diverges from turn 2 on its last token.
        history[history.count - 1] = 99_999
        let decision = HistoryBoundary.decide(
            historyTokens: history, renderedPromptTokens: prompt,
            promptTokens: prompt, reusedTokenCount: 0)
        guard case .skip = decision else {
            Issue.record("expected a skip, got \(decision)")
            return
        }
    }

    @Test("A generation-prompt render that differs from the prompt is skipped")
    func skipsWhenTheRenderIsNotGenerations() {
        let (history, prompt) = Self.prompt(history: 600)
        var rendered = prompt
        rendered[rendered.count - 1] += 1
        let decision = HistoryBoundary.decide(
            historyTokens: history, renderedPromptTokens: rendered,
            promptTokens: prompt, reusedTokenCount: 0)
        guard case .skip = decision else {
            Issue.record("expected a skip, got \(decision)")
            return
        }
    }

    @Test("A boundary at or past the prompt's end is skipped")
    func skipsABoundaryThatLeavesNothingToGenerateFrom() {
        // A template whose generation prompt renders to nothing: the history
        // is the whole prompt, and no token would be left for the step that
        // samples the first generated token.
        let whole = Array(0..<600)
        #expect(
            HistoryBoundary.decide(
                historyTokens: whole, renderedPromptTokens: whole,
                promptTokens: whole, reusedTokenCount: 0)
                != .snapshot(boundary: 600))
        let longer = whole + [7]
        #expect(
            HistoryBoundary.decide(
                historyTokens: longer, renderedPromptTokens: whole,
                promptTokens: whole, reusedTokenCount: 0)
                != .snapshot(boundary: 601))
    }

    @Test("A turn's own entry is skipped only after a snapshot of an untrimmable cache")
    func recordsTheTurnEntryUnlessTheSnapshotCannotBeTrimmedPast() {
        // Empty caches: trimmability is a property of the cache type here.
        let hybrid: [any KVCache] = [MambaCache(), KVCacheSimple()]
        let plain: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
        #expect(!MLXSwiftEngine.recordsTurnEntry(historySnapshotStored: true, caches: hybrid))
        #expect(MLXSwiftEngine.recordsTurnEntry(historySnapshotStored: false, caches: hybrid))
        #expect(MLXSwiftEngine.recordsTurnEntry(historySnapshotStored: true, caches: plain))
        #expect(MLXSwiftEngine.recordsTurnEntry(historySnapshotStored: false, caches: plain))
    }

    @Test("A cold file promotes with the class it was demoted with, old files as assistant")
    func promotesWithTheStoredCacheType() {
        #expect(PromptCacheStore.promotedCacheType(["cacheType": "user"]) == .user)
        #expect(PromptCacheStore.promotedCacheType(["cacheType": "system"]) == .system)
        #expect(PromptCacheStore.promotedCacheType([:]) == .assistant)
        #expect(PromptCacheStore.promotedCacheType(["cacheType": "bogus"]) == .assistant)
    }

    @Test("A boundary the cache already reached is skipped")
    func skipsWhenTheCacheAlreadyServedTheBoundary() {
        // The exact repeat of a turn: the previous snapshot is a strict prefix
        // of this prompt and served it whole.
        let (history, prompt) = Self.prompt(history: 600)
        for reused in [600, 601] {
            let decision = HistoryBoundary.decide(
                historyTokens: history, renderedPromptTokens: prompt,
                promptTokens: prompt, reusedTokenCount: reused)
            guard case .skip = decision else {
                Issue.record("reused \(reused): expected a skip, got \(decision)")
                return
            }
        }
    }
}
