// Copyright © 2026 macMLX. English comments only.

/// Where a request's prompt is split for the history-boundary prompt-cache
/// snapshot, or why it is not.
///
/// Agent clients (Codex over `/v1/responses`, Claude Code over `/v1/messages`)
/// replay the whole conversation every turn. The entry the engine records
/// after a turn — the prompt, the generated tokens and the stop token — is not
/// a prefix of the next turn's prompt on templates that render a previous
/// assistant turn differently from how the model produced it (Qwen3.6 and
/// Mellum2 render a turn's reasoning only for the turns after the last user
/// query, and only as the client replays it, so a new query drops it from
/// history), and the full prompt is not one either
/// when the generation prompt itself is re-rendered (Qwen3.6 opens `<think>`
/// `\n` and closes the same turn as `<think>` `\n\n` `</think>`). Serving the
/// shared prefix out of either then needs a trim, which hybrid
/// (linear-attention) and sliding-window caches refuse, so the next turn paid a
/// full prefill. The end of the last message before the generation prompt —
/// the boundary `B`, the chat rendered with `add_generation_prompt=false` — is
/// a prefix of the next turn on every template measured, so the engine stores
/// the cache at `B` as well, and the next turn reuses it whole, through the
/// store's "shorter" branch, with no trim.
///
/// MLX-free: the engine renders the two token sequences and hands them here.
enum HistoryBoundary: Equatable, Sendable {
    /// Snapshot the cache after the first `boundary` prompt tokens.
    case snapshot(boundary: Int)
    /// No snapshot for this request; `reason` goes to the debug log.
    case skip(reason: String)

    /// The fewest tokens a snapshot must add beyond what the prompt cache
    /// already served. Below it the boundary is skipped: a short chat would pay
    /// a cache copy and a hot-tier entry per turn to save a prefill of a few
    /// hundred tokens at most.
    static let minimumBoundarySnapshotTokens = 256

    /// Decide the boundary for one request.
    ///
    /// Only the prefix check guards correctness, and it is complete: the key
    /// stored is `promptTokens[0..<boundary]` and the cache is computed from
    /// exactly those tokens, so any prefix is a valid entry. Whether it is
    /// also where the next turn's prompt diverges is what the render is for,
    /// and a render that disagreed with generation's would only cost the
    /// reuse.
    ///
    /// - Parameters:
    ///   - historyTokens: the request's chat rendered without the generation
    ///     prompt.
    ///   - promptTokens: the prompt the engine is about to prefill.
    ///   - reusedTokenCount: how many leading prompt tokens the prompt cache
    ///     already served (0 on a miss).
    /// - Returns: `.snapshot` only when `historyTokens` is an exact prefix of
    ///   the prompt, ends strictly inside the part still to prefill (so at
    ///   least one token follows it for the generation step) and adds at least
    ///   ``minimumBoundarySnapshotTokens`` beyond the reused prefix.
    static func decide(
        historyTokens: [Int],
        promptTokens: [Int],
        reusedTokenCount: Int
    ) -> HistoryBoundary {
        let boundary = historyTokens.count
        guard boundary < promptTokens.count else {
            return .skip(reason: "the history render (\(boundary) tokens) does not end before the prompt does")
        }
        guard promptTokens[0..<boundary].elementsEqual(historyTokens) else {
            return .skip(reason: "the history render is not a prefix of the prompt")
        }
        guard boundary > reusedTokenCount else {
            return .skip(reason: "the prompt cache already served \(reusedTokenCount) tokens, past the boundary at \(boundary)")
        }
        guard boundary - reusedTokenCount >= minimumBoundarySnapshotTokens else {
            return .skip(
                reason: "the boundary at \(boundary) adds \(boundary - reusedTokenCount) tokens beyond the "
                    + "\(reusedTokenCount) reused, under \(minimumBoundarySnapshotTokens)")
        }
        return .snapshot(boundary: boundary)
    }
}
