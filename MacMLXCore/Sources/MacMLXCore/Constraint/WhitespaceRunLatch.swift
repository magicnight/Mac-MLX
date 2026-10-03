// Copyright © 2026 macMLX. English comments only.

/// Detects a model that has started emitting nothing but whitespace where the
/// grammar permits any amount of it.
///
/// JSON allows unbounded whitespace between tokens, so a constrained decode can
/// spin: after `{` the automaton accepts spaces, tabs and newlines forever, and
/// a model whose preferred continuation is masked may keep sampling them until
/// `max_tokens` — a 200 with an empty, truncated document (seen on a real
/// checkpoint: `{\n  \t\n  \t\n …`). Once `threshold` consecutive sampled
/// tokens were whitespace-only, ``isActive`` latches on for the rest of the
/// generation and the constraint processor withholds whitespace-only tokens at
/// structural positions (never inside a string, where spaces are data). The
/// latch does not reset: a model that got here has shown a pathological
/// preference, and releasing it would let it alternate between whitespace runs
/// and forced structural tokens, burning the budget just as surely (the same
/// rule as mlx-swift-lm's `WhitespaceRunTracker`).
struct WhitespaceRunLatch: Equatable, Sendable {
    /// Consecutive whitespace-only sampled tokens that trip the latch.
    let threshold: Int
    private(set) var consecutive = 0
    private(set) var isActive = false

    init(threshold: Int = 3) {
        self.threshold = Swift.max(1, threshold)
    }

    /// Record the token that was just sampled.
    mutating func record(whitespaceOnly: Bool) {
        if whitespaceOnly {
            consecutive += 1
            if consecutive >= threshold { isActive = true }
        } else {
            consecutive = 0
        }
    }
}
