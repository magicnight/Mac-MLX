// Copyright © 2026 macMLX. English comments only.

/// Pure (MLX-free) arithmetic behind ragged-cohort prefill assembly for
/// ``BatchKVCache``.
///
/// These are the host-side book-keeping rules a scheduler runs BEFORE touching
/// MLX: turning a cohort of unequal-length prompts into one left-padded
/// `[B, Lmax]` block plus the per-row `leftPadding` that ``BatchKVCache`` needs.
/// Kept free of any `MLXArray` so the semantics are unit-tested under a plain
/// `swift test` in CI (no Metal), mirroring how the A2a scheduling core is
/// tested MLX-free.
///
/// Left-padding convention (matches mlx-lm): prompts are right-justified to the
/// longest length, pad tokens fill the LEFT, and `leftPadding[i] = Lmax - len[i]`
/// so every row's last real token sits at the final column — which is why the
/// batched forward can sample every row from the single last position.
public enum BatchPrefillAssembly {
    /// Left-pad a cohort to a common width.
    ///
    /// - Parameters:
    ///   - prompts: per-row token sequences (may differ in length).
    ///   - padToken: the id used for left padding. Its value is irrelevant to the
    ///     result because ``BatchKVCache``'s mask excludes pad columns from
    ///     attention — any in-vocabulary id is safe.
    /// - Returns: `padded` rows (all length `Lmax`, row-aligned with `prompts`)
    ///   and the per-row `leftPadding`. An empty cohort yields two empty arrays.
    public static func leftPad(
        prompts: [[Int]], padToken: Int
    ) -> (padded: [[Int]], leftPadding: [Int]) {
        guard let maxLength = prompts.map({ $0.count }).max() else {
            return ([], [])
        }
        var padded: [[Int]] = []
        var leftPadding: [Int] = []
        padded.reserveCapacity(prompts.count)
        leftPadding.reserveCapacity(prompts.count)
        for prompt in prompts {
            let pad = maxLength - prompt.count
            padded.append(Array(repeating: padToken, count: pad) + prompt)
            leftPadding.append(pad)
        }
        return (padded, leftPadding)
    }

    /// Per-row left-padding for a cohort given only its prompt lengths.
    /// `leftPadding[i] = max(lengths) - lengths[i]`.
    public static func leftPadding(forLengths lengths: [Int]) -> [Int] {
        guard let maxLength = lengths.max() else { return [] }
        return lengths.map { maxLength - $0 }
    }

    /// The per-row RoPE offsets a fresh ``BatchKVCache`` starts at,
    /// `-leftPadding[i]`, so each row's first real token is RoPE position 0.
    public static func initialOffsets(leftPadding: [Int]) -> [Int] {
        leftPadding.map { -$0 }
    }

    /// The left-shift `filter` applies after an eviction to reclaim now-common
    /// pad columns: the minimum surviving left-padding, clamped at 0.
    public static func filterLeftShift(leftPadding: [Int]) -> Int {
        max(0, leftPadding.min() ?? 0)
    }
}
