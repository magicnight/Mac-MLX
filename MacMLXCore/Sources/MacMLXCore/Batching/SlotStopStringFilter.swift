// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Per-slot incremental stop-string detector for batched decode.
///
/// This is a faithful PORT of mlx-swift-lm's `StopStringFilter`
/// (`MLXLMCommon/Evaluate.swift:2162`), which is declared `internal` and so is
/// not importable from macMLX. One instance is owned by each ``BatchDecodeSlot``
/// (the upstream single-stream filter is not batch-aware; a batched decode needs
/// independent buffers per row).
///
/// Semantics preserved verbatim from upstream:
///  - Stop strings are matched longest-first (ties broken lexicographically) so
///    the earliest/longest match wins.
///  - `process(_:)` emits the text preceding the earliest complete stop match and
///    holds back any suffix that could still become the prefix of a stop string,
///    so a stop string split across token boundaries is never partially emitted.
///  - Once stopped, all further input is swallowed.
///  - `finish()` releases any held-back suffix when generation ends WITHOUT a stop
///    (e.g. EOS or max-tokens) — the suffix was a false-alarm partial match.
///
/// Pure `String` logic: no MLX, no tokenizer. Detokenization happens upstream of
/// this filter (see ``IncrementalTextDecoder``), so this type is fully unit
/// testable without a model or the Metal runtime.
struct SlotStopStringFilter {
    let stopStrings: [String]
    var buffer = ""
    var stopped = false

    init(stopStrings: Set<String>) {
        self.stopStrings = stopStrings.filter { !$0.isEmpty }.sorted {
            if $0.count == $1.count {
                return $0 < $1
            }
            return $0.count > $1.count
        }
    }

    var isEnabled: Bool {
        !stopStrings.isEmpty
    }

    /// Feed a freshly detokenized text chunk. Returns the text safe to emit now
    /// (or `nil` if nothing is emittable yet) and whether a stop string just
    /// completed. After `stopped` is `true`, always returns `(nil, true)`.
    mutating func process(_ chunk: String) -> (text: String?, stopped: Bool) {
        guard !stopped else {
            return (nil, true)
        }
        guard isEnabled else {
            return (chunk.isEmpty ? nil : chunk, false)
        }

        buffer += chunk

        if let stopRange = earliestStopRange(in: buffer) {
            let text = String(buffer[..<stopRange.lowerBound])
            buffer = ""
            stopped = true
            return (text.isEmpty ? nil : text, true)
        }

        let suffixLength = longestStopPrefixSuffixLength(in: buffer)
        let emitEnd = buffer.index(buffer.endIndex, offsetBy: -suffixLength)
        let text = String(buffer[..<emitEnd])
        buffer = String(buffer[emitEnd...])
        return (text.isEmpty ? nil : text, false)
    }

    /// Flush the held-back suffix when generation ends without a stop match.
    /// Returns `nil` once stopped (the suffix was already discarded) or when the
    /// filter is disabled / empty.
    mutating func finish() -> String? {
        guard isEnabled, !stopped, !buffer.isEmpty else {
            return nil
        }
        let text = buffer
        buffer = ""
        return text
    }

    private func earliestStopRange(in text: String) -> Range<String.Index>? {
        var earliest: Range<String.Index>?
        for stopString in stopStrings {
            guard let range = text.range(of: stopString) else {
                continue
            }
            if let current = earliest {
                if range.lowerBound < current.lowerBound {
                    earliest = range
                }
            } else {
                earliest = range
            }
        }
        return earliest
    }

    private func longestStopPrefixSuffixLength(in text: String) -> Int {
        var longest = 0
        for stopString in stopStrings {
            let maxLength = Swift.min(text.count, stopString.count - 1)
            guard maxLength > longest else {
                continue
            }
            for length in stride(from: maxLength, through: longest + 1, by: -1) {
                if text.suffix(length) == stopString.prefix(length) {
                    longest = length
                    break
                }
            }
        }
        return longest
    }
}
