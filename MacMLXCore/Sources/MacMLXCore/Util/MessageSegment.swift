import Foundation

/// Ordered segment in a rendered assistant message.
///
/// Chat models (qwen3, gemma4, deepseek-r1) emit reasoning wrapped in
/// `<think>…</think>` tags and the actual answer after. The UI wants
/// to collapse the reasoning into a toggle and render the answer
/// normally, so we parse the raw stream text into ordered segments
/// first and let the view decide how to present each.
public enum MessageSegment: Equatable, Sendable {
    /// Plain text (can still contain Markdown).
    case text(String)
    /// Reasoning block. `isClosed == false` means we're still streaming
    /// inside the `<think>…`, so the view should show a spinner rather
    /// than the static "thought" label.
    case think(String, isClosed: Bool)
}

public enum MessageSegmenter {

    private static let openTag = "<think>"
    private static let closeTag = "</think>"

    /// Parse a possibly-streaming message body into ordered segments.
    ///
    /// Handles four cases:
    /// 1. Ordinary `<think>…</think>answer` — one think + one text.
    /// 2. Streaming `<think>partial` (no close yet) — one open think.
    /// 3. Missing opener `…</think>answer` — qwen3's chat template
    ///    injects `<think>` in the prompt, so the model only emits the
    ///    close. Treat everything before `</think>` as think content.
    /// 4. No tags at all — single `.text` segment.
    public static func parse(_ content: String) -> [MessageSegment] {
        var segments: [MessageSegment] = []
        var cursor = content.startIndex

        // Implicit-open (qwen3) handling — run exactly once up-front.
        if let firstOpen = content.range(of: openTag),
           let firstClose = content.range(of: closeTag),
           firstClose.lowerBound < firstOpen.lowerBound {
            // close appears before any open → treat everything from
            // start up to the close as implicit think content.
            let implicit = String(content[cursor..<firstClose.lowerBound])
            if !implicit.isEmpty {
                segments.append(.think(implicit, isClosed: true))
            }
            cursor = firstClose.upperBound
        } else if content.range(of: openTag) == nil,
                  let firstClose = content.range(of: closeTag) {
            // no open tag at all but a close exists → same implicit case.
            let implicit = String(content[cursor..<firstClose.lowerBound])
            if !implicit.isEmpty {
                segments.append(.think(implicit, isClosed: true))
            }
            cursor = firstClose.upperBound
        }

        // Normal open/close pairs (and trailing open for streaming).
        while let openRange = content.range(of: openTag, range: cursor..<content.endIndex) {
            // Text before this open tag.
            let before = String(content[cursor..<openRange.lowerBound])
            if !before.isEmpty {
                segments.append(.text(before))
            }
            let afterOpen = openRange.upperBound
            if let closeRange = content.range(of: closeTag, range: afterOpen..<content.endIndex) {
                let think = String(content[afterOpen..<closeRange.lowerBound])
                segments.append(.think(think, isClosed: true))
                cursor = closeRange.upperBound
            } else {
                // Streaming: open but no close yet. Everything after
                // this open is an in-flight think block.
                let think = String(content[afterOpen..<content.endIndex])
                segments.append(.think(think, isClosed: false))
                cursor = content.endIndex
                break
            }
        }

        // Trailing text after the last close.
        let trailing = String(content[cursor..<content.endIndex])
        if !trailing.isEmpty {
            segments.append(.text(trailing))
        }

        return segments
    }

    /// Split complete assistant output into reasoning + answer for the
    /// `reasoning_content` API convention (DeepSeek / mlx-lm / LM Studio).
    ///
    /// Reasoning models emit `<think>…</think>` blocks; qwen3 only emits
    /// the closing tag because its chat template injects the opener into
    /// the prompt (handled by `parse`). Returns the joined reasoning text
    /// (`nil` when there was none, so non-reasoning models are untouched)
    /// and the answer with every think block removed.
    ///
    /// - Note: Operates on the *complete* text — streaming callers need an
    ///   incremental splitter that tracks the `<think>`/`</think>` boundary
    ///   across chunks (tags can be split mid-token).
    public static func splitReasoning(
        _ content: String
    ) -> (reasoning: String?, answer: String) {
        var reasoning = ""
        var answer = ""
        for segment in parse(content) {
            switch segment {
            case .think(let text, _): reasoning += text
            case .text(let text): answer += text
            }
        }
        return (reasoning.isEmpty ? nil : reasoning, answer)
    }

    /// Whether `promptText` (a rendered chat-template prompt) ends inside
    /// an unclosed `<think>` block — the last `<think>` has no following
    /// `</think>`. qwen3's template injects exactly this, so the model's
    /// first generated token is reasoning even though the opening tag
    /// never appears in the output stream. Used to seed
    /// `ReasoningStreamSplitter` for streaming responses.
    public static func promptOpensThink(_ promptText: String) -> Bool {
        guard let lastOpen = promptText.range(of: openTag, options: .backwards)?.lowerBound
        else { return false }
        guard let lastClose = promptText.range(of: closeTag, options: .backwards)?.lowerBound
        else { return true }
        return lastOpen > lastClose
    }
}

/// Incremental reasoning/answer splitter for **streaming** responses.
///
/// Streaming can't see the whole message, so it can't retroactively
/// reclassify text once emitted. The key ambiguity — qwen3's template
/// injects `<think>` into the *prompt*, so the model's first token is
/// reasoning with no opening tag in the stream — is resolved by the
/// caller seeding `startInReasoning` from whether the prompt opened a
/// think block (see `InferenceEngine.promptOpensThinkBlock`). From there
/// this tracks the `<think>` / `</think>` boundary, including tags split
/// across chunk boundaries, and returns the reasoning + answer deltas for
/// each pushed chunk. Call `finish()` at end-of-stream to flush any tail.
public struct ReasoningStreamSplitter {
    private var inReasoning: Bool
    private var buffer = ""

    private static let open = "<think>"
    private static let close = "</think>"

    public init(startInReasoning: Bool) {
        self.inReasoning = startInReasoning
    }

    /// Feed the next chunk; returns the reasoning + answer deltas for it.
    public mutating func push(_ text: String) -> (reasoning: String, answer: String) {
        buffer += text
        var reasoning = ""
        var answer = ""

        // Consume every complete tag in order.
        while true {
            let openR = buffer.range(of: Self.open)
            let closeR = buffer.range(of: Self.close)
            let next: (range: Range<String.Index>, opens: Bool)?
            switch (openR, closeR) {
            case let (o?, c?): next = o.lowerBound < c.lowerBound ? (o, true) : (c, false)
            case let (o?, nil): next = (o, true)
            case let (nil, c?): next = (c, false)
            case (nil, nil): next = nil
            }
            guard let tag = next else { break }
            emit(String(buffer[buffer.startIndex..<tag.range.lowerBound]), &reasoning, &answer)
            inReasoning = tag.opens
            buffer = String(buffer[tag.range.upperBound...])
        }

        // Emit everything except a possible partial trailing tag (so we
        // never split a `<think>`/`</think>` across two deltas).
        let keep = partialTagSuffixLength(buffer)
        if buffer.count > keep {
            let idx = buffer.index(buffer.endIndex, offsetBy: -keep)
            emit(String(buffer[buffer.startIndex..<idx]), &reasoning, &answer)
            buffer = String(buffer[idx...])
        }
        return (reasoning, answer)
    }

    /// Flush any buffered tail (a trailing `<`-like fragment that turned
    /// out not to be a tag). Call once when the stream finishes.
    public mutating func finish() -> (reasoning: String, answer: String) {
        var reasoning = ""
        var answer = ""
        emit(buffer, &reasoning, &answer)
        buffer = ""
        return (reasoning, answer)
    }

    private func emit(_ s: String, _ reasoning: inout String, _ answer: inout String) {
        guard !s.isEmpty else { return }
        if inReasoning { reasoning += s } else { answer += s }
    }

    /// Length of the trailing substring of `s` that is a proper prefix of
    /// `<think>` or `</think>` — held back so we don't emit half a tag.
    private func partialTagSuffixLength(_ s: String) -> Int {
        var best = 0
        for tag in [Self.open, Self.close] {
            var len = min(tag.count - 1, s.count)
            while len >= 1 {
                if tag.hasPrefix(s.suffix(len)) {
                    best = max(best, len)
                    break
                }
                len -= 1
            }
        }
        return best
    }
}
