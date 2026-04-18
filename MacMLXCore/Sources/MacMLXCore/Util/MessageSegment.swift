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
}
