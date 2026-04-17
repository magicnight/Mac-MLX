import Foundation

/// JSON encoder/decoder pair for store files that need sub-second
/// timestamp ordering (e.g. `ConversationStore`'s `updatedAt` — rapid
/// chat saves can fire 5+ times per second during active generation).
///
/// Design:
/// - **Encoding**: `.secondsSince1970` emits dates as `Double` (full 64-bit
///   precision). Pre-v0.3 `ConversationStore` used `.iso8601`, which the
///   default `ISO8601DateFormatter` encodes as `"2026-04-17T12:34:56Z"` —
///   whole seconds only. Rapid-fire saves all landed at the same encoded
///   timestamp, making `list()` sort order undefined.
/// - **Decoding**: tolerant of three historical shapes:
///   1. `Double` (the new default — `secondsSince1970`)
///   2. ISO-8601 string without fractional seconds (what pre-v0.3 wrote)
///   3. ISO-8601 string with fractional seconds (forward-compat)
///   Existing users' JSON files keep round-tripping; new writes use the
///   higher-precision shape.
public enum JSONCoding {

    /// Encoder producing pretty-printed, sorted-key JSON with high-
    /// precision timestamps. Output is deterministic (sorted keys) so
    /// atomic writes don't spuriously diff on re-save.
    public static func precisionEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .secondsSince1970
        return encoder
    }

    /// Decoder that accepts dates in any of three shapes: `Double`
    /// seconds-since-1970 (new), plain ISO-8601 string (pre-v0.3), or
    /// ISO-8601 string with fractional seconds.
    public static func tolerantDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { dec in
            let container = try dec.singleValueContainer()
            // Try numeric first — cheapest and the new default.
            if let seconds = try? container.decode(Double.self) {
                return Date(timeIntervalSince1970: seconds)
            }
            // Fall back to ISO-8601 string variants.
            let raw = try container.decode(String.self)
            if let date = Self.iso8601Basic.date(from: raw) {
                return date
            }
            if let date = Self.iso8601Fractional.date(from: raw) {
                return date
            }
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unrecognised date format: \(raw)"
            )
        }
        return decoder
    }

    // MARK: - Formatters (nonisolated lets)
    // ISO8601DateFormatter is thread-safe for date(from:) calls; we can
    // safely share two singletons.

    nonisolated(unsafe) private static let iso8601Basic: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    nonisolated(unsafe) private static let iso8601Fractional: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
}
