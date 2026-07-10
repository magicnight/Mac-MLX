// Copyright © 2026 macMLX. English comments only.

/// Why an OpenAI `response_format` body could not be turned into a
/// ``ResponseFormat``. Both cases map to an HTTP 400 at the server boundary; the
/// ``description`` is the client-facing message.
public enum ResponseFormatError: Error, Equatable, Sendable, CustomStringConvertible {
    /// A structurally valid request that asks for a feature outside the
    /// supported subset (nested objects/arrays, combinators, non-object roots,
    /// …). Reported verbatim so a client learns exactly what to drop — never
    /// silently downgraded.
    case unsupportedFeature(String)

    /// A malformed `response_format` (wrong JSON shapes, an undeclared required
    /// property, an empty enum, …).
    case invalidFormat(String)

    public var description: String {
        switch self {
        case .unsupportedFeature(let detail):
            return "unsupported schema feature: \(detail)"
        case .invalidFormat(let detail):
            return "invalid response_format: \(detail)"
        }
    }
}
