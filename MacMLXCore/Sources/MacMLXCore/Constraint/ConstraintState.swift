// Copyright © 2026 macMLX. English comments only.

/// The decode-time constraint state — either the generic JSON automaton (C1) or
/// a schema-specific one (C2) — behind one value-typed interface.
///
/// ``JSONConstraintProcessor`` holds a `ConstraintState`, classifies each
/// candidate token by whether ``walk(_:)`` accepts its bytes from here, and
/// commits the sampled token by reassigning to the walked result. Uniting the
/// two automata here keeps the processor free of C1/C2 branching.
public enum ConstraintState: Sendable {
    case json(JSONGrammarState)
    case schema(SchemaConstraintState)

    /// The initial state for a request's ``ResponseFormat``.
    ///
    /// - Parameters:
    ///   - format: the validated constraint.
    ///   - maxDepth: nesting cap for the generic JSON automaton (ignored by the
    ///     flat-object schema automaton).
    public static func initial(for format: ResponseFormat, maxDepth: Int = 64) -> ConstraintState {
        switch format {
        case .jsonObject:
            return .json(JSONGrammarState(maxDepth: maxDepth))
        case .jsonSchema(let schema):
            return .schema(SchemaConstraintState(schema: schema))
        }
    }

    /// Whether a complete document has been produced — the accept state in which
    /// EOS becomes legal.
    @inlinable
    public var isComplete: Bool {
        switch self {
        case .json(let state): return state.isComplete
        case .schema(let state): return state.isComplete
        }
    }

    /// Walk a byte sequence, returning the resulting state or `nil` if any byte
    /// is illegal from here.
    @inlinable
    public func walk<S: Sequence>(_ bytes: S) -> ConstraintState? where S.Element == UInt8 {
        switch self {
        case .json(let state):
            return state.walk(bytes).map(ConstraintState.json)
        case .schema(let state):
            return state.walk(bytes).map(ConstraintState.schema)
        }
    }

    /// Whether this exact byte sequence is a legal continuation from here.
    @inlinable
    public func accepts<S: Sequence>(_ bytes: S) -> Bool where S.Element == UInt8 {
        walk(bytes) != nil
    }

    /// A short, human-readable description of the current automaton position,
    /// for diagnostics only (e.g. the "no legal token — forcing EOS" log). Not a
    /// wire format and never parsed.
    public var diagnosticDescription: String {
        switch self {
        case .json(let state): return state.diagnosticDescription
        case .schema(let state): return state.diagnosticDescription
        }
    }
}
