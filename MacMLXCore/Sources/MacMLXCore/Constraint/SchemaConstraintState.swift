// Copyright © 2026 macMLX. English comments only.

/// A byte-level automaton that constrains generation to a specific
/// ``JSONSchemaObject`` (Track C — C2).
///
/// Where ``JSONGrammarState`` accepts *any* well-formed JSON, this accepts only
/// the flat object described by a compiled schema: keys drawn from the declared
/// set (each at most once, all required ones present, in any order) and each
/// value matching its declared ``SchemaValueType``. It is the runtime companion
/// to ``ResponseFormatDecoder`` and, like ``JSONGrammarState``, is a pure value
/// type — token classification is a non-mutating ``walk(_:)`` fold, MLX-free and
/// unit-testable.
public struct SchemaConstraintState: Hashable, Sendable {

    /// The structural position within the object.
    @usableFromInline
    enum Phase: Hashable, Sendable {
        /// Before the object: whitespace then `{`.
        case beforeObject
        /// After `{` or after `,`. `afterComma` forbids the object close (no
        /// trailing comma).
        case expectKeyOrClose(afterComma: Bool)
        /// Inside a key string, matching declared names not yet emitted.
        case inKey(accumulated: [UInt8])
        /// A complete key was read; the `:` separator is required.
        case expectColon(key: String)
        /// After `:`; whitespace then the first byte of the typed value.
        case expectValue(key: String)
        /// Inside a typed value.
        case value(key: String, state: SchemaScalarState)
        /// A value completed; whitespace, `,`, or the object close `}`.
        case afterValue
        /// The object closed with all required keys present — the accept state.
        case done
    }

    @usableFromInline let schema: JSONSchemaObject
    @usableFromInline var emitted: Set<String>
    @usableFromInline var phase: Phase

    /// A fresh automaton positioned before the schema's object.
    public init(schema: JSONSchemaObject) {
        self.schema = schema
        self.emitted = []
        self.phase = .beforeObject
    }

    /// Whether the schema object has been fully and validly produced — the
    /// accept state, and the only state in which EOS is permitted.
    @inlinable
    public var isComplete: Bool { phase == .done }

    /// Advance over one byte, returning the resulting state or `nil` when the
    /// byte is illegal.
    @inlinable
    public func advanced(over byte: UInt8) -> SchemaConstraintState? {
        var next = self
        return next.applyInPlace(byte) ? next : nil
    }

    /// Fold ``advanced(over:)`` over a byte sequence; `nil` if any byte is
    /// rejected.
    @inlinable
    public func walk<S: Sequence>(_ bytes: S) -> SchemaConstraintState? where S.Element == UInt8 {
        var state = self
        for byte in bytes {
            guard state.applyInPlace(byte) else { return nil }
        }
        return state
    }

    /// A short description of the current structural position, for diagnostics
    /// (e.g. the constraint processor's "no legal token" log). Not a wire
    /// format — the reflected `phase`/`emitted` values are for humans.
    public var diagnosticDescription: String {
        "schema(phase: \(phase), emitted: \(emitted.sorted()), complete: \(isComplete))"
    }

    // MARK: - Transitions

    @usableFromInline
    mutating func applyInPlace(_ byte: UInt8) -> Bool {
        switch phase {
        case .beforeObject:
            if SchemaBytes.isWhitespace(byte) { return true }
            if byte == SchemaBytes.lBrace { phase = .expectKeyOrClose(afterComma: false); return true }
            return false

        case .expectKeyOrClose(let afterComma):
            return expectKeyOrClose(byte, afterComma: afterComma)

        case .inKey(let accumulated):
            return inKey(byte, accumulated: accumulated)

        case .expectColon(let key):
            if SchemaBytes.isWhitespace(byte) { return true }
            if byte == SchemaBytes.colon { phase = .expectValue(key: key); return true }
            return false

        case .expectValue(let key):
            if SchemaBytes.isWhitespace(byte) { return true }
            return startValue(byte, key: key)

        case .value(let key, let state):
            switch state.step(byte) {
            case .consumed(let next):
                phase = .value(key: key, state: next)
                return true
            case .completed:
                return finishValue(key)
            case .endedBefore:
                // A number ended before this byte: emit the key, move to
                // `afterValue`, and re-dispatch the byte there (one level only).
                guard finishValue(key) else { return false }
                return afterValue(byte)
            case .rejected:
                return false
            }

        case .afterValue:
            return afterValue(byte)

        case .done:
            return SchemaBytes.isWhitespace(byte)
        }
    }

    @usableFromInline
    mutating func expectKeyOrClose(_ byte: UInt8, afterComma: Bool) -> Bool {
        if SchemaBytes.isWhitespace(byte) { return true }
        if byte == SchemaBytes.quote {
            guard !remainingKeys.isEmpty else { return false }
            phase = .inKey(accumulated: [])
            return true
        }
        if byte == SchemaBytes.rBrace {
            guard !afterComma, requiredSatisfied else { return false }
            phase = .done
            return true
        }
        return false
    }

    @usableFromInline
    mutating func inKey(_ byte: UInt8, accumulated: [UInt8]) -> Bool {
        if byte == SchemaBytes.quote {
            // Close the key only if it exactly equals a remaining declared name.
            guard let name = remainingKeys.first(where: { Array($0.utf8) == accumulated }) else {
                return false
            }
            phase = .expectColon(key: name)
            return true
        }
        // Otherwise the byte must extend the key toward some remaining name.
        let position = accumulated.count
        let stillViable = remainingKeys.contains { name in
            let bytes = Array(name.utf8)
            return bytes.count > position
                && bytes[position] == byte
                && Array(bytes[0..<position]) == accumulated
        }
        guard stillViable else { return false }
        phase = .inKey(accumulated: accumulated + [byte])
        return true
    }

    /// Enter the typed value machine for `key` from its first byte.
    @usableFromInline
    mutating func startValue(_ byte: UInt8, key: String) -> Bool {
        guard let type = schema.property(named: key)?.type,
              let state = SchemaScalarState.start(byte, type: type) else { return false }
        phase = .value(key: key, state: state)
        return true
    }

    /// Record `key` as emitted and move to `afterValue`. Always succeeds; typed
    /// to return `Bool` so it composes in the transition expressions.
    @usableFromInline
    mutating func finishValue(_ key: String) -> Bool {
        emitted.insert(key)
        phase = .afterValue
        return true
    }

    @usableFromInline
    mutating func afterValue(_ byte: UInt8) -> Bool {
        if SchemaBytes.isWhitespace(byte) { return true }
        if byte == SchemaBytes.comma {
            // A comma promises another member. Once every declared key has been
            // emitted there is none left to promise, and `expectKeyOrClose(
            // afterComma: true)` would then admit nothing but whitespace — the
            // model could never close the object and would run to max_tokens
            // emitting blanks (seen on a real checkpoint). The only legal
            // continuations here are whitespace and the close.
            guard !remainingKeys.isEmpty else { return false }
            phase = .expectKeyOrClose(afterComma: true)
            return true
        }
        if byte == SchemaBytes.rBrace {
            guard requiredSatisfied else { return false }
            phase = .done
            return true
        }
        return false
    }

    // MARK: - Schema helpers

    /// Declared property names not yet emitted — the only keys a new member may
    /// open, which also enforces "each key at most once".
    @usableFromInline
    var remainingKeys: [String] {
        schema.properties.map { $0.name }.filter { !emitted.contains($0) }
    }

    /// Whether every required name has been emitted (checked at the object close).
    @usableFromInline
    var requiredSatisfied: Bool {
        schema.required.allSatisfy { emitted.contains($0) }
    }
}
