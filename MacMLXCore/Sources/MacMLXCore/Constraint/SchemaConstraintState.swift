// Copyright © 2026 macMLX. English comments only.

/// A byte-level automaton that constrains generation to a specific compiled
/// schema, a root ``SchemaValueType`` (Track C — C2).
///
/// Where ``JSONGrammarState`` accepts *any* well-formed JSON, this accepts only
/// documents of the compiled schema: at the root, a value of the root's type —
/// an object, an array or a scalar; in every object, keys drawn from that
/// object's declared set (each at most once, all required ones present, in any
/// order) and each value matching its declared ``SchemaValueType`` — nested
/// objects and arrays included, every array within its item bounds. Keys and
/// enum values are matched scalar by scalar — a scalar outside ASCII raw or
/// as a JSON escape, the quote, the backslash and control characters as an
/// escape, every other ASCII scalar raw (``LiteralMatch``). It is the runtime
/// companion to
/// ``ResponseFormatDecoder`` and, like ``JSONGrammarState``, is a pure value
/// type — token classification is a non-mutating ``walk(_:)`` fold, MLX-free
/// and unit-testable.
///
/// ## Shape
/// A stack of open containers (``Frame``) plus one lexical ``Mode``. The schema
/// itself lives in a shared, immutable ``SchemaProgram`` built once in
/// ``init(root:)``. The scalar in progress lives in the mode, never in a
/// frame, so string and number bytes — most of any document — never touch the
/// stack, and a walk copies at most the open frames, once.
///
/// ## No dead ends
/// Every reachable state can still reach a complete document: `,` is legal
/// only where another member or item can follow, the compiler guarantees
/// `minItems <= maxItems` and that every required key is declared, every
/// literal has an all-ASCII spelling (printable ASCII raw, everything else
/// escaped, so a key the tokenizer cannot spell raw is still reachable), and
/// `\u` escapes are cut off as soon as they cannot complete.
public struct SchemaConstraintState: Hashable, Sendable {

    /// One open container.
    @usableFromInline
    enum Frame: Hashable, Sendable {
        /// An object of node `node`. A member is marked `emitted` when its key's
        /// closing quote is read.
        case object(node: Int32, emitted: PropertyMask)
        /// An array of node `node`; `count` items have been started — held at
        /// `minItems` once an unbounded array has that many, past which the
        /// count decides nothing (see ``startValue(_:node:)``).
        case array(node: Int32, count: Int)
    }

    /// The lexical position.
    @usableFromInline
    enum Mode: Hashable, Sendable {
        /// Whitespace, then the first byte of a value of `node`: at the root,
        /// after `:`, and after `,` in an array.
        case expectValue(node: SchemaProgram.NodeRef)
        /// Just after `[`: an item, or `]` when the array may be empty.
        case arrayOpen
        /// Just after `{` (`afterComma == false`) or after `,` in an object: a
        /// key not yet emitted, or `}` (never right after a comma).
        case objectOpen(afterComma: Bool)
        /// Inside a key: the match against the members not yet emitted.
        case key(LiteralMatch)
        /// A key was read; whitespace, `:`, then a value of `value`.
        case colon(value: SchemaProgram.NodeRef)
        /// Inside a scalar value.
        case scalar(SchemaScalarState)
        /// A value just completed in the innermost container: whitespace, `,`,
        /// or that container's close. With an empty stack the root value has
        /// ended — the accept state, where only whitespace may follow.
        case afterValue
    }

    @usableFromInline let program: SchemaProgram
    @usableFromInline var stack: ContiguousArray<Frame>
    @usableFromInline var mode: Mode

    /// A fresh automaton positioned before the root value.
    public init(root: SchemaValueType) {
        let program = SchemaProgram(root: root)
        self.program = program
        self.stack = []
        self.stack.reserveCapacity(4)
        self.mode = .expectValue(node: program.root)
    }

    /// A fresh automaton positioned before a root object.
    public init(schema: JSONSchemaObject) {
        self.init(root: .object(schema))
    }

    /// Whether the root value has been fully and validly produced — the accept
    /// state, and the only state in which EOS is permitted. A root number has
    /// no terminator byte, so a root in a terminal number state counts too
    /// (as in ``JSONGrammarState/isComplete``).
    @inlinable
    public var isComplete: Bool {
        guard stack.isEmpty else { return false }
        switch mode {
        case .afterValue: return true
        case .scalar(let scalar): return scalar.isCompleteNumber
        default: return false
        }
    }

    /// Whether the automaton is inside a string literal — a key (matched byte by
    /// byte against the declared names), a string value or an enum literal —
    /// where whitespace is data rather than formatting. The constraint
    /// processor consults this before withholding whitespace.
    public var isInsideString: Bool {
        switch mode {
        case .key: return true
        case .scalar(let scalar): return scalar.isInsideString
        default: return false
        }
    }

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
    /// (e.g. the constraint processor's "no legal token" log, often the only
    /// trace a cut-off generation leaves). Not a wire format: keys appear by
    /// name — a key's remaining candidates and each open object's emitted keys
    /// — and the rest of the mode is reflected.
    public var diagnosticDescription: String {
        let frames = stack.map { frame -> String in
            switch frame {
            case .object(let node, let emitted): return "object(emitted: \(names(emitted, node: node)))"
            case .array(let node, let count):
                // An unbounded array's count holds at `minItems` (see
                // ``startValue(_:node:)``); there it means "at least".
                let array = program.arrays[Int(node)]
                return array.maxItems == nil && count == array.minItems
                    ? "array(count: ≥\(count))" : "array(count: \(count))"
            }
        }
        var modeText = "\(mode)"
        if case .key(let match) = mode, case .object(let node, _)? = stack.last {
            modeText = "key(position: \(match.unit), candidates: \(names(match.candidates, node: node)), "
                + "progress: \(match.progress))"
        }
        return "schema(mode: \(modeText), frames: [\(frames.joined(separator: ", "))], complete: \(isComplete))"
    }

    /// The declared names of `members` in object node `node`: the first eight,
    /// then a count, so a very wide object cannot turn one log line into
    /// megabytes.
    private func names(_ members: PropertyMask, node: Int32) -> String {
        let keys = program.objects[Int(node)].keys
        let listed = keys.indices.filter(members.contains)
        var shown = listed.prefix(8).map { "\"\(keys[$0].text)\"" }
        if listed.count > 8 { shown.append("… (+\(listed.count - 8))") }
        return "[" + shown.joined(separator: ", ") + "]"
    }

    /// Two states are equal when they are at the same position of the same
    /// schema: the programs are compared by identity first, then table by
    /// table, scalar by scalar — not through the schema values they were
    /// compiled from, whose `String` equality is canonical and would equate a
    /// precomposed key with a decomposed one the automaton tells apart.
    public static func == (lhs: SchemaConstraintState, rhs: SchemaConstraintState) -> Bool {
        lhs.mode == rhs.mode && lhs.stack == rhs.stack
            && (lhs.program === rhs.program || lhs.program.isEquivalent(to: rhs.program))
    }

    /// Covers the position only, which is consistent with ``==``.
    public func hash(into hasher: inout Hasher) {
        hasher.combine(mode)
        hasher.combine(stack)
    }

    // MARK: - Transitions

    @usableFromInline
    mutating func applyInPlace(_ byte: UInt8) -> Bool {
        switch mode {
        case .expectValue(let node):
            if SchemaBytes.isWhitespace(byte) { return true }
            return startValue(byte, node: node)

        case .arrayOpen:
            if SchemaBytes.isWhitespace(byte) { return true }
            guard case .array(let node, _)? = stack.last else { return false }
            let array = program.arrays[Int(node)]
            if byte == SchemaBytes.rBracket {
                guard array.minItems == 0 else { return false }
                return closeContainer()
            }
            return startValue(byte, node: array.item)

        case .objectOpen(let afterComma):
            if SchemaBytes.isWhitespace(byte) { return true }
            guard case .object(let node, let emitted)? = stack.last else { return false }
            let object = program.objects[Int(node)]
            if byte == SchemaBytes.quote {
                let remaining = object.all.subtracting(emitted)
                guard !remaining.isEmpty else { return false }
                mode = .key(LiteralMatch(candidates: remaining))
                return true
            }
            if byte == SchemaBytes.rBrace {
                guard !afterComma, object.required.isSubset(of: emitted) else { return false }
                return closeContainer()
            }
            return false

        case .key(let match):
            guard case .object(let node, var emitted)? = stack.last else { return false }
            switch match.step(byte, literals: program.objects[Int(node)].keys) {
            case .continued(let next):
                mode = .key(next)
                return true
            case .completed(let member):
                // The closing quote matched a remaining name exactly.
                emitted.insert(member)
                stack[stack.count - 1] = .object(node: node, emitted: emitted)
                mode = .colon(value: program.objects[Int(node)].values[member])
                return true
            case .rejected:
                return false
            }

        case .colon(let value):
            if SchemaBytes.isWhitespace(byte) { return true }
            guard byte == SchemaBytes.colon else { return false }
            mode = .expectValue(node: value)
            return true

        case .scalar(let scalar):
            switch scalar.step(byte, program: program) {
            case .consumed(let next):
                mode = .scalar(next)
                return true
            case .completed:
                mode = .afterValue
                return true
            case .endedBefore:
                // A number ended before this byte; the byte belongs to the
                // container. Re-dispatch it once.
                mode = .afterValue
                return afterValue(byte)
            case .rejected:
                return false
            }

        case .afterValue:
            return afterValue(byte)
        }
    }

    /// Open a value of `node` from its first byte. An item of the innermost
    /// array is counted first, and refused once the array holds `maxItems`
    /// (which also covers `maxItems == 0` right after `[`).
    @usableFromInline
    mutating func startValue(_ byte: UInt8, node: SchemaProgram.NodeRef) -> Bool {
        if case .array(let arrayNode, let count)? = stack.last {
            let array = program.arrays[Int(arrayNode)]
            if let maxItems = array.maxItems, count >= maxItems { return false }
            // In an unbounded array the count past `minItems` decides nothing
            // (only `]` consults it), so it stops there: the states of such an
            // array stay finite, and a search over them can finish.
            let counted = array.maxItems == nil ? Swift.min(count + 1, array.minItems) : count + 1
            stack[stack.count - 1] = .array(node: arrayNode, count: counted)
        }
        switch node {
        case .object(let objectNode):
            guard byte == SchemaBytes.lBrace else { return false }
            stack.append(.object(node: objectNode, emitted: .empty))
            mode = .objectOpen(afterComma: false)
            return true
        case .array(let arrayNode):
            guard byte == SchemaBytes.lBracket else { return false }
            stack.append(.array(node: arrayNode, count: 0))
            mode = .arrayOpen
            return true
        case .scalar(let scalarNode):
            let kind = program.scalars[Int(scalarNode)]
            guard let scalar = SchemaScalarState.start(byte, node: scalarNode, kind: kind) else { return false }
            mode = .scalar(scalar)
            return true
        }
    }

    /// After a completed value: whitespace, `,`, or the innermost container's
    /// close; nothing once the root has closed.
    @usableFromInline
    mutating func afterValue(_ byte: UInt8) -> Bool {
        if SchemaBytes.isWhitespace(byte) { return true }
        guard let top = stack.last else { return false }
        switch top {
        case .object(let node, let emitted):
            let object = program.objects[Int(node)]
            if byte == SchemaBytes.comma {
                // A comma promises another member. With every declared key
                // emitted there is none left to promise, and after the comma
                // only whitespace would be legal: the model could never close
                // the object and would run to max_tokens emitting blanks (seen
                // on a real checkpoint).
                guard !object.all.subtracting(emitted).isEmpty else { return false }
                mode = .objectOpen(afterComma: true)
                return true
            }
            if byte == SchemaBytes.rBrace {
                guard object.required.isSubset(of: emitted) else { return false }
                return closeContainer()
            }
            return false
        case .array(let node, let count):
            let array = program.arrays[Int(node)]
            if byte == SchemaBytes.comma {
                // The same promise for items: refused once the array is full,
                // which is how `maxItems` is enforced byte by byte.
                if let maxItems = array.maxItems, count >= maxItems { return false }
                mode = .expectValue(node: array.item)
                return true
            }
            if byte == SchemaBytes.rBracket {
                guard count >= array.minItems else { return false }
                return closeContainer()
            }
            return false
        }
    }

    /// Pop the innermost container; the value it held is complete. Always
    /// succeeds; typed to return `Bool` so it composes in the transitions.
    @usableFromInline
    mutating func closeContainer() -> Bool {
        stack.removeLast()
        mode = .afterValue
        return true
    }
}
