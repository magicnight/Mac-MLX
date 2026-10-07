// Copyright © 2026 macMLX. English comments only.

/// A compiled root ``SchemaValueType`` in the flat, index-addressed form the
/// schema automaton runs on: every object, array and scalar node of the schema
/// tree in its own table, keys and enum values pre-split into scalars.
///
/// Built once per ``SchemaConstraintState/init(root:)`` and shared by every
/// state derived from it, so walking a token never re-encodes a key or an enum
/// value, and a state's frames can name their node with a small integer.
/// Immutable after construction.
@usableFromInline
final class SchemaProgram: Sendable {

    /// A node of the compiled schema: an index into ``objects``, ``arrays`` or
    /// ``scalars``.
    @usableFromInline
    enum NodeRef: Hashable, Sendable {
        case object(Int32)
        case array(Int32)
        case scalar(Int32)
    }

    @usableFromInline
    struct ObjectNode: Equatable, Sendable {
        /// Declared property names, in declaration order. A name declared
        /// twice keeps its first occurrence.
        @usableFromInline let keys: [SchemaLiteral]
        /// The value node of each key, parallel to ``keys``.
        @usableFromInline let values: [NodeRef]
        /// Every member index.
        @usableFromInline let all: PropertyMask
        /// The required members. A `required` name that is not declared maps to
        /// a phantom index (`keys.count`) that is never emitted, so such an
        /// object can never close — the behaviour a hand-built schema had
        /// before nesting existed. The compiler never produces one.
        @usableFromInline let required: PropertyMask
    }

    @usableFromInline
    struct ArrayNode: Equatable, Sendable {
        @usableFromInline let item: NodeRef
        @usableFromInline let minItems: Int
        /// `nil` means unbounded.
        @usableFromInline let maxItems: Int?
    }

    @usableFromInline
    enum ScalarKind: Equatable, Sendable {
        case string
        case number
        case integer
        case boolean
        /// The enum values.
        case stringEnum([SchemaLiteral])
        /// A number or integer within a range.
        case boundedNumber(NumberRange)
    }

    @usableFromInline let objects: [ObjectNode]
    @usableFromInline let arrays: [ArrayNode]
    @usableFromInline let scalars: [ScalarKind]
    /// The root value — an object, an array or a scalar.
    @usableFromInline let root: NodeRef

    /// Whether `other` runs the same automaton: the same tables, literal for
    /// literal and scalar for scalar. Two programs compiled from the same
    /// schema value are equivalent, and only those — a key or value spelled
    /// with a different normalisation has different scalars.
    func isEquivalent(to other: SchemaProgram) -> Bool {
        root == other.root && objects == other.objects && arrays == other.arrays && scalars == other.scalars
    }

    init(root schema: SchemaValueType) {
        var builder = Builder()
        let root = builder.add(schema)
        self.objects = builder.objects
        self.arrays = builder.arrays
        self.scalars = builder.scalars
        self.root = root
    }

    /// Flattens the schema tree into the node tables, children first.
    private struct Builder {
        var objects: [ObjectNode] = []
        var arrays: [ArrayNode] = []
        var scalars: [ScalarKind] = []

        mutating func add(_ type: SchemaValueType) -> NodeRef {
            switch type {
            case .string:
                return addScalar(.string)
            case .number:
                return addScalar(.number)
            case .integer:
                return addScalar(.integer)
            case .boundedInteger(let bounds):
                return addScalar(.boundedNumber(NumberRange(bounds)))
            case .boundedNumber(let bounds):
                return addScalar(.boundedNumber(NumberRange(bounds)))
            case .boolean:
                return addScalar(.boolean)
            case .stringEnum(let values):
                return addScalar(.stringEnum(values.map(SchemaLiteral.init)))
            case .object(let object):
                return addObject(object)
            case .array(let items, let minItems, let maxItems):
                let item = add(items)
                arrays.append(ArrayNode(item: item, minItems: minItems, maxItems: maxItems))
                return .array(Int32(arrays.count - 1))
            }
        }

        mutating func addObject(_ object: JSONSchemaObject) -> NodeRef {
            // Keyed by scalars, as the automaton matches keys; `String` equality
            // is canonical, so a decomposed "é" would otherwise find a
            // precomposed one the matcher never equates it with.
            var index: [[UInt32]: Int] = [:]
            var keys: [SchemaLiteral] = []
            var values: [NodeRef] = []
            for property in object.properties {
                let literal = SchemaLiteral(property.name)
                guard index[literal.scalars] == nil else { continue }
                index[literal.scalars] = keys.count
                keys.append(literal)
                values.append(add(property.type))
            }
            var required = PropertyMask()
            for name in object.required {
                required.insert(index[name.unicodeScalars.map(\.value)] ?? keys.count)
            }
            objects.append(ObjectNode(keys: keys, values: values, all: .all(count: keys.count), required: required))
            return .object(Int32(objects.count - 1))
        }

        mutating func addScalar(_ kind: ScalarKind) -> NodeRef {
            scalars.append(kind)
            return .scalar(Int32(scalars.count - 1))
        }
    }
}
