# Structured output: nested objects, arrays and `$ref` — design

**Date:** 2026-10-03
**Status:** design, validated by a prototype; implementation follows this document
**Why:** Apple's Foundation Models `@Generable` emits nested object schemas (as `$defs` + `$ref`), arrays with `minItems`/`maxItems`, `const`, and `x-order`. Today every one of those is a 400 from `ResponseFormatDecoder`, so macMLX cannot serve guided generation for anything beyond a flat object. Verified end to end: Apple's client's flat guided generation passes against a live Qwen3.6 checkpoint (after PR #125); the nested one gets `httpError(statusCode: 400)`.

## What exists today (file:line as of PR #125)

| Concern | Where | Today |
|---|---|---|
| Property allow-list | `Constraint/ResponseFormatDecoder.swift` `allowedPropertyKeys` | Only `type`, `enum`, `description`, `title`, `default`. Everything else is a 400. |
| Nested types | `compilePropertyType` | `type: object/array` → `unsupportedFeature("nested … on property …")` |
| Root object | `compileObjectSchema` | Reads only `type`, `additionalProperties`, `properties`, `required`. **All other root keywords are silently ignored** (bug C3 below). |
| M2 literal rule | `requireLiteralMatchable` | Keys and enum values must be ASCII with no `"`, `\` or control characters (the runtime matcher compares literal bytes). |
| Model | `SchemaValueType.swift`, `JSONSchemaObject.swift` | Flat scalars only |
| Automaton | `SchemaConstraintState.swift` | One `Phase` and one `emitted: Set<String>`; per-byte `remainingKeys` recomputes `map`+`filter`, key bytes re-encoded per byte, enum list re-encoded per walk |
| C1 reference | `JSONGrammarState.swift` | `maxDepth` counts open containers; `afterValue` decided per container type |
| Consumers | `ConstraintState.swift`, `JSONConstraintProcessor.swift`, `MLXSwiftEngine.swift`, `HummingbirdServer.swift` | Use only `init(schema:)`, `walk`, `isComplete`, `diagnosticDescription`. No changes needed outside `Constraint/`. |

## What Apple actually sends (measured on this machine, macOS 27 SDK)

- Nested `@Generable` types become root `$defs` with bare `{"$ref":"#/$defs/Address"}` refs, used both as a property schema and as `items`. `$defs` entries carry `title`, `x-order`, `additionalProperties: false`, and `required` (which may be `[]`).
- Optionals (`Int?`, `Address?`, `[Int]?`) are the plain schema omitted from `required`. No `anyOf`, no `null`, no type array — even with `@Generable(representNilExplicitlyInGeneratedContent: true)`.
- Array guides: `@Guide(.count(2))` → `minItems: 2, maxItems: 2`; `.minimumCount`/`.maximumCount` → `minItems`/`maxItems`; `[[Int]]` → array of arrays.
- String enums: `@Guide(.anyOf([…]))` and payload-free `@Generable enum` → `{"type":"string","enum":[…]}`.
- `@Guide(.constant("fixed"))` → `{"const":"fixed"}` with no `type`.
- Guides that stay a 400 after this work: `.range`/`.minimum`/`.maximum` → `minimum`/`maximum`; Regex guides → `pattern`.
- `generating: [T].self`, `Int.self` and `Enum.self` produce non-object roots (follow-up, see decisions).
- Apple's client sends `{"type":"json_schema","json_schema":{"name","schema","strict":true}}`; the envelope parser already ignores the extra keys.
- Apple's TripPlanner sample has the enum value `"Lençóis Maranhenses"`; it still 400s after this work because of M2 (non-ASCII). Follow-up.
- Depth seen in upstream mlx-swift-lm goldens: up to 5 nested containers; property names such as `type`, `title`, `description`, `items` collide with schema keywords and must be accepted as property names.

## Pre-existing bugs in the shipped flat path

- **C1 — whitespace trap.** `afterValue` accepted `,` with every declared key emitted; after that only whitespace was legal, so forced EOS never fired and the model ran to `max_tokens`. **Fixed in PR #125** (`guard !remainingKeys.isEmpty`).
- **C2 — dead surrogate-escape prefixes.** `stringUnicodeValue` checks surrogates only at the 4th hex digit, so `\uDC`–`\uDF` outside a pair and `\uD83D\u00` are accepted and then no byte is legal → forced EOS with truncated output. Fix: on the 1st hex digit, if `expectingLow` the digit must be `D`; on the 2nd digit, `expectingLow` must equal `(0xDC...0xDF).contains(value)`. (`JSONGrammarState` has the identical dead end; out of scope here — follow-up.)
- **C3 — root silently drops constraints.** The decoder accepts a root with `allOf`, `anyOf`, `minProperties`, `patternProperties`, … and enforces none of them, violating the project's own rule that an unenforceable constraint must 400, never be silently downgraded.

## Prototype validation (scratch, not in the repo)

A full prototype of the design below ran: 41,447 ported assertions (every existing `SchemaConstraintStateTests` and `ResponseFormatDecoderTests` expectation, verbatim) with 0 failures; 48,000 fuzzed documents agreeing exactly with an independent strict RFC 8259 parser + recursive validator; every accepted document also accepted by `JSONGrammarState`; identical complete-document acceptance to the shipped flat automaton on 24,000 flat documents (the only prefix differences are the C1/C2 dead prefixes); 0 trap states in a BFS + backward co-reachability search over 3.68M states across 300 random schemas; upstream golden tiers 1–4 and the TripPlanner (ASCII) document accepted. Performance (150K-token synthetic vocabulary, `-O`, best of 5): never slower than today, 6–15× faster at key positions (70-property object key position: 1,050 ms → 72 ms).

## Design

### R1. Data model

```swift
// SchemaValueType.swift
public enum SchemaValueType: Equatable, Hashable, Sendable, Codable {
    case string, number, integer, boolean
    case stringEnum([String])
    /// Nested object: inline `properties` or a resolved `$ref`.
    case object(JSONSchemaObject)
    /// Every element is `items`. `maxItems == nil` means unbounded.
    /// The compiler guarantees 0 <= minItems <= (maxItems ?? .max).
    indirect case array(items: SchemaValueType, minItems: Int, maxItems: Int?)
}
```

`JSONSchemaObject` unchanged apart from its doc comment ("an object schema at any depth"). `ResponseFormat` unchanged; the root stays an object. No persisted encodings of `GenerateRequest` exist, so the larger synthesized `Codable` shape is safe.

Internal automaton types (new files, one type per file; helper types nested):

```swift
// PropertyMask.swift: no allocation for <= 64 members.
@usableFromInline struct PropertyMask: Hashable, Sendable {
    var low: UInt64        // members 0..<64
    var high: [UInt64]     // members >= 64; empty otherwise; kept trimmed so == is semantic
    // empty, all(count:), isEmpty, contains, insert, subtracting, isSubset(of:), filtered(_:), first(where:)
}

// SchemaProgram.swift: immutable, built once in SchemaConstraintState.init, shared by every derived state.
@usableFromInline final class SchemaProgram: Sendable {
    enum NodeRef: Hashable, Sendable { case object(Int32), array(Int32), scalar(Int32) }
    struct ObjectNode: Sendable {
        let keys: [[UInt8]]       // UTF-8 names, precomputed once
        let values: [NodeRef]
        let all: PropertyMask
        let required: PropertyMask
    }
    struct ArrayNode: Sendable { let item: NodeRef; let minItems: Int; let maxItems: Int? }
    enum ScalarKind: Sendable { case string, number, integer, boolean, stringEnum([[UInt8]]) }
    let objects: [ObjectNode]; let arrays: [ArrayNode]; let scalars: [ScalarKind]
    let root: NodeRef; let source: JSONSchemaObject
    init(root: JSONSchemaObject)
}
```

Builder rules (keep today's semantics for hand-built schemas): duplicate property names — first occurrence wins; an undeclared `required` name maps to a phantom bit (index = `keys.count`) that is never set, so `}` is never legal (matches today).

```swift
// SchemaScalarState.swift: today's ValueState, made position-agnostic.
@usableFromInline enum SchemaScalarState: Hashable, Sendable {
    case stringBody, stringEscape, stringUnicode(digitsSeen: Int, value: Int, expectingLow: Bool)
    case stringHighSurrogateBackslash, stringHighSurrogateU
    case enumBody(node: Int32, position: Int, candidates: PropertyMask)
    case numberAfterMinus, numberAfterLeadingZero, numberIntDigits, numberAfterDot,
         numberFracDigits, numberAfterExp, numberAfterExpSign, numberExpDigits
    case intAfterMinus, intAfterZero, intDigits
    case literal(isTrue: Bool, matched: Int)    // replaces literal(remaining: [UInt8])
    enum Step { case consumed(SchemaScalarState), completed, endedBefore, rejected }
    static func start(_ byte: UInt8, node: Int32, kind: SchemaProgram.ScalarKind) -> SchemaScalarState?
    func step(_ byte: UInt8, program: SchemaProgram) -> Step
}
```

`.completed`: the byte belongs to the value and ends it (closing `"`, last literal byte). `.endedBefore`: a number ended before this byte; the container re-dispatches it. The transition table is today's with the C2 fix. Number and literal states never carry a key, so the same machine serves object members and array items.

```swift
// SchemaConstraintState.swift
public struct SchemaConstraintState: Hashable, Sendable {
    @usableFromInline enum Frame: Hashable, Sendable {
        case object(node: Int32, emitted: PropertyMask)   // a key is marked emitted when its closing quote is read
        case array(node: Int32, count: Int)               // items started
    }
    @usableFromInline enum Mode: Hashable, Sendable {
        case expectValue(node: SchemaProgram.NodeRef)     // root start, after ':', after ',' in an array
        case arrayOpen                                    // just after '['
        case objectOpen(afterComma: Bool)                 // just after '{' or ','
        case key(position: Int, candidates: PropertyMask)
        case colon(value: SchemaProgram.NodeRef)
        case scalar(SchemaScalarState)                    // the scalar in progress lives here, never in a frame
        case afterValue                                   // with an empty stack this is the accept state
    }
    @usableFromInline let program: SchemaProgram
    @usableFromInline var stack: ContiguousArray<Frame>
    @usableFromInline var mode: Mode
    // Public API unchanged: init(schema:), isComplete, advanced(over:), walk(_:), diagnosticDescription.
    // ==   compares mode and stack, then (program === other || program.source == other.source).
    // hash covers mode and stack only, which is consistent with ==.
}
```

Marking a key emitted at its closing quote rather than at value completion gives the same language (nothing reads that frame's `emitted` between those points). `diagnosticDescription` becomes `schema(mode:…, depth:…, complete:…)`; optionally keep a `current` key index in object frames to print a breadcrumb like `$.days[2].activities[1]`.

### R2. Compiler (`ResponseFormatDecoder`)

Constants: `maxSchemaDepth = 32` (open containers, root counts as 1; same counting as `JSONGrammarState.maxDepth`, and 32 ≤ 64 keeps C2 ⊆ C1); `maxSchemaNodes = 4096` (every `compileValue` call counts, including each `$ref` hop).

Dispatch precedence: `$ref`, then `const`, then `type`. Every kind also accepts the annotations `description`, `title`, `default`, `examples`, `$comment`.

| Kind | Detected by | Allowed keywords (besides annotations) |
|---|---|---|
| Root | top level | `type` (`"object"` or absent; today's message otherwise), `properties`, `required`, `additionalProperties` (`false` only), `x-order`, and root-only `$defs`, `definitions`, `$schema`, `$id` |
| Ref | has `$ref` | annotations only |
| Const | has `const` | `type` (must be `"string"` if present). String value only → `.stringEnum([v])`, M2 applied. |
| Object | `type: "object"` | `type`, `properties` (required, non-empty), `required`, `additionalProperties` (`false`), `x-order` |
| Array | `type: "array"` | `type`, `items` (required, a schema object), `minItems`, `maxItems` |
| Scalar | any other or absent `type` | Today's `compilePropertyType`, verbatim: same check order, same messages, with sorted key iteration (deterministic errors) |

Algorithm:
1. `compileRootSchema`: root type check (unchanged); build `Context { defs, definitions, expanding: [String], nodes }` — `$defs` and `definitions` are separate tables and must be objects (else `invalidFormat`); call `compileObject(isRoot: true, depth: 1)`.
2. `compileObject`, in order: allow-list check (`at the schema root` / `on property 'path'`); depth check (`depth <= maxSchemaDepth`); `additionalProperties` check; `properties` — missing at the root → `invalidFormat("schema.properties object is required")` (today's message), missing when nested → `unsupportedFeature("nested object without 'properties' on property 'p' (free-form objects are not supported)")`, must be non-empty; for each name, sorted: M2 check, then `compileValue(child, path: p.name, depth)`; `required` as today (each entry declared).
3. `compileValue`: `ctx.spend()`. **`$ref` path:** any non-annotation sibling → `unsupportedFeature("schema keyword 'K' alongside '$ref' on property 'p' (only annotations may accompany '$ref')")`; the value must be a string; resolve (rules below); if already in `expanding` → `unsupportedFeature("recursive schema: '$ref' 'R' on property 'p' refers back to itself (recursive schemas cannot be bounded)")` — the set is scoped to the current path, so the same def used twice in sibling positions is fine; if `expanding.count >= maxSchemaDepth` → 400 (bounds compiler recursion); push the ref and recurse with the same path and depth. `const` and `type` dispatch as in the table. `object` and `array` children use `depth + 1`.
4. `compileArray`: allow-list and depth checks; `items` — missing → `unsupportedFeature("nested array without 'items' on property 'p' (free-form arrays are not supported)")`, an array → `"tuple-form 'items'"`, a bool → `"boolean 'items' schema"`, other non-object → `invalidFormat`; `minItems`/`maxItems` must be `.int(n >= 0)` else `invalidFormat("… must be a non-negative integer")` (note `JSONValue` already decodes `3.0` and `1e2` as `.int`); `minItems > maxItems` → `invalidFormat("minItems (m) exceeds maxItems (M) …")` (required, or `[` could never close). Item path: `p[]`.
5. **`$ref` resolution:** only `#/$defs/<seg>` and `#/definitions/<seg>` — exactly one non-empty segment, RFC 6901 unescaping (`~0`, `~1`), no `%`. Anything else (remote refs, `#`, `#/properties/…`) → `unsupportedFeature("'$ref' 'R' on property 'p' (only '#/$defs/<name>' and '#/definitions/<name>' are supported)")`. Missing target → `invalidFormat("… does not resolve to a definition")`. Resolution is lazy: an unreferenced def is never compiled. Nested `$defs`, `$id`, `$schema` hit the allow-list → 400 (a nested `$id` would rebase refs).
6. Type arrays (`type: ["string","null"]`) → `unsupportedFeature("type arrays (e.g. nullable unions) …")`.
7. `x-order` is accepted and ignored. Property order stays sorted by name.

Paths: `name`, `parent.child`, `list[]`, `list[].field`. A root-level path is the bare name, so every existing message stays byte-identical. Existing tests constrain messages: "nested object", "nested array", and `'$ref'` must still appear where they do today.

Flat-case decoder deltas (behaviour changes to list in the CHANGELOG): root `allOf`/`anyOf`/`minProperties`/`patternProperties`/… go from ACCEPT to a 400 (C3 fix); `examples` and `$comment` on a property go from 400 to accepted; `const` goes from 400 to accepted and enforced; a type array goes from `invalidFormat` to `unsupportedFeature`; the free-form nested object/array messages are elaborated but keep the old substrings.

### R3. Automaton transitions

| Mode | Whitespace | Other bytes |
|---|---|---|
| `expectValue(n)` | stay | `startValue(byte, n)` |
| `arrayOpen` | stay | `]` only if `minItems == 0` → pop. Otherwise `startValue(byte, item)`. |
| `objectOpen(afterComma)` | stay | `"` only if `all − emitted ≠ ∅` → `key(0, all − emitted)`. `}` only if `!afterComma` and `required ⊆ emitted` → pop. |
| `key(p, C)` | (just a byte) | `"`: needs `i ∈ C` with `keys[i].count == p` → `emitted ∪= {i}`, mode `colon(values[i])`. Other bytes: `C' = {i ∈ C : keys[i][p] == byte}`; needs `C' ≠ ∅` → `key(p+1, C')`. |
| `colon(v)` | stay | `:` → `expectValue(v)` |
| `scalar(s)` | per scalar | `consumed` → `scalar(s')`. `completed` → `afterValue`. `endedBefore` → `afterValue`, then re-dispatch the byte to `afterValue` (once only). `rejected` → nil. |
| `afterValue`, top is an object | stay | `,` only if `all − emitted ≠ ∅` (C1) → `objectOpen(true)`. `}` only if `required ⊆ emitted` → pop. |
| `afterValue`, top is an array | stay | `,` only if `count < maxItems` (this is how `maxItems` is enforced byte by byte) → `expectValue(item)`. `]` only if `count >= minItems` → pop. |
| `afterValue`, empty stack | stay | reject (root complete) |

`startValue(byte, n)`: if the top frame is an array, first require `count < maxItems` (covers `maxItems == 0` right after `[`), then `count += 1`; then `{` pushes `object(emitted: ∅)` and sets `objectOpen(false)`, `[` pushes `array(count: 0)` and sets `arrayOpen`, a scalar uses `SchemaScalarState.start`, any other byte rejects. Pop does `stack.removeLast()` and sets `mode = .afterValue`. `isComplete = stack.isEmpty && mode == .afterValue`. Empty containers: `[]` legal iff `minItems == 0`; `{}` legal iff no required keys. No runtime depth check needed (bounded by the compiled schema).

No dead ends: every reachable state can reach acceptance given the `,` guards, `min ≤ max` at compile time, M2, `required ⊆ declared`, and the progressive surrogate check. The BFS property test asserts this.

Keeping `walk` cheap: `var state = self` is one retain on `program`, one on the `stack` buffer, a copy of `mode`; the scalar in progress lives in `mode`, so string/number bytes never touch `stack`; frames are small and reference-free (Int32 ids + inline `UInt64` mask) in a `ContiguousArray`, so the first frame mutation in a walk copies at most `depth` frames; key and enum bytes are precomputed in `SchemaProgram` and candidates are bitmasks narrowed per byte; no `String` threaded through value bytes.

### R4. Test plan

Existing tests that must stay green unchanged: `ResponseFormatDecoderTests` (19), `SchemaConstraintStateTests` (now 15), `JSONConstraintProcessorDecisionTests` (6) + gated `MaskTests` (3), `JSONGrammarStateTests`, `TokenVocabularyTableTests`, `StructuredOutputServerTests` (3), the `HummingbirdServerTests` `response_format` combination tests, `StructuredOutputThinkingTests` (5), `StructuredOutputModelTests` (gated).

New fixtures under `MacMLXCore/Tests/MacMLXCoreTests/Fixtures/` (already copied by the test target): `fm_generable_schemas_fixture.json` (keys `Person`, `PersonNoRange`, `Explicit`, from the real-framework probe) and `fm_itinerary_production_fixture.json` (Apple's TripPlanner schema as emitted by `@Generable Itinerary`, copied from mlx-swift-lm `Tests/MLXFoundationModelsTests/TestHelpers.swift`). Compiled objects list properties sorted by name.

Decoder matrix (each in the json_schema envelope; "mutation caught" names the defect each case exists to catch):

| # | Input | Expect | Mutation caught |
|---|---|---|---|
| D1 | `{"type":"object","properties":{"home":{"type":"object","properties":{"street":{"type":"string"},"zip":{"type":"integer"}},"required":["street"],"additionalProperties":false}},"required":["home"]}` | `home == .object(JSONSchemaObject(properties: [street: .string, zip: .integer], required: ["street"]))` | object dispatch removed |
| D2 | `"tags":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3}`; and one with no bounds | `.array(.string, 1, 3)`; `.array(.string, 0, nil)` | min/max swapped; default `maxItems = 0` |
| D3 | `"rows":{"type":"array","items":{"type":"array","items":{"type":"integer"}}}`, `"people":{"type":"array","items":{"type":"object","properties":{"n":{"type":"string"}}}}` | array of arrays; array of objects | `items` not recursed |
| D4 | `$defs:{"Addr":{object street}}`, `definitions:{"Tag":{"type":"string","enum":["a","b"]}}`; `"home":{"$ref":"#/$defs/Addr","description":"d"}`, `"tags":{"type":"array","items":{"$ref":"#/definitions/Tag"}}` | resolved; annotation sibling allowed | `definitions` looked up in `$defs` |
| D5 | the same def referenced twice (`x` and `y.items`) | compiles | global visited set instead of path-scoped |
| D6 | root `title`, `$schema`, `$id`, `x-order`, `$comment`; property `examples`, `default` | equal to the bare compile | `x-order` missing from the object allow-list |
| D7 | `{"const":"fixed"}`; `{"type":"string","const":"fixed"}` | `.stringEnum(["fixed"])` | `const` branch removed |
| D8 | real `PersonNoRange` and `Explicit` fixtures | `tags == .array(.string, 2, 2)`, `home == .object(Address)`, `matrix == .array(.array(.integer, 0, nil), 0, nil)`, `konst == .stringEnum(["fixed"])`, `mood == .stringEnum(["happy","sad"])` | any shape regression |
| D9 | properties named `type`, `items`, `properties`, `$ref`, `description` | 5 string properties | allow-list applied to property names |
| D10 | a chain of 32 vs 33 nested objects | accept / `"nesting deeper than 32"` | off-by-one |
| R1 | `N: {properties: {next: {$ref: N}}}` | `"recursive schema"` | cycle check removed |
| R2 | `A → B → A` | `"recursive schema"` | only self-cycles detected |
| R3 | `#/$defs/Missing` | `invalidFormat` `"does not resolve"` | missing ref treated as free-form |
| R4 | `https://x/y.json`, `#/properties/m`, `#`, `#/$defs/a/b` | contains `'$ref'` | prefix-only matching |
| R5 | `{"$ref":"#/$defs/A","type":"object"}` | `"alongside '$ref'"` | siblings ignored |
| R6 | `minItems:3,maxItems:2`; `-1`; `2.5`; `"3"` | `"exceeds maxItems"` / `"non-negative integer"` | `min > max` accepted (creates a trap) |
| R7 | `items:[…]`, `items:true`, `uniqueItems`, `contains`, `prefixItems` | `"tuple-form"`, `"boolean 'items'"`, `'<key>'` | array allow-list loosened |
| R8 | nested `additionalProperties:true`; nested `$defs` or `$id` | 400 | root-only keys allowed when nested |
| R9 | root `minProperties`, `allOf`, `anyOf`, `patternProperties`, `dependentRequired` | contains `'<key>'` and "at the schema root" (behaviour change, C3) | root allow-list skipped |
| R10 | `type:["string","null"]`; `anyOf:[{string},{null}]`; `type:"null"` | `"type arrays"`, `'anyOf'`, `"property type 'null'"` | null silently allowed |
| R11 | nested key `café`; nested enum value with `\`; `const` containing `"` | `"non-ASCII"` / `"requires JSON escaping"` | M2 applied only at depth 1 |
| R12 | 20 levels × fan-out 4 of `$ref` | `"schema too large"`, returned quickly | budget removed (hangs) |
| R13 | `const: 3`; `type: integer` with `const` | `"non-string 'const'"` / `"'const' on non-string"` | non-string const accepted |
| R14 | real `Person` fixture (has `.range`) | contains `'maximum'` | numeric bounds silently dropped |
| R15 | real TripPlanner production fixture | `"non-ASCII"` (documents the M2 gap) | — |

Automaton walks (`SchemaConstraintStateTests`):

| # | Case | Mutation caught |
|---|---|---|
| N1 | Schema `{name: string, home: {street (required), zip: integer}}`, `home` required. Accept `{"home":{"street":"Main"}}` and `{"name":"x","home":{"zip":1,"street":"M"}}`. Reject `{"home":{}}`, `{"home":{"street":"M","street":"N"}}`, `{"home":{"street":"M","name":"N"}}`, `{"home":"x"}`, and the parent `home` key twice. | one `emitted` set shared across frames |
| N2 | `{"home":{}}` and `{"home":{ }}` when nothing nested is required | empty-object close rejected |
| N3 | `tags` = array(string, 2…3). Reject `[]` and `["a"]`. Accept 2 or 3 items. Walk `{"tags":["a","b","c",` → nil. Walk `{"tags":["a"]` → nil. Reject trailing and leading commas. | `,` or `]` bound check removed |
| N4 | Unbounded integers: accept `[]`, `[1,-2,30]`, `[0 ,0]`; reject `[1.5]`, `[01]`, `[1}` | re-dispatch to the wrong container |
| N5 | `maxItems: 0`: accept `{"z":[]}`; walk `{"z":[t` → nil | start-time bound check removed |
| N6 | Enum items: `["x","xy","x"]` accepted, `["y"]` rejected | — |
| N7 | `[1e5,-0.5]` and `[3]` accepted; `[1,2,3]` rejected when max is 2 | number terminators not re-dispatched |
| N8 | Array of objects (`id` required, max 2): required enforced per item; third item rejected; `,` rejected when full; each item gets a fresh `emitted` set; repeated keys across items allowed | `count` not incremented for container items |
| N9 | Matrix: accept `[[1],[2,3],[4]]` and `[]`; reject `[[]]` (inner min 1), `[[1,2,3]]`, `[1]` | inner/outer frames mixed up |
| N10 | One token spanning several frames: from `{"x":[{"y":["a`, walking `"]},{"y":[]}]}` → complete; walking `","b` → nil (inner max 1) | — |
| N11 | Nested keys `a` / `ab` sharing a prefix, `ab` required; `{"o":{"a":1,"a"` → nil | candidates not narrowed or not checked against `emitted` |
| N12 | Single-key schema `{"a":"x",` → nil; nested `{"o":{"k":true,` → nil | C1 guard removed |
| N13 | `{"msg":"\uDD` → nil; `{"msg":"\uD83D\u00` → nil; `{"msg":"\uD83D\uD` → non-nil | C2 pruning removed, or over-pruned |
| N14 | BFS trap property: fixed and seeded schemas, alphabet ``{}[],:" Ɖdc.-eEtrufalsn`` plus key/enum bytes, about 40k-state cap, backward co-reachability; assert 0 traps | either guard removed |
| N15 | Differential: seeded schemas × valid and mutated documents. Automaton equals the oracle (test helper `ReferenceSchemaValidator.swift`, ~170 lines). Every accepted document is also accepted by `JSONGrammarState`. About 500×12 documents in ~1 s. | any transition bug |
| N16 | 70 properties (`p0`…`p69`, required `p65` and `p3`), a duplicate high-index key, all 70 present then `,` → nil; enum of 100 values | high-word mask bugs |
| N17 | Golden tier 1–4 documents; TripPlanner (ASCII) document; enum mutation rejected | — |
| N18 | Equal schemas at the same position → `==` and same hash; different positions → `!=` | `==` by program identity only |

Processor decisions with a stub vocabulary (`JSONConstraintProcessorDecisionTests`, `selectLegalToken` / `isLegal`):

| # | Setup | Expect | Mutation caught |
|---|---|---|---|
| P1 | `tags` array(string, 0…1), after `{"tags":["a"`; vocabulary `[",\"", "\"]", "\"]}"]`, order `[0,1,2]` | id 1 | `,` max guard removed |
| P2 | array(string, 2…∞), same prefix; vocabulary `["\"]", "\",\""]`, order `[0,1]` | id 1 | `]` min guard removed |
| P3 | Single-key `{a}`, after `{"a":"x"`; vocabulary `["\",", "\"}"]`, order `[0,1]` | id 1 | C1 guard removed |
| P4 | `{o: {k: boolean}}`; vocabulary `["}", "</s>"]`, stop `{1}`, order `[1,0]` | At `{"o":{"k":true}`: id 0. At `{"o":{"k":true}}`: id 1. | `isComplete` ignores the stack |
| P5 | Full array, vocabulary `[",\"", "x"]` | nil (forced EOS path) | — |
| P6 | Items: array(object{id, required}, 1…1), after `{"items":[{"id":1`; vocabulary `["},{", "}]}"]`, order `[0,1]` | id 1 | item count not incremented on `{` |

Server (`StructuredOutputServerTests`): S1 — a nested schema (the real `Explicit` fixture) → 200 with the stub engine; S2 — a recursive `$ref` → 400 whose message contains "unsupported schema feature" and "recursive schema"; the existing nested-400 test stays.

Gated E2E (`StructuredOutputModelTests.testC2NestedConformsToSchema`): the TripPlanner ASCII schema through the decoder; output validated with `ReferenceSchemaValidator`. Run on this machine against Qwen3.6-27B-4bit (`MACMLX_STRUCTURED_MODEL=Qwen3.6-27B-4bit`).

### R5. Implementation order (each step compiles and keeps every test green)

0. **C2 fix in the shipped automaton** (progressive surrogate checks) + `Hashable` (synthesizable) + N13 + N14 (BFS). C1 is already in.
1. **Pure refactor:** extract `SchemaScalarState` + `Step` (and a small `SchemaBytes` constants file so constants are not copied a third time). The flat automaton uses it with today's semantics. Tests unchanged.
2. **Model cases, `PropertyMask`, `SchemaProgram`, and the frame/mode rewrite of `SchemaConstraintState`.** The decoder still rejects nested shapes; automaton tests build nested schemas directly. Add N1–N11, N15–N18 and the oracle helper.
3. **Recursive decoder:** per-kind allow-lists, `$ref`, `const`, root allow-list (C3), both caps; decoder matrix, fixtures, S1 and S2.
4. **Processor tests, docs and E2E:** P1–P6, the gated E2E; doc comments in `ResponseFormatDecoder` (supported-subset header), `SchemaValueType`, `JSONSchemaObject`, `ResponseFormatError`, `ConstraintState`; CHANGELOG (list the C3 behaviour change explicitly); README subset line.
5. **Follow-ups (separate PRs):** non-object roots (`ResponseFormat.jsonSchemaValue(SchemaValueType)`); M2 non-ASCII (Apple's TripPlanner sample needs it; requires a tokenizer-spellability and deadlock analysis); the C2 surrogate prefix fix in `JSONGrammarState`; a completion reserve (close open containers near `max_tokens`); a whitespace-run guard (upstream has `WhitespaceRunTracker`).

Size: ~1,100 production lines touched (3 new files, `SchemaConstraintState` 461 → ~250, `ResponseFormatDecoder` 232 → ~380), ~1,200 test lines, 2 fixture files.

## Decisions

1. **Null support: no** in this iteration — Apple expresses optionals only by omission; `anyOf` and type arrays stay explicit 400s naming the keyword; adding `nullable` later is additive (first-byte dispatch on `n` collides with nothing).
2. **Caps:** `maxSchemaDepth = 32`, `maxSchemaNodes = 4096`.
3. **C3 root strictness: yes** (a behaviour change, in the CHANGELOG). `x-order` influences nothing. `const` is in.
4. **Known remaining Apple gaps, as follow-ups:** M2 non-ASCII (TripPlanner's `Lençóis Maranhenses`), non-object roots, `minimum`/`maximum`/`pattern`.

## Risks

- Token budget: nested documents are longer; a large `minItems` or deep nesting plus a small `max_tokens` truncates into incomplete JSON (an existing failure class, now more likely).
- Whitespace runaway: more structural gaps where whitespace is legal; follow-up guard.
- Sampling-path cost inside strings (~35 ms per full-vocabulary scan) unchanged but more relevant as documents get longer.
- Rewrite regression: mitigated by the verbatim port of existing tests, the differential test against the shipped automaton, the oracle fuzzing and the BFS check.
