// Copyright © 2026 macMLX. English comments only.

import Foundation

@testable import MacMLXCore

/// Loads the JSON-schema fixtures of the structured-output tests (see
/// `Fixtures/README.md` for where each one comes from).
enum StructuredOutputFixtures {

    struct FixtureError: Error, CustomStringConvertible {
        let description: String
    }

    /// One upstream golden: a schema and a document generated under it.
    struct Golden {
        let schema: JSONValue
        let document: String
    }

    static func data(_ name: String) throws -> Data {
        guard let url = Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures") else {
            throw FixtureError(description: "\(name).json is missing from the test bundle")
        }
        return try Data(contentsOf: url)
    }

    static func json(_ name: String) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: data(name))
    }

    /// One schema of `fm_generable_schemas_fixture.json`: `Person`,
    /// `PersonNoRange` or `Explicit`.
    static func generable(_ key: String) throws -> JSONValue {
        guard case .object(let schemas) = try json("fm_generable_schemas_fixture"), let schema = schemas[key] else {
            throw FixtureError(description: "fm_generable_schemas_fixture.json has no '\(key)'")
        }
        return schema
    }

    /// Apple's TripPlanner `Itinerary` schema as the framework emits it. One
    /// enum value is non-ASCII, so this compiles to a 400 today.
    static func itinerary() throws -> JSONValue {
        try json("fm_itinerary_production_fixture")
    }

    /// ``itinerary()`` with its non-ASCII enum value removed, which the
    /// supported subset can compile.
    static func asciiItinerary() throws -> JSONValue {
        withoutNonASCIIEnumValues(try itinerary())
    }

    /// Upstream's golden for constrained-decoding tier `tier` (1–4).
    static func golden(tier: Int) throws -> Golden {
        struct File: Decodable {
            let schema: String
            let document: String
        }
        let file = try JSONDecoder().decode(File.self, from: data("schema_tier\(tier)_steps"))
        let schema = try JSONDecoder().decode(JSONValue.self, from: Data(file.schema.utf8))
        return Golden(schema: schema, document: file.document)
    }

    /// `schema` inside the OpenAI `json_schema` response-format envelope.
    static func responseFormat(_ schema: JSONValue, name: String = "T") -> JSONValue {
        .object([
            "type": .string("json_schema"),
            "json_schema": .object(["name": .string(name), "strict": .bool(true), "schema": schema]),
        ])
    }

    /// Compile `schema` through ``ResponseFormatDecoder``.
    static func compile(_ schema: JSONValue) throws -> JSONSchemaObject {
        guard case .jsonSchema(let object)? = try ResponseFormatDecoder.decode(responseFormat(schema)) else {
            throw FixtureError(description: "the decoder returned no json_schema constraint")
        }
        return object
    }

    /// Drop every non-ASCII string from every `enum` array, recursively.
    static func withoutNonASCIIEnumValues(_ value: JSONValue) -> JSONValue {
        switch value {
        case .object(let members):
            var result: [String: JSONValue] = [:]
            for (key, member) in members {
                if key == "enum", case .array(let entries) = member {
                    result[key] = .array(entries.filter { entry in
                        guard case .string(let text) = entry else { return true }
                        return text.unicodeScalars.allSatisfy(\.isASCII)
                    })
                } else {
                    result[key] = withoutNonASCIIEnumValues(member)
                }
            }
            return .object(result)
        case .array(let items):
            return .array(items.map(withoutNonASCIIEnumValues))
        default:
            return value
        }
    }
}
