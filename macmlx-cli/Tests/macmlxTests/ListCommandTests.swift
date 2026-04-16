import Testing
import Foundation
@testable import macmlx
import MacMLXCore

/// Tests for the `macmlx list` command.
///
/// These tests exercise JSON output shape by using the internal `LocalModel`
/// Codable conformance rather than spawning a subprocess.
@Suite("ListCommand")
struct ListCommandTests {

    @Test
    func localModelEncodesAsExpectedJSON() throws {
        let model = LocalModel(
            id: "Qwen3-8B-4bit",
            displayName: "Qwen3-8B-4bit",
            directory: URL(filePath: "/tmp/models/Qwen3-8B-4bit"),
            sizeBytes: 4_500_000_000,
            format: .mlx,
            quantization: "4bit",
            parameterCount: nil,
            architecture: nil
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode([model])
        let json = try JSONDecoder().decode([[String: JSONValue]].self, from: data)

        let first = try #require(json.first)
        #expect(first["id"] == .string("Qwen3-8B-4bit"))
        #expect(first["displayName"] == .string("Qwen3-8B-4bit"))
        #expect(first["quantization"] == .string("4bit"))
    }

    @Test
    func humanSizeFormatsCorrectly() {
        let model = LocalModel(
            id: "test",
            displayName: "test",
            directory: URL(filePath: "/tmp"),
            sizeBytes: 4_500_000_000,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        #expect(model.humanSize == "4.50 GB")
    }

    @Test
    func humanSizeMBFormatsCorrectly() {
        let model = LocalModel(
            id: "test",
            displayName: "test",
            directory: URL(filePath: "/tmp"),
            sizeBytes: 500_000_000,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        #expect(model.humanSize == "500 MB")
    }
}

// MARK: - JSON decode helpers

/// Minimal JSON value type for test assertions.
enum JSONValue: Equatable, Decodable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let n = try? container.decode(Double.self) {
            self = .number(n)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let a = try? container.decode([JSONValue].self) {
            self = .array(a)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }
}
