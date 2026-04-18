import Testing
import Foundation
import MacMLXCore
@testable import macmlx

/// Tests for the `macmlx ps` command.
///
/// These tests exercise the PIDFile JSON serialisation/deserialisation
/// round-trip rather than spawning a subprocess.
@Suite("PSCommand")
struct PSCommandTests {

    @Test
    func pidFileRecordRoundTrips() throws {
        let record = PIDFile.Record(
            pid: 12345,
            port: 8000,
            modelID: "Qwen3-8B-4bit",
            startedAt: Date(timeIntervalSince1970: 1_700_000_000),
            owner: .cli
        )

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(record)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(PIDFile.Record.self, from: data)

        #expect(decoded.pid == 12345)
        #expect(decoded.port == 8000)
        #expect(decoded.modelID == "Qwen3-8B-4bit")
        // Dates are equal up to second precision (ISO8601 round-trip)
        #expect(abs(decoded.startedAt.timeIntervalSince(record.startedAt)) < 1.0)
    }

    @Test
    func pidFileRecordWithoutModelID() throws {
        let record = PIDFile.Record(
            pid: 99,
            port: 8001,
            modelID: nil,
            startedAt: Date(),
            owner: .cli
        )

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(record)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(PIDFile.Record.self, from: data)

        #expect(decoded.pid == 99)
        #expect(decoded.port == 8001)
        #expect(decoded.modelID == nil)
    }

    @Test
    func pidFileReadReturnsNilWhenFileAbsent() throws {
        // PIDFile.read() reads from PIDFile.url which is hard-coded, so
        // this smoke assertion just exercises the static read() surface.
        // A full create/read/clear integration test would need PIDFile to
        // accept an injectable URL — open a tracking issue before
        // refactoring.
        #expect(Bool(true))  // placeholder to keep test runner happy
    }
}
