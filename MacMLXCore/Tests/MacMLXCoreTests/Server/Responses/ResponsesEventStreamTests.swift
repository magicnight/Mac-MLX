import Foundation
import Testing

@testable import MacMLXCore

// MARK: - ResponsesEventStream Tests
//
// Pure: the SSE frame format and the sequence counter.

@Suite("ResponsesEventStream")
struct ResponsesEventStreamTests {

    /// Splits a frame into its event name and decoded payload.
    private func parse(_ frame: String) throws -> (event: String, payload: [String: Any]) {
        #expect(frame.hasSuffix("\n\n"))
        let lines = frame.split(separator: "\n", omittingEmptySubsequences: true)
        #expect(lines.count == 2)
        let event = try #require(lines.first?.split(separator: " ", maxSplits: 1).last)
        let data = try #require(lines.last?.dropFirst("data: ".count))
        let payload = try #require(try JSONSerialization.jsonObject(with: Data(data.utf8)) as? [String: Any])
        return (String(event), payload)
    }

    @Test
    func frameIsAnEventLineThenADataLine() throws {
        var stream = ResponsesEventStream()
        let frame = try stream.frame("response.output_text.delta", ["delta": "hi", "output_index": 0])
        #expect(frame.hasPrefix("event: response.output_text.delta\ndata: {"))
        let (event, payload) = try parse(frame)
        #expect(event == "response.output_text.delta")
        #expect(payload["type"] as? String == "response.output_text.delta")
        #expect(payload["delta"] as? String == "hi")
        #expect(payload["sequence_number"] as? Int == 0)
    }

    @Test
    func sequenceNumbersStartAtZeroAndIncreaseByOne() throws {
        var stream = ResponsesEventStream()
        var numbers: [Int] = []
        for type in ["response.created", "response.in_progress", "response.completed"] {
            let (_, payload) = try parse(try stream.frame(type))
            numbers.append(try #require(payload["sequence_number"] as? Int))
        }
        #expect(numbers == [0, 1, 2])
        #expect(stream.nextSequenceNumber == 3)
    }
}
