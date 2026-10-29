// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Encodes the server-sent events of a streamed Responses API response.
///
/// Each frame is `event: <type>\ndata: <json>\n\n`; the payload carries its
/// `type` and a `sequence_number` that starts at 0 and grows by one per
/// frame, as OpenAI's stream does. There is no `[DONE]` sentinel: the stream
/// ends after its one terminal event (`response.completed`, `.incomplete` or
/// `.failed`). A value type owned by the writer loop, which mutates it once
/// per frame.
struct ResponsesEventStream: Sendable {

    /// The sequence number the next frame gets.
    private(set) var nextSequenceNumber = 0

    /// The SSE frame for one event; `payload` holds its fields other than
    /// `type` and `sequence_number`.
    mutating func frame(_ type: String, _ payload: [String: Any] = [:]) throws -> String {
        var object = payload
        object["type"] = type
        object["sequence_number"] = nextSequenceNumber
        let data = try JSONSerialization.data(withJSONObject: object)
        nextSequenceNumber += 1
        return "event: \(type)\ndata: \(String(decoding: data, as: UTF8.self))\n\n"
    }
}
