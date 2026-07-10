import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// Track E — `LogprobsCaptureProcessor` records the sampled token's logprob and
/// the top-N alternatives without altering the logits. Metal-gated (logSoftmax /
/// argSort run on MLX).
///
/// Logits are `log(p)` for probs summing to 1, so `logSoftmax(logits) == logits`
/// exactly (`logsumexp == 0`): token0 = 0.50, token1 = 0.30, token2 = 0.20.
@Suite("LogprobsCaptureProcessor")
struct LogprobsCaptureProcessorTests {

    @Test(
        "captures the sampled token's logprob and top-N alternatives, logits unchanged",
        .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
    func capturesLogprobs() throws {
        let sink = LogprobsCaptureProcessor.Sink()
        var processor = LogprobsCaptureProcessor(inner: nil, sink: sink, topN: 2)
        let input: [Float] = [Float(log(0.5)), Float(log(0.3)), Float(log(0.2))]
        let logits = MLXArray(input).reshaped([1, 3])

        // process returns the logits untouched (capture only observes).
        let out = processor.process(logits: logits).reshaped([3]).asArray(Float.self)
        #expect(out == input)

        // Sampling token 1 records its logprob (log 0.3) and the top-2 (0, 1).
        processor.didSample(token: MLXArray([Int32(1)]))
        let entry = try #require(sink.popFirst())
        #expect(entry.tokenID == 1)
        #expect(abs(entry.logprob - Float(log(0.3))) < 1e-4)
        #expect(entry.top.map(\.id) == [0, 1])
        #expect(abs(entry.top[0].logprob - Float(log(0.5))) < 1e-4)
        #expect(abs(entry.top[1].logprob - Float(log(0.3))) < 1e-4)
        // FIFO drained.
        #expect(sink.popFirst() == nil)
    }

    @Test(
        "topN 0 records only the sampled token's logprob, no alternatives",
        .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
    func topNZero() throws {
        let sink = LogprobsCaptureProcessor.Sink()
        var processor = LogprobsCaptureProcessor(inner: nil, sink: sink, topN: 0)
        let logits = MLXArray([Float(log(0.5)), Float(log(0.5))]).reshaped([1, 2])
        _ = processor.process(logits: logits)
        processor.didSample(token: MLXArray([Int32(0)]))
        let entry = try #require(sink.popFirst())
        #expect(entry.tokenID == 0)
        #expect(entry.top.isEmpty)
    }
}
