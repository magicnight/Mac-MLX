import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// Track E — `LogitBiasProcessor` numeric behavior. The additive-bias math runs
/// on MLX, so the numeric checks are gated on a reachable Metal backend (they run
/// under xcodebuild; `swift test` skips them). The empty-init check is Metal-free.
@Suite("LogitBiasProcessor")
struct LogitBiasProcessorTests {

    @Test("empty bias yields no processor (Metal-free)")
    func emptyBiasIsNil() {
        #expect(LogitBiasProcessor(bias: [:]) == nil)
        #expect(LogitBiasProcessor(bias: [-1: 5]) == nil)  // only negative ids → dropped
    }

    @Test(
        "bias is added to exactly the named columns; out-of-range ids are ignored",
        .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
    func additiveBias() throws {
        let vocab = 5
        let base = MLXArray([Float(0), 1, 2, 3, 4]).reshaped([1, vocab])
        // id 99 is out of range for vocab 5 → must be a no-op, not a crash.
        let processor = try #require(LogitBiasProcessor(bias: [2: 3.0, 4: -1.0, 99: 5.0]))
        let out = processor.process(logits: base).reshaped([vocab]).asArray(Float.self)
        #expect(out[0] == 0.0)
        #expect(out[1] == 1.0)
        #expect(out[2] == 5.0)   // 2 + 3
        #expect(out[3] == 3.0)
        #expect(out[4] == 3.0)   // 4 - 1
    }
}
