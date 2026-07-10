import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import MacMLXCore

/// Track E — `XTCSampler` mask semantics, deterministic via `probability = 1.0`
/// (XTC always applies, so the random gate never matters) and an `ArgMaxSampler`
/// base (so the observed token reveals which columns survived the mask). Metal-
/// gated: the softmax/threshold/mask run on MLX.
///
/// Logits are `log(p)` for a hand-picked distribution, so `softmax(logits) == p`
/// exactly (the four probs sum to 1, so `logsumexp == 0`):
///   token0 = 0.50, token1 = 0.30, token2 = 0.15, token3 = 0.05
@Suite("XTCSampler")
struct XTCSamplerTests {

    private func logits() -> MLXArray {
        MLXArray([Float(log(0.5)), Float(log(0.3)), Float(log(0.15)), Float(log(0.05))])
            .reshaped([1, 4])
    }

    @Test(
        "XTC excludes the top choices, keeping the least-probable above threshold",
        .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
    func excludesTopChoices() {
        // threshold 0.1 ⇒ candidates {0:0.5, 1:0.3, 2:0.15}; least is token2, so
        // tokens 0 and 1 are masked to -inf. ArgMax over the survivors {2, 3} → 2.
        let sampler = XTCSampler(base: ArgMaxSampler(), probability: 1.0, threshold: 0.1)
        #expect(sampler.sample(logits: logits()).item(Int.self) == 2)
    }

    @Test(
        "with fewer than two tokens above threshold XTC is a no-op",
        .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
    func noOpWhenNoCandidates() {
        // threshold 0.6 ⇒ no token's prob exceeds it, so nothing is masked and the
        // argmax is the natural top token 0. (Applied unconditionally at prob 1.0.)
        let sampler = XTCSampler(base: ArgMaxSampler(), probability: 1.0, threshold: 0.6)
        #expect(sampler.sample(logits: logits()).item(Int.self) == 0)
    }
}
