import Testing

@testable import MacMLXCore

// MARK: - BatchRoutingPolicy Tests (A2d)
//
// Pure, MLX-free request-level gate for the continuous-batching server path.
// Model-level eligibility (VLM / architecture / cache) is the seam's job and is
// NOT covered here.

@Suite("BatchRoutingPolicy")
struct BatchRoutingPolicyTests {

    @Test
    func batchingDisabledNeverAttempts() {
        // No seam installed (the default-off switch) → always legacy, regardless
        // of the request. This is the zero-regression guarantee.
        #expect(!BatchRoutingPolicy.shouldAttemptBatch(batchingEnabled: false, hasDraftModel: false))
        #expect(!BatchRoutingPolicy.shouldAttemptBatch(batchingEnabled: false, hasDraftModel: true))
    }

    @Test
    func plainRequestWithSeamAttempts() {
        // Seam installed + an ordinary request (no speculative decoding) → attempt
        // the batched path. The seam still has the final say (may return nil).
        #expect(BatchRoutingPolicy.shouldAttemptBatch(batchingEnabled: true, hasDraftModel: false))
    }

    @Test
    func draftModelForcesSequentialEvenWithSeam() {
        // A resident draft model (speculative decoding) is mutually exclusive with
        // batching in v1 — mirrors mlx-lm's `is_batchable`.
        #expect(!BatchRoutingPolicy.shouldAttemptBatch(batchingEnabled: true, hasDraftModel: true))
    }
}
