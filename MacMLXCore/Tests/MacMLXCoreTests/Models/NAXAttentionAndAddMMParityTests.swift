import Testing
import Foundation
import MLX
import MLXFast
@testable import MacMLXCore

/// Parity guards for two upstream NAX defects, both of which produce wrong
/// numbers rather than errors.
///
/// - `ml-explore/mlx#3361` — the NAX attention kernel holds tile positions in
///   `short`, so a KV sequence past 32767 wraps negative and the mask is read
///   from the wrong offsets. Our vendored tree carries the same declarations.
/// - `ml-explore/mlx#3422` — "AddMM was completely broken on NAX", because the
///   epilogue took its destination fragment by value and dropped the bias. Our
///   tree predates the refactor that introduced it, so this is a guard against
///   regressing into that shape, not a carried fix.
///
/// On hardware without NAX both degrade to ordinary parity checks and pass
/// either way, so the suite is safe to run anywhere.
@Suite(
    "NAX attention and addmm parity",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct NAXAttentionAndAddMMParityTests {

    private func cosineSimilarity(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let a = lhs.asType(.float32).flattened()
        let b = rhs.asType(.float32).flattened()
        let dot = MLX.sum(a * b).item(Float.self)
        let na = MLX.sqrt(MLX.sum(a * a)).item(Float.self)
        let nb = MLX.sqrt(MLX.sum(b * b)).item(Float.self)
        guard na > 0, nb > 0 else { return 0 }
        return dot / (na * nb)
    }

    // MARK: - mlx#3361 — mask indexing past a 32K KV sequence

    /// A masked attention whose KV sequence sits either side of the 32767 the
    /// tile positions overflow at. The short case is the control: if only the
    /// long one diverges, the boundary — not attention in general — is what
    /// moved.
    ///
    /// The query length matters. MLX routes to the vector kernel at
    /// `query_sequence_length <= 8` and to the full (steel/NAX) kernel above
    /// it, and only the full kernel carries the overflowing arithmetic — so a
    /// single-row query would exercise the wrong kernel and pass regardless.
    /// bfloat16 at head dim 64 is what selects NAX over the generic steel path.
    @Test(
        "masked attention matches its float32 reference either side of 32K KV",
        arguments: [8192, 40960])
    func maskedAttentionMatchesFloat32Reference(kvLength: Int) {
        let heads = 2, headDim = 64, queries = 128
        let scale = 1.0 / Float(headDim).squareRoot()

        MLXRandom.seed(11)
        let q = MLXRandom.normal([1, heads, queries, headDim], scale: 0.05).asType(.bfloat16)
        let k = MLXRandom.normal([1, heads, kvLength, headDim], scale: 0.05).asType(.bfloat16)
        let v = MLXRandom.normal([1, heads, kvLength, headDim], scale: 0.05).asType(.bfloat16)

        // An additive mask that admits everything: any indexing overflow shows
        // up as attention reading the wrong mask entries, not as a different
        // masking policy.
        let mask = MLXArray.zeros([1, 1, queries, kvLength], type: Float32.self)
            .asType(.bfloat16)

        let underTest = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(mask))

        // Reference: the same attention in float32, spelled out.
        let scores = MLX.softmax(
            MLX.matmul(q.asType(.float32) * scale, k.asType(.float32).transposed(0, 1, 3, 2)),
            axis: -1)
        let reference = MLX.matmul(scores, v.asType(.float32))
        MLX.eval(underTest, reference)

        let similarity = cosineSimilarity(underTest, reference)
        #expect(
            similarity > 0.99,
            """
            masked attention diverged from its float32 reference at KV=\
            \(kvLength): cosine \(similarity). Past 32767 on NAX hardware this \
            is ml-explore/mlx#3361 — the tile positions overflow `short`.
            """)
        #expect(
            MLX.all(MLX.isFinite(underTest)).item(Bool.self),
            "masked attention produced non-finite values at KV=\(kvLength)")
    }

    // MARK: - mlx#3422 — addmm on NAX

    /// `addmm` is `bias + a @ b`, which is what every biased `Linear` becomes.
    /// Upstream's NAX epilogue once bound its destination fragment by value, so
    /// the bias was accumulated into a copy and discarded; our vendored tree
    /// binds a reference and never had it. This holds that property.
    @Test(
        "addmm matches an explicit bias-plus-matmul",
        arguments: [(512, 512, 512), (1024, 768, 2048), (2048, 2048, 512)])
    func addmmMatchesExplicitBiasPlusMatmul(m: Int, k: Int, n: Int) {
        MLXRandom.seed(7)
        let a = MLXRandom.normal([m, k], scale: 0.05).asType(.bfloat16)
        let b = MLXRandom.normal([k, n], scale: 0.05).asType(.bfloat16)
        let bias = MLXRandom.normal([n], scale: 0.05).asType(.bfloat16)

        let viaAddMM = MLX.addMM(bias, a, b)
        let explicit = MLX.matmul(a, b) + bias
        MLX.eval(viaAddMM, explicit)

        let similarity = cosineSimilarity(viaAddMM, explicit)
        #expect(
            similarity > 0.999,
            """
            addmm diverged from an explicit bias + matmul (M=\(m), K=\(k), \
            N=\(n)): cosine \(similarity). On NAX hardware this is the shape \
            ml-explore/mlx#3422 fixed.
            """)
    }
}
