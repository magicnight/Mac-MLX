import Testing
import Foundation
import MLX
import MLXFast
@testable import MacMLXCore

/// Parity guards for two upstream NAX defects, both of which produce wrong
/// numbers rather than errors.
///
/// - `ml-explore/mlx#3361` — the NAX attention kernel holds tile positions in
///   `short`, so a KV sequence past 32767 wraps negative. The overflowing
///   declarations live in two disjoint blocks, the causal one and the
///   array-mask one, and each test below covers one of them.
/// - `ml-explore/mlx#3422` — "AddMM was completely broken on NAX", because the
///   epilogue took its destination fragment by value and dropped the bias. Our
///   tree predates the refactor that introduced it, so this is a guard against
///   regressing into that shape, not a carried fix.
///
/// A caveat these tests cannot state in code: NAX needs macOS 26.2+ and an M5
/// generation GPU. Anywhere else the kernels below are the generic Metal ones
/// and every case passes regardless, so a green run on CI — which is
/// paravirtualized — is not evidence that anything here is pinned. Only a
/// local run on M5 hardware is.
@Suite(
    "NAX attention and addmm parity",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct NAXAttentionAndAddMMParityTests {

    // Peaked enough that attention actually selects. With q, k ~ N(0, s²) at
    // head dim D and scale 1/√D, the logits have standard deviation s² — so
    // the 0.05 that keeps other parity tests in a comfortable numeric range
    // would put every logit within ±0.01 of zero, make the softmax uniform,
    // and reduce both tests to "is the output finite". At s = 1 the weights
    // concentrate, and reading the wrong keys changes the answer.
    private static let inputScale: Float = 1.0

    private static let heads = 2
    private static let headDim = 64
    private static let queries = 128

    private func cosineSimilarity(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let a = lhs.asType(.float32).flattened()
        let b = rhs.asType(.float32).flattened()
        let dot = MLX.sum(a * b).item(Float.self)
        let na = MLX.sqrt(MLX.sum(a * a)).item(Float.self)
        let nb = MLX.sqrt(MLX.sum(b * b)).item(Float.self)
        guard na > 0, nb > 0 else { return 0 }
        return dot / (na * nb)
    }

    /// Query, key and value for one case, plus the float32 logits the
    /// references are built from.
    private func inputs(kvLength: Int, seed: UInt64) -> (q: MLXArray, k: MLXArray, v: MLXArray, logits: MLXArray) {
        let heads = Self.heads, headDim = Self.headDim, queries = Self.queries
        MLXRandom.seed(seed)
        let q = MLXRandom.normal([1, heads, queries, headDim], scale: Self.inputScale).asType(.bfloat16)
        let k = MLXRandom.normal([1, heads, kvLength, headDim], scale: Self.inputScale).asType(.bfloat16)
        let v = MLXRandom.normal([1, heads, kvLength, headDim], scale: Self.inputScale).asType(.bfloat16)
        let scale = 1.0 / Float(headDim).squareRoot()
        let logits = MLX.matmul(
            q.asType(.float32) * scale, k.asType(.float32).transposed(0, 1, 3, 2))
        return (q, k, v, logits)
    }

    // MARK: - mlx#3361, array-mask block

    /// The array-mask block reads the mask through `load_safe`, whose bounds
    /// check is `r < lim_x && (c + j) < lim_y` with no lower bound — so a
    /// position that has wrapped negative passes the check and indexes before
    /// the buffer. That is why this case announces itself as non-finite output
    /// rather than as a subtly different answer.
    ///
    /// The mask is a ramp rather than a constant precisely so that the weaker
    /// failure is visible too: reading a *different in-bounds* entry of an
    /// all-zeros mask would be indistinguishable from reading the right one.
    ///
    /// Query length decides the kernel. MLX routes to the vector path at
    /// `query_sequence_length <= 8` and to the full steel/NAX path above it,
    /// and only the latter carries the overflowing arithmetic — a single-row
    /// query exercises a kernel that never had the bug and passes regardless.
    /// bfloat16 at head dim 64 is what then selects NAX over generic steel.
    ///
    /// KV=8192 is the control: if only the long case moves, the 32767 boundary
    /// is what moved, not attention in general.
    @Test(
        "array-masked attention matches its float32 reference either side of 32K KV",
        arguments: [8192, 40960])
    func arrayMaskedAttentionMatchesFloat32Reference(kvLength: Int) {
        let (q, k, v, logits) = inputs(kvLength: kvLength, seed: 11)

        // A bounded, position-dependent additive mask.
        let ramp = (MLXArray(0 ..< kvLength) % 13).asType(.float32) * -0.05
        let maskF32 = ramp.reshaped([1, 1, 1, kvLength])
            + MLXArray.zeros([1, 1, Self.queries, kvLength], type: Float32.self)

        let underTest = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 1.0 / Float(Self.headDim).squareRoot(),
            mask: .array(maskF32.asType(.bfloat16)))

        let reference = MLX.matmul(MLX.softmax(logits + maskF32, axis: -1), v.asType(.float32))
        MLX.eval(underTest, reference)

        let similarity = cosineSimilarity(underTest, reference)
        #expect(
            similarity > 0.99,
            """
            array-masked attention diverged from its float32 reference at KV=\
            \(kvLength): cosine \(similarity). Past 32767 on NAX hardware this \
            is ml-explore/mlx#3361 — the mask-block tile positions overflow \
            `short`.
            """)
        #expect(
            MLX.all(MLX.isFinite(underTest)).item(Bool.self),
            "array-masked attention produced non-finite values at KV=\(kvLength)")
    }

    // MARK: - mlx#3361, causal block

    /// This is NOT a guard for `mlx#3361`, and measurement says so: it passes
    /// against the pre-fix pin at KV=40960, where the array-mask case above
    /// fails outright.
    ///
    /// The reason is worth keeping. The causal block widened the same two
    /// declarations, but it uses them only in `(r < c) ? neg_inf : fg[loc]`.
    /// When `r` and `c` both wrap, they shift by the same 65536 and the
    /// comparison is unchanged — the overflow cancels. They always do both
    /// wrap, because the block is gated on `kb >= kb_lim - ~2 blocks`, so it
    /// runs only near the diagonal where `base_col` is within two tiles of
    /// `base_row`. Upstream's change there is cleanup; ours followed it for
    /// uniformity. Only the mask block, which feeds the position to
    /// `load_safe` as an *index*, can actually go wrong.
    ///
    /// What this does hold is ordinary causal parity at a KV length nothing
    /// else in the suite reaches, either side of the same boundary.
    @Test(
        "causal attention matches its float32 reference either side of 32K KV",
        arguments: [8192, 40960])
    func causalAttentionMatchesFloat32Reference(kvLength: Int) {
        let (q, k, v, logits) = inputs(kvLength: kvLength, seed: 13)

        let underTest = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 1.0 / Float(Self.headDim).squareRoot(),
            mask: .causal)

        // MLX aligns a short query with the tail of the KV sequence, so query
        // i attends to keys 0 ... (kvLength - queries + i).
        let queryRow = (MLXArray(0 ..< Self.queries) + (kvLength - Self.queries))
            .reshaped([Self.queries, 1])
        let keyColumn = MLXArray(0 ..< kvLength).reshaped([1, kvLength])
        let blocked = (keyColumn .> queryRow).asType(.float32) * -1e9
        let reference = MLX.matmul(
            MLX.softmax(logits + blocked.reshaped([1, 1, Self.queries, kvLength]), axis: -1),
            v.asType(.float32))
        MLX.eval(underTest, reference)

        let similarity = cosineSimilarity(underTest, reference)
        #expect(
            similarity > 0.99,
            """
            causal attention diverged from its float32 reference at KV=\
            \(kvLength): cosine \(similarity). Past 32767 on NAX hardware this \
            is ml-explore/mlx#3361 — the causal-block tile positions overflow \
            `short` and mask the wrong cells.
            """)
        #expect(
            MLX.all(MLX.isFinite(underTest)).item(Bool.self),
            "causal attention produced non-finite values at KV=\(kvLength)")
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
