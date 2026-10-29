import Testing
import Foundation
import MLX
@testable import MacMLXCore

/// Guards GPU affine quantization against `ml-explore/mlx#4775`.
///
/// The Metal `affine_quantize` kernel (`mlx/backend/metal/kernels/quantized.h`)
/// started its running maximum at 0 instead of the most negative value, so a
/// group whose values are all negative was quantized over `[min, 0]` rather
/// than `[min, max]`: a scale too coarse for the values actually present, and
/// a reconstruction error of up to half that scale where the true range would
/// have been exact. The CPU implementation always started at `-inf`, so the
/// two streams disagreed. It is built by the JIT from the stringified
/// `mlx-generated/quantized.cpp`, which is where the fork patches it.
///
/// macMLX reaches it whenever a request sets `kv_bits`: `QuantizedKVCache`
/// quantizes keys and values on the GPU as they are written, one group per
/// `kv_group_size` slice of a head, and a slice whose values are all negative
/// is quantized over the wrong range. Every `MLX.quantized` call on the
/// default stream takes the kernel, whatever the shape: `fast::Quantize`
/// has no GPU fallback for the affine mode at the group sizes (32, 64, 128)
/// and bit widths (2–6, 8) this vintage accepts.
///
/// The groups are upstream's: integer grids that span exactly `2^bits - 1`
/// units, shifted all negative, all positive, and across zero, plus a constant
/// negative and a zero group. On those the correct scale is 1 and the bias an
/// integer, so dequantizing must give back the input exactly; that is the
/// expected value, and the CPU stream's quantizer is checked against it too.
/// Against the fork's unpatched base (1026d23) all 18 group-size and bit-width
/// cases came back with max |diff| 1.0 on the GPU and 0 on the CPU.
@Suite(
    "Affine quantize on signed groups",
    .enabled(if: mlxMetallibIsAvailable, "Requires default.metallib (run under xcodebuild)"))
struct AffineQuantizeSignedGroupTests {

    private static let cases: [(groupSize: Int, bits: Int)] = [32, 64, 128].flatMap { groupSize in
        [2, 3, 4, 5, 6, 8].map { (groupSize: groupSize, bits: $0) }
    }

    /// Five groups, one row each: upstream `test_affine_quantize_signed_groups`.
    private func signedGroups(groupSize: Int, bits: Int) -> MLXArray {
        let bins = (1 << bits) - 1
        var grid = (0 ..< groupSize).map { $0 % (bins + 1) }
        grid[groupSize - 1] = bins
        let rows: [[Int]] = [
            grid.map { $0 - 2 * bins },  // all negative: the case #4775 broke
            grid.map { 2 * bins - $0 },  // all positive
            grid.map { $0 - (1 << (bits - 1)) },  // straddles zero
            Array(repeating: -2, count: groupSize),
            Array(repeating: 0, count: groupSize),
        ]
        return MLXArray(rows.flatMap { $0 }.map(Float.init), [rows.count, groupSize])
    }

    private func roundTripError(_ w: MLXArray, groupSize: Int, bits: Int, stream: StreamOrDevice)
        -> Float
    {
        let (wq, scales, biases) = MLX.quantized(
            w, groupSize: groupSize, bits: bits, stream: stream)
        let restored = MLX.dequantized(
            wq, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, stream: stream)
        return MLX.max(MLX.abs(w - restored)).item(Float.self)
    }

    @Test(
        "GPU affine quantization restores unit-spaced signed groups exactly",
        arguments: cases)
    func gpuRoundTripIsExact(groupSize: Int, bits: Int) {
        let w = signedGroups(groupSize: groupSize, bits: bits)

        let cpuError = roundTripError(w, groupSize: groupSize, bits: bits, stream: .cpu)
        let gpuError = roundTripError(w, groupSize: groupSize, bits: bits, stream: .gpu)

        #expect(
            cpuError < 1e-5,
            "the CPU quantizer, the reference, no longer restores the grids: max |diff| \(cpuError)")
        #expect(
            gpuError < 1e-5,
            """
            GPU affine quantization did not restore unit-spaced groups \
            (group size \(groupSize), \(bits) bits): max |diff| \(gpuError), \
            CPU \(cpuError). This is ml-explore/mlx#4775 — affine_quantize \
            starts w_max at 0, so an all-negative group is quantized over \
            [min, 0].
            """)
    }
}
