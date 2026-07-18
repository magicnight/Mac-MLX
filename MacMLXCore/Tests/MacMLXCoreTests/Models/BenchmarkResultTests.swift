import Testing
import Foundation
@testable import MacMLXCore

@Test
func benchmarkResultRoundTripsThroughJSON() throws {
    let r = BenchmarkResult(
        modelID: "Qwen3-8B-4bit",
        engineID: .mlxSwift,
        promptTokens: 256,
        completionTokens: 512,
        promptTPS: 1200.5,
        generationTPS: 78.2,
        ttftMs: 142.0,
        timestamp: Date(timeIntervalSince1970: 1_745_000_000),
        system: SystemInfo(chip: "Apple M3 Pro", ramGB: 36)
    )
    let data = try JSONEncoder().encode(r)
    let decoded = try JSONDecoder().decode(BenchmarkResult.self, from: data)
    #expect(decoded.modelID == r.modelID)
    #expect(decoded.engineID == r.engineID)
    #expect(decoded.promptTPS == r.promptTPS)
    #expect(decoded.system.chip == "Apple M3 Pro")
    #expect(decoded.bottleneck == nil)   // absent by default
}

/// v0.7 backward compatibility: a benchmark JSON written before the `bottleneck`
/// field existed has no such key. The synthesized `Decodable` must decode it as a
/// nil optional (via `decodeIfPresent`) so old history still loads.
///
/// The result under test carries a real attribution so the encoded JSON actually
/// contains a `bottleneck` key — then we strip it. Encoding a nil-bottleneck result
/// instead would be a no-op strip (`encodeIfPresent` omits the key already), which
/// would not exercise the absent-key decode at all; the guard below asserts the key
/// was really present before removal.
@Test
func benchmarkResultDecodesLegacyJSONWithoutBottleneckField() throws {
    let r = BenchmarkResult(
        modelID: "Llama-3.2-3B-4bit",
        engineID: .mlxSwift,
        promptTokens: 128,
        completionTokens: 256,
        promptTPS: 900.0,
        generationTPS: 65.0,
        ttftMs: 110.0,
        timestamp: Date(timeIntervalSince1970: 1_745_000_000),
        system: SystemInfo(chip: "Apple M2", ramGB: 16)
    ).withBottleneck(
        BenchmarkBottleneck(
            category: .bandwidthBound,
            phase: .decode,
            advice: "decode appears bandwidth-bound",
            confidence: 0.8,
            restsOnEstimatedBandwidth: true,
            decodeFrameCount: 5,
            hardware: BenchmarkBottleneck.Readouts(
                peakGPUUtilization: 0.97,
                meanGPUUtilization: 0.9,
                peakBandwidthGBs: 118,
                meanBandwidthGBs: 110,
                peakThermalPressure: .fair
            )
        )
    )
    let data = try JSONEncoder().encode(r)
    var object = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any])
    // The key is genuinely present now; removing it simulates a pre-v0.7 file that
    // never wrote it. (Guard: prove the strip is not a no-op.)
    #expect(object["bottleneck"] != nil)
    object.removeValue(forKey: "bottleneck")
    let legacy = try JSONSerialization.data(withJSONObject: object)

    let decoded = try JSONDecoder().decode(BenchmarkResult.self, from: legacy)
    #expect(decoded.bottleneck == nil)
    #expect(decoded.modelID == "Llama-3.2-3B-4bit")
    #expect(decoded.generationTPS == 65.0)
}

/// A result carrying a bottleneck attribution round-trips it intact.
@Test
func benchmarkResultRoundTripsWithBottleneck() throws {
    let bottleneck = BenchmarkBottleneck(
        category: .bandwidthBound,
        phase: .decode,
        advice: "Decode appears memory-bandwidth-bound.",
        confidence: 0.9,
        restsOnEstimatedBandwidth: true,
        decodeFrameCount: 12,
        hardware: BenchmarkBottleneck.Readouts(
            peakGPUUtilization: 0.98,
            meanGPUUtilization: 0.95,
            peakBandwidthGBs: 120,
            meanBandwidthGBs: 112,
            peakThermalPressure: .fair
        )
    )
    let r = BenchmarkResult(
        modelID: "Qwen3-8B-4bit",
        engineID: .mlxSwift,
        promptTokens: 256,
        completionTokens: 512,
        promptTPS: 1200.5,
        generationTPS: 78.2,
        ttftMs: 142.0,
        system: SystemInfo(chip: "Apple M3 Pro", ramGB: 36)
    ).withBottleneck(bottleneck)

    let data = try JSONEncoder().encode(r)
    let decoded = try JSONDecoder().decode(BenchmarkResult.self, from: data)
    #expect(decoded.bottleneck == bottleneck)
}
