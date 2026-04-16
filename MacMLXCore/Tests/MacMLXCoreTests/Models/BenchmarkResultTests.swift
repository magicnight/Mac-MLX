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
}
