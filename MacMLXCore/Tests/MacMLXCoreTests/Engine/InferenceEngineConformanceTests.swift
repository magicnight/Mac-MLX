import Testing
import Foundation
@testable import MacMLXCore

@Test
func stubEngineReportsExpectedInitialState() async {
    let engine = StubInferenceEngine(engineID: .mlxSwift)
    let id = await engine.engineID
    let status = await engine.status
    let loaded = await engine.loadedModel
    let version = await engine.version
    #expect(id == .mlxSwift)
    #expect(status == .idle)
    #expect(loaded == nil)
    #expect(version == "stub-1")
}

@Test
func stubEngineLoadTransitionsStateToReady() async throws {
    let engine = StubInferenceEngine(engineID: .mlxSwift)
    let model = LocalModel(
        id: "test", displayName: "Test",
        directory: URL(filePath: "/tmp"),
        sizeBytes: 0, format: .mlx,
        quantization: nil, parameterCount: nil, architecture: nil
    )
    try await engine.load(model)
    let status = await engine.status
    let loaded = await engine.loadedModel
    #expect(status == .ready(model: "test"))
    #expect(loaded?.id == "test")
}

@Test
func stubEngineGenerateYieldsExpectedChunks() async throws {
    let engine = StubInferenceEngine(engineID: .mlxSwift)
    let model = LocalModel(
        id: "test", displayName: "Test",
        directory: URL(filePath: "/tmp"),
        sizeBytes: 0, format: .mlx,
        quantization: nil, parameterCount: nil, architecture: nil
    )
    try await engine.load(model)
    let request = GenerateRequest(model: "test", messages: [
        ChatMessage(role: .user, content: "hi")
    ])
    var collected: [String] = []
    for try await chunk in engine.generate(request) {
        collected.append(chunk.text)
    }
    #expect(collected == ["stub-", "response"])
}

@Test
func stubEngineHealthCheckReturnsTrue() async {
    let engine = StubInferenceEngine(engineID: .mlxSwift)
    let healthy = await engine.healthCheck()
    #expect(healthy == true)
}
