import Testing
@testable import MacMLXCore
import Foundation

// MARK: - MLXSwiftEngine unit tests
//
// These tests cover state-machine behaviour and error paths only.
// They do NOT load a real MLX model (no network or disk access needed).

@Suite("MLXSwiftEngine")
struct MLXSwiftEngineTests {

    // MARK: Initial state

    @Test("New engine starts idle with correct identity and version")
    func newEngineStartsIdle() async {
        let e = MLXSwiftEngine()
        let s = await e.status
        let id = await e.engineID
        let v = await e.version
        #expect(s == .idle)
        #expect(id == .mlxSwift)
        #expect(v.contains("mlx-swift-lm"))
    }

    @Test("New engine has no loaded model")
    func newEngineHasNoLoadedModel() async {
        let e = MLXSwiftEngine()
        let m = await e.loadedModel
        #expect(m == nil)
    }

    // MARK: Load failure

    @Test("Loading a nonexistent directory throws EngineError.modelLoadFailed and sets error status")
    func loadingNonexistentDirectoryFailsWithEngineError() async {
        let e = MLXSwiftEngine()
        let bogus = LocalModel(
            id: "nonexistent-\(UUID().uuidString)",
            displayName: "Bogus",
            directory: URL(filePath: "/tmp/definitely-not-a-model-\(UUID().uuidString)"),
            sizeBytes: 0,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
        do {
            try await e.load(bogus)
            Issue.record("Expected load(_:) to throw, but it succeeded")
        } catch let err as EngineError {
            switch err {
            case .modelLoadFailed:
                // Expected path — pass
                break
            default:
                Issue.record("Got EngineError but wrong variant: \(err)")
            }
        } catch {
            Issue.record("Expected EngineError, got \(type(of: error)): \(error)")
        }

        let status = await e.status
        if case .error = status {
            // Correct — pass
        } else {
            Issue.record("Expected status to be .error after failed load, got \(status)")
        }

        // loadedModel must be nil after a failed load
        let model = await e.loadedModel
        #expect(model == nil)
    }

    // MARK: Generate without a loaded model

    @Test("generate(_:) without a loaded model yields EngineError.modelNotLoaded")
    func generateWithoutLoadedModelYieldsModelNotLoadedError() async {
        let e = MLXSwiftEngine()
        let req = GenerateRequest(
            model: "x",
            messages: [ChatMessage(role: .user, content: "hi")]
        )
        do {
            for try await _ in e.generate(req) { /* drain */ }
            Issue.record("Expected stream to throw, but it finished normally")
        } catch let err as EngineError {
            #expect(err == .modelNotLoaded)
        } catch {
            Issue.record("Expected EngineError.modelNotLoaded, got \(type(of: error)): \(error)")
        }
    }

    // MARK: Health check

    @Test("healthCheck returns true when engine is idle (no model loaded)")
    func healthCheckReturnsTrueWhenIdle() async {
        let e = MLXSwiftEngine()
        let healthy = await e.healthCheck()
        #expect(healthy == true)
    }

    // MARK: Unload

    @Test("unload on an idle engine leaves status idle")
    func unloadOnIdleEngineRemainsIdle() async throws {
        let e = MLXSwiftEngine()
        try await e.unload()
        let s = await e.status
        #expect(s == .idle)
        let m = await e.loadedModel
        #expect(m == nil)
    }
}
