import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
import os

@testable import MacMLXCore

// MARK: - MLXSwiftEngine unload drain (#136)
//
// `unload()` hands MLX's cached buffers back after releasing a resident
// model. The drain initializes Metal, which the SPM test job cannot do, so
// the engine's drain is a seam here, and the model is a container with no
// weights: enough for `unload()` and `load()` to treat it as resident. What
// these tests pin down: no drain for an engine that never held a model; one
// drain, after the container is gone, for an engine that did, through
// whichever of its three references held it; a load over a resident model
// releasing it first and draining again when the load fails; a load that
// fails once any model has loaded in the process draining once; the draft
// model dropped or swapped out draining as well; and an unload while runs
// are counted waiting for the last one to end (the counter is driven
// directly here; no generation runs). The real-weights numbers, and the
// streams, are the gated tests in `Mellum2SmokeTests`.
//
// The weightless container stays MLX-free only as long as `Module.init()`,
// `ModelContext.init` (which walks the module tree to set eval mode) and
// `ModelContainer.init` create no arrays, which they do not in mlx-swift-lm
// 3.32.3.

/// A model with no weights, enough `LanguageModel` to sit in a container. No
/// method is ever called.
private final class WeightlessModel: Module, LanguageModel {
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        fatalError("a weightless model is never run")
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        fatalError("a weightless model is never run")
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("a weightless model is never run")
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        fatalError("a weightless model is never run")
    }

    func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus {
        fatalError("a weightless model is never run")
    }
}

private struct NoInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("a weightless model takes no input")
    }
}

private struct NoTokenizer: Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Watches whether the container is gone by the time a drain runs.
private final class ContainerWatch: @unchecked Sendable {
    weak var container: ModelContainer?
    init(_ container: ModelContainer) { self.container = container }
}

@Suite("MLXSwiftEngine unload drain (#136)")
struct MLXSwiftEngineUnloadTests {

    /// One entry per drain: whether the container had already been released.
    private typealias Drains = OSAllocatedUnfairLock<[Bool]>

    /// No on-disk prompt cache tier: an engine with one sweeps and prunes the
    /// real `~/.mac-mlx/kv-cache` when it is built, which a unit test has no
    /// business doing.
    private static let noColdTier = PromptCacheConfig(coldEnabled: false)

    private func model(_ id: String, format: ModelFormat) -> LocalModel {
        LocalModel(
            id: id,
            displayName: id,
            directory: URL(fileURLWithPath: "/nonexistent/\(id)-\(UUID().uuidString)", isDirectory: true),
            sizeBytes: 0,
            format: format,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
    }

    /// A GGUF model is refused by the format switch, before any factory or
    /// Metal work: the shape of a load that fails without touching MLX.
    private func refused() -> LocalModel { model("refused", format: .gguf) }

    /// An MLX model whose directory does not exist: the factory fails on the
    /// config read, before any MLX work, through the generic catch (the one a
    /// tokenizer failure after the weights are in would take).
    private func missing() -> LocalModel { model("missing", format: .mlx) }

    /// An engine whose drains are recorded instead of performed, in a
    /// process where no model has loaded (the gated smokes may have loaded
    /// one before these run; what they left behind is not this suite's).
    private func countingEngine() async -> (MLXSwiftEngine, Drains) {
        let engine = MLXSwiftEngine(promptCache: Self.noColdTier)
        let drains = Drains(initialState: [])
        await engine.setReleaseCachedBuffers { drains.withLock { $0.append(true) } }
        await engine.setMetalKnownUp { false }
        return (engine, drains)
    }

    /// An engine holding a weightless container through the references asked
    /// for, and nothing else holding it: the engine is its sole owner once
    /// this returns, so a drain can tell whether the release came first.
    private func engineHolding(
        container: Bool = true, draft: Bool = false, batch: Bool = false
    ) async -> (MLXSwiftEngine, ContainerWatch, Drains) {
        let engine = MLXSwiftEngine(promptCache: Self.noColdTier)
        let drains = Drains(initialState: [])
        let weightless = ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(
                    directory: URL(fileURLWithPath: "/nonexistent/weightless", isDirectory: true)),
                model: WeightlessModel(),
                processor: NoInputProcessor(),
                tokenizer: NoTokenizer()))
        let watch = ContainerWatch(weightless)
        await engine.installForTesting(
            container: container ? weightless : nil,
            draft: draft ? weightless : nil,
            batch: batch ? weightless : nil,
            model: model("resident", format: .mlx))
        await engine.setReleaseCachedBuffers { drains.withLock { $0.append(watch.container == nil) } }
        await engine.setMetalKnownUp { false }
        return (engine, watch, drains)
    }

    @Test("unloading an engine that never held a model does not touch MLX's cache")
    func neverLoadedEngineDoesNotDrain() async throws {
        let (engine, drains) = await countingEngine()
        try await engine.unload()
        try await engine.unload()
        let status = await engine.status
        #expect(drains.withLock { $0 } == [])
        #expect(status == .idle)
    }

    @Test("unloading after a load that failed before reaching MLX does not drain")
    func failedLoadThenUnloadDoesNotDrain() async throws {
        let (engine, drains) = await countingEngine()
        await #expect(throws: EngineError.self) {
            try await engine.load(refused())
        }
        #expect(drains.withLock { $0 } == [], "no model has loaded in the process, so nothing may initialize Metal")
        try await engine.unload()
        let status = await engine.status
        #expect(drains.withLock { $0 } == [], "the unload of a never-loaded engine never drains")
        #expect(status == .idle)
    }

    @Test("a load that fails once any model has loaded in the process drains once")
    func failedLoadAfterAModelLoadedInTheProcessDrainsOnce() async throws {
        let (engine, drains) = await countingEngine()
        await engine.setMetalKnownUp { true }
        await #expect(throws: EngineError.self) {
            try await engine.load(refused())
        }
        #expect(drains.withLock { $0 } == [true], "Metal is up, so what the load may have allocated is handed back")
        await #expect(throws: EngineError.self) {
            try await engine.load(missing())
        }
        #expect(drains.withLock { $0 } == [true, true], "through the generic catch as well")
        try await engine.unload()
        #expect(drains.withLock { $0 } == [true, true], "the unload still has nothing to release")
    }

    @Test("dropping the draft model drains, with the main model still resident")
    func droppingTheDraftDrains() async throws {
        let (engine, _, drains) = await engineHolding(container: true, draft: true)
        try await engine.ensureDraftContainer(requestedDraftModelID: nil)
        let hasDraft = await engine.hasDraftContainer
        #expect(!hasDraft)
        #expect(drains.withLock { $0 } == [false], "one drain; the main container is still held")
        try await engine.ensureDraftContainer(requestedDraftModelID: nil)
        #expect(drains.withLock { $0 } == [false], "nothing to release the second time")
    }

    @Test("a draft load that fails releases the old draft first and drains again")
    func failedDraftLoadReleasesTheOldDraftAndDrainsAgain() async throws {
        let (engine, _, drains) = await engineHolding(container: true, draft: true)
        await #expect(throws: EngineError.self) {
            try await engine.ensureDraftContainer(
                requestedDraftModelID: "missing-draft-\(UUID().uuidString)")
        }
        let hasDraft = await engine.hasDraftContainer
        #expect(!hasDraft)
        #expect(drains.withLock { $0 } == [false, false], "the old draft's release, then the failed load's")
    }

    @Test("a draft load that fails with no draft resident drains once, the main model being up")
    func failedDraftLoadWithNoDraftDrainsOnce() async throws {
        let (engine, _, drains) = await engineHolding(container: true)
        await #expect(throws: EngineError.self) {
            try await engine.ensureDraftContainer(
                requestedDraftModelID: "missing-draft-\(UUID().uuidString)")
        }
        #expect(drains.withLock { $0 } == [false])
    }

    @Test("a draft dropped during a counted run drains when the run ends")
    func draftDroppedDuringARunDrainsWhenItEnds() async throws {
        let (engine, _, drains) = await engineHolding(container: true, draft: true)
        await engine.beginGenerationRun()
        try await engine.ensureDraftContainer(requestedDraftModelID: nil)
        #expect(drains.withLock { $0 } == [], "the run may be using the draft")
        await engine.endGenerationRun()
        #expect(drains.withLock { $0 } == [false])
    }

    @Test("unloading a resident model drains once, after the container is released")
    func residentUnloadDrainsOnceAfterTheRelease() async throws {
        let (engine, watch, drains) = await engineHolding()
        try await engine.unload()
        let status = await engine.status
        #expect(drains.withLock { $0 } == [true], "one drain, with the container already gone")
        #expect(watch.container == nil)
        #expect(status == .idle)
    }

    @Test("the batch-serving reference is dropped before the drain, or the drain frees nothing")
    func batchServingReferenceIsDroppedBeforeTheDrain() async throws {
        let (engine, watch, drains) = await engineHolding(container: true, batch: true)
        try await engine.unload()
        #expect(drains.withLock { $0 } == [true])
        #expect(watch.container == nil)
    }

    @Test("a draft left behind without its target still counts as resident")
    func draftAloneCountsAsResident() async throws {
        let (engine, watch, drains) = await engineHolding(container: false, draft: true)
        try await engine.unload()
        #expect(drains.withLock { $0 } == [true])
        #expect(watch.container == nil)
    }

    @Test("a batch-serving copy left behind without its target still counts as resident")
    func batchCopyAloneCountsAsResident() async throws {
        let (engine, watch, drains) = await engineHolding(container: false, batch: true)
        try await engine.unload()
        #expect(drains.withLock { $0 } == [true])
        #expect(watch.container == nil)
    }

    @Test("a load over a resident model releases it first and drains again when the load fails")
    func loadOverResidentReleasesFirstAndDrainsAgainOnFailure() async throws {
        let (engine, watch, drains) = await engineHolding()
        await #expect(throws: EngineError.self) {
            try await engine.load(refused())
        }
        let status = await engine.status
        #expect(
            drains.withLock { $0 } == [true, true],
            "the unload's drain after the release, then the failed load's")
        #expect(watch.container == nil)
        guard case .error = status else {
            Issue.record("a refused load leaves the engine in .error, got \(status)")
            return
        }
    }

    @Test("a load over a resident model that fails in the factory drains again too")
    func loadOverResidentFailingInTheFactoryDrainsAgain() async throws {
        let (engine, watch, drains) = await engineHolding()
        await #expect(throws: EngineError.self) {
            try await engine.load(missing())
        }
        #expect(
            drains.withLock { $0 } == [true, true],
            "the unload's drain after the release, then the failed load's, through the generic catch")
        #expect(watch.container == nil)
    }

    @Test("an unload during one counted run drains when that run ends")
    func unloadDuringOneRunDrainsWhenItEnds() async throws {
        let (engine, _, drains) = await engineHolding()
        await engine.beginGenerationRun()
        try await engine.unload()
        #expect(drains.withLock { $0 } == [])
        await engine.endGenerationRun()
        #expect(drains.withLock { $0 } == [true])
    }

    @Test("a load that fails while the deferred drain is pending leaves it pending")
    func loadDuringADeferredDrainKeepsItPending() async throws {
        let (engine, _, drains) = await engineHolding()
        await engine.beginGenerationRun()
        try await engine.unload()
        await #expect(throws: EngineError.self) {
            try await engine.load(refused())
        }
        #expect(drains.withLock { $0 } == [], "nothing was resident for the load to release, and the run still holds the old model")
        await engine.endGenerationRun()
        #expect(drains.withLock { $0 } == [true], "the deferred drain survives the load")
    }

    @Test("an unload during a generation drains when the last generation ends")
    func unloadDuringAGenerationDrainsWhenItEnds() async throws {
        let (engine, _, drains) = await engineHolding()
        await engine.beginGenerationRun()
        await engine.beginGenerationRun()
        try await engine.unload()
        #expect(drains.withLock { $0 } == [], "the running generations hold the weights")
        await engine.endGenerationRun()
        #expect(drains.withLock { $0 } == [], "one is still running")
        await engine.endGenerationRun()
        #expect(drains.withLock { $0 } == [true], "the last one to finish drains")
        await engine.beginGenerationRun()
        await engine.endGenerationRun()
        #expect(drains.withLock { $0 } == [true], "and only once")
    }
}
