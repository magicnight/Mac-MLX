// Copyright © 2026 macMLX. English comments only.

import Foundation
import MLX
import XCTest
import os

@testable import MacMLXCore

/// Gated, real-weights smoke for the pure-Swift Mellum 2 port (Track G).
///
/// Unlike the numeric-parity suites (tiny synthetic fixtures), this loads the
/// REAL 4-bit checkpoint through the full engine path and end-to-end exercises:
///
///  a. **overlay resolution** — `MLXSwiftEngine.load` runs
///     `ModelOverlay.registerAll()`, so `LLMModelFactory` resolves
///     `config.json`'s `model_type: mellum` to `Mellum2Model`.
///  b. **quantized load + sanitize** — the mixed-precision quantized weights
///     (gate/attention 8-bit, `switch_mlp` 4-bit, per the checkpoint's
///     `quantization` block) load into the stock `Linear`/`SwitchGLU`/
///     `Embedding` layers; the pre-stacked `switch_mlp` short-circuits
///     `sanitize`.
///  c. **mixed cache + generation** — `newCache` builds the per-layer mix
///     (`RotatingKVCache` for sliding layers, `KVCacheSimple` for full), and
///     greedy decode produces coherent, non-empty text. tok/s is printed for
///     the record (not asserted — hardware-dependent).
///
/// GATED — never runs in CI (7.36 GB download). Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_MELLUM_SMOKE=1`, and
///   3. the model directory exists on disk (env `MACMLX_MELLUM_MODEL` overrides
///      the directory name under `~/.mac-mlx/models`; default
///      `Mellum2-12B-A2.5B-Thinking-4bit`).
///
/// Run (once `jedisct1/Mellum2-12B-A2.5B-Thinking-mlx-4bit` is downloaded to
/// `~/.mac-mlx/models/Mellum2-12B-A2.5B-Thinking-4bit`):
///   MACMLX_RUN_MELLUM_SMOKE=1 TEST_RUNNER_MACMLX_RUN_MELLUM_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/Mellum2SmokeTests/testMellum2SmokeGeneratesCoherentText
final class Mellum2SmokeTests: XCTestCase {

    private func modelDirectory(_ name: String) -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(name)", directoryHint: .isDirectory)
    }

    private func localModel(id: String, directory: URL) -> LocalModel {
        LocalModel(
            id: id,
            displayName: id,
            directory: directory,
            sizeBytes: 0,
            format: .mlx,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
    }

    /// Loads the real Mellum2-12B-A2.5B-Thinking-4bit checkpoint and greedy-
    /// decodes a fixed-answer continuation, checking the result is coherent
    /// (the topic-anchor technique the other real-model smokes use).
    ///
    /// Mellum 2 is a THINKING model: the chat template opens a reasoning flow
    /// before the answer, so the budget is generous (256 tokens) and the anchor
    /// is searched over the FULL raw stream (reasoning + answer) rather than a
    /// parsed content field — the project-memory discipline for thinking models.
    func testMellum2SmokeGeneratesCoherentText() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_MELLUM_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_MELLUM_SMOKE=1 to run the Mellum 2 real-weights smoke test")
        }

        let modelID =
            ProcessInfo.processInfo.environment["MACMLX_MELLUM_MODEL"]
            ?? "Mellum2-12B-A2.5B-Thinking-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Mellum 2 model dir not found: \(directory.path)")
        }

        let engine = MLXSwiftEngine()
        try await engine.load(localModel(id: modelID, directory: directory))

        // Fixed-answer prompt: greedy decoding of a competent model must
        // continue the planet sequence with Mars. Same anchor technique the
        // Qwen3.6 / speculative-decoding smokes use.
        let prompt = "Planets in order from the Sun: Mercury, Venus, Earth,"
        let parameters = GenerationParameters(
            temperature: 0, topP: 1.0, maxTokens: 256, stream: true)
        let request = GenerateRequest(
            model: modelID,
            messages: [ChatMessage(role: .user, content: prompt)],
            parameters: parameters
        )

        var text = ""
        var completionTokens: Int?
        let start = Date()
        for try await chunk in engine.generate(request) {
            text += chunk.text
            if let usage = chunk.usage { completionTokens = usage.completionTokens }
        }
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertFalse(text.isEmpty, "Mellum 2 must produce real output, not an early-exit stub")
        XCTAssertTrue(
            text.contains("Mars"),
            "greedy continuation of 'Mercury, Venus, Earth,' must name the next planet "
                + "for output to count as coherent — got: \(text)")

        if let completionTokens, elapsed > 0 {
            let tokPerSec = Double(completionTokens) / elapsed
            print(
                "MELLUM2_SMOKE model=\(modelID) completionTokens=\(completionTokens) "
                    + "elapsed=\(String(format: "%.2f", elapsed))s "
                    + "tokPerSec=\(String(format: "%.1f", tokPerSec))")
        }
    }

    private struct Collected {
        var text = ""
        var finish: FinishReason?
        var promptTokens: Int?
        var completionTokens: Int?
        var cachedPromptTokens: Int?
    }

    private func collect(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> Collected {
        var out = Collected()
        for try await chunk in engine.generate(request) {
            out.text += chunk.text
            if let reason = chunk.finishReason { out.finish = reason }
            if let usage = chunk.usage {
                out.promptTokens = usage.promptTokens
                out.completionTokens = usage.completionTokens
                out.cachedPromptTokens = usage.cachedPromptTokens
            }
        }
        return out
    }

    /// A response that ends on a stop token leaves that token's position in
    /// the working cache (the iterator steps it before the loop sees it), and
    /// the entry is recorded under a key that includes the token (#142). Turn
    /// 1 is a one-word question under a budget far above what the answer
    /// needs, so it ends on `.stop`; the record then has the stop token in its
    /// key, the cache reports as many positions as the key has (Mellum 2's
    /// caches are all attention caches, so they report them), and the token
    /// is in neither the text nor the completion count. Turn 2, the same
    /// conversation with the answer and a follow-up, is served warm and
    /// matches a cold oracle byte for byte. Mellum 2's chat template renders
    /// the previous assistant turn without its reasoning, so the warm turn
    /// reuses turn 1's prompt only; the extension hit through the stop token
    /// is `Gemma4PromptCacheSmokeTests`' case. Turn 2's own record, ending on
    /// its budget, holds exactly its key's positions as well. Both engines
    /// run without the cold tier, so the test neither reads nor clears the
    /// developer's cache.
    func testMellum2EOSStopRecordsTheStopTokenInItsKey() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_MELLUM_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_MELLUM_SMOKE=1 to run the Mellum 2 real-weights smoke test")
        }

        let modelID =
            ProcessInfo.processInfo.environment["MACMLX_MELLUM_MODEL"]
            ?? "Mellum2-12B-A2.5B-Thinking-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Mellum 2 model dir not found: \(directory.path)")
        }

        let noColdTier = PromptCacheConfig(coldEnabled: false)
        let engine = MLXSwiftEngine(promptCache: noColdTier)
        try await engine.load(localModel(id: modelID, directory: directory))
        addTeardownBlock { try await engine.unload() }

        let question = ChatMessage(
            role: .user, content: "Answer with one word: what colour is the sky on a clear day?")
        let turn1 = GenerateRequest(
            model: modelID, messages: [question],
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 900, stream: true))
        let first = try await collect(engine, turn1)
        XCTAssertEqual(
            first.finish, .stop,
            "turn 1 must end on the model's stop token, not on its budget: \(String(describing: first.finish)) after \(first.text.count) characters")
        XCTAssertFalse(
            first.text.contains("<|im_end|>") || first.text.contains("<|endoftext|>"),
            "the stop token is not part of the text: ...\(first.text.suffix(40))")
        let recordOrNil = await engine.lastPromptCacheRecord
        let record = try XCTUnwrap(recordOrNil, "turn 1 went through the prompt cache")
        print(
            "[eos-smoke] turn 1: key=\(record.tokenCount) held=\(record.heldPositions) stop=\(record.endedOnStopToken) "
                + "stored=\(record.recorded) prompt=\(first.promptTokens ?? -1) completion=\(first.completionTokens ?? -1) "
                + "characters=\(first.text.count)")
        XCTAssertTrue(record.endedOnStopToken, "the loop delivered the stop token")
        XCTAssertTrue(record.recorded, "the entry is recorded")
        XCTAssertEqual(
            record.heldPositions, record.tokenCount,
            "the cache holds exactly the key's positions, the stop token's included; if this fails after an "
                + "mlx-swift-lm bump, re-check #142's premise that the iterator steps the stop token before the loop sees it")
        let promptTokens = try XCTUnwrap(first.promptTokens)
        let completionTokens = try XCTUnwrap(first.completionTokens)
        XCTAssertEqual(
            record.tokenCount, promptTokens + completionTokens + 1,
            "the key is the prompt, the tokens that became text, and the stop token")

        let turn2 = GenerateRequest(
            model: modelID,
            messages: [
                question, ChatMessage(role: .assistant, content: first.text),
                ChatMessage(role: .user, content: "And at sunset? One word."),
            ],
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 24, stream: true))
        let warm = try await collect(engine, turn2)
        let secondOrNil = await engine.lastPromptCacheRecord
        let second = try XCTUnwrap(secondOrNil, "turn 2 went through the prompt cache")
        let oracle = MLXSwiftEngine(promptCache: noColdTier)
        try await oracle.load(localModel(id: modelID, directory: directory))
        addTeardownBlock { try await oracle.unload() }
        let cold = try await collect(oracle, turn2)
        print(
            "[eos-smoke] turn 2: warm reused \(warm.cachedPromptTokens ?? 0) of \(record.tokenCount); "
                + "finish=\(String(describing: warm.finish)) key=\(second.tokenCount) held=\(second.heldPositions) "
                + "stop=\(second.endedOnStopToken); warm=\(warm.text.debugDescription) cold=\(cold.text.debugDescription)")
        XCTAssertGreaterThan(warm.cachedPromptTokens ?? 0, 0, "turn 2 is served from the entry turn 1 recorded")
        XCTAssertEqual(warm.text, cold.text, "a turn served from the recorded entry matches the cold oracle")
        // The record of a turn served from a hit and ended on its budget.
        XCTAssertEqual(warm.finish, .length, "turn 2's budget of 24 ends inside the reasoning")
        XCTAssertTrue(second.recorded)
        XCTAssertEqual(
            second.endedOnStopToken, warm.finish == .stop,
            "the record says whether the turn ended on a stop token (a budget stop delivers none)")
        XCTAssertEqual(second.heldPositions, second.tokenCount, "a turn served from a hit records exactly its key's positions too")
    }

    /// Unloading the model must hand its memory back (#136). MLX keeps freed
    /// buffers in its cache, and before the drain an unload of this checkpoint
    /// left 7,021 MB of the 7,035 MB it had used there. Loads it, generates,
    /// loads it again WITHOUT unloading (the engine releases the first copy
    /// before the second allocates, one drain, and the peak stays near one
    /// copy), unloads (a second drain), and reads MLX's counters. Then loads
    /// once more: a drained model's next load allocates its buffers afresh
    /// instead of reusing cached ones, and must still work.
    ///
    /// Same gate and model as the generation smoke above.
    func testMellum2UnloadHandsItsMemoryBack() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_MELLUM_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_MELLUM_SMOKE=1 to run the Mellum 2 real-weights smoke test")
        }

        let modelID =
            ProcessInfo.processInfo.environment["MACMLX_MELLUM_MODEL"]
            ?? "Mellum2-12B-A2.5B-Thinking-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Mellum 2 model dir not found: \(directory.path)")
        }

        let engine = MLXSwiftEngine()
        let drains = OSAllocatedUnfairLock<Int>(initialState: 0)
        await engine.setReleaseCachedBuffers {
            // Counted after the drain, so a read that follows a poll on the
            // count cannot land while the drain is still running.
            EngineMemory.releaseCachedBuffers()
            drains.withLock { $0 += 1 }
        }
        let model = localModel(id: modelID, directory: directory)

        func generate() async throws -> String {
            let request = GenerateRequest(
                model: modelID,
                messages: [ChatMessage(role: .user, content: "Planets in order from the Sun: Mercury, Venus, Earth,")],
                parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 8, stream: true)
            )
            var text = ""
            for try await chunk in engine.generate(request) { text += chunk.text }
            return text
        }

        try await engine.load(model)
        let first = try await generate()
        XCTAssertFalse(first.isEmpty, "the loaded model must generate")
        let resident = Memory.activeMemory
        XCTAssertGreaterThan(
            resident, 256 << 20, "a real checkpoint is resident after its load, got \(resident) B")

        // A load over the resident model releases it first: one drain, a peak
        // near one copy rather than two, and the engine ready on the new load.
        Memory.peakMemory = 0   // the setter resets MLX's peak counter
        try await engine.load(model)
        let reloadPeak = Memory.peakMemory
        let afterReload = await engine.status
        XCTAssertEqual(drains.withLock { $0 }, 1, "a load over a resident model drains once")
        XCTAssertLessThan(
            reloadPeak, resident + resident / 2,
            "a load over a resident model must not hold two copies (peak \(reloadPeak) B against \(resident) B resident)")
        XCTAssertEqual(afterReload, .ready(model: modelID))

        let unloadStart = Date()
        try await engine.unload()
        let unloadSeconds = Date().timeIntervalSince(unloadStart)
        let cached = Memory.cacheMemory
        let active = Memory.activeMemory
        print(
            "[unload-smoke] resident=\(resident) B, reload peak=\(reloadPeak) B, "
                + "unload took \(String(format: "%.3f", unloadSeconds)) s, "
                + "after unload: active=\(active) B cache=\(cached) B")
        XCTAssertEqual(drains.withLock { $0 }, 2, "unload drains once more")
        XCTAssertLessThan(
            cached, 64 << 20,
            "the model's buffers must leave MLX's cache (7 GB stay there without the drain)")
        XCTAssertLessThan(active, resident / 4, "unload must release the weights")

        // The price of the drain: the next load allocates its buffers afresh.
        let start = Date()
        try await engine.load(model)
        let reloadSeconds = Date().timeIntervalSince(start)
        let second = try await generate()
        print("[unload-smoke] reload after drain took \(String(format: "%.2f", reloadSeconds)) s")
        XCTAssertFalse(second.isEmpty, "a reload after the drain must generate")
        try await engine.unload()
    }

    /// An unload that lands while a response is streaming cannot free the
    /// weights: the generation holds the container until it ends. The drain
    /// is deferred to that moment (#136). Starts a stream, unloads after the
    /// first token, and checks that nothing was drained yet, that the stream
    /// still finishes, and that the drain then runs and empties MLX's cache.
    func testMellum2UnloadDuringAStreamDrainsWhenItEnds() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_MELLUM_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_MELLUM_SMOKE=1 to run the Mellum 2 real-weights smoke test")
        }

        let modelID =
            ProcessInfo.processInfo.environment["MACMLX_MELLUM_MODEL"]
            ?? "Mellum2-12B-A2.5B-Thinking-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Mellum 2 model dir not found: \(directory.path)")
        }

        let engine = MLXSwiftEngine()
        let drains = OSAllocatedUnfairLock<Int>(initialState: 0)
        await engine.setReleaseCachedBuffers {
            // Counted after the drain, so a read that follows a poll on the
            // count cannot land while the drain is still running.
            EngineMemory.releaseCachedBuffers()
            drains.withLock { $0 += 1 }
        }
        try await engine.load(localModel(id: modelID, directory: directory))
        let resident = Memory.activeMemory

        let request = GenerateRequest(
            model: modelID,
            messages: [ChatMessage(role: .user, content: "Count from one to thirty, one number per line.")],
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 48, stream: true)
        )
        let chunks = OSAllocatedUnfairLock<Int>(initialState: 0)
        let stream = Task {
            var text = ""
            for try await chunk in engine.generate(request) {
                text += chunk.text
                chunks.withLock { $0 += 1 }
            }
            return text
        }
        // Wait for the stream to be under way (polling, never a fixed sleep).
        let firstChunkDeadline = Date().addingTimeInterval(60)
        while chunks.withLock({ $0 }) == 0, Date() < firstChunkDeadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertGreaterThan(chunks.withLock { $0 }, 0, "the stream must have started")

        try await engine.unload()
        XCTAssertEqual(drains.withLock { $0 }, 0, "nothing can be drained while the stream holds the weights")

        let text = try await stream.value
        XCTAssertFalse(text.isEmpty, "the stream finishes on the container it captured")
        let drainDeadline = Date().addingTimeInterval(60)
        while drains.withLock({ $0 }) == 0, Date() < drainDeadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertEqual(drains.withLock { $0 }, 1, "the last generation to finish drains")
        // The stream ended, so the worker has let go: the weights are no
        // longer active, and the drain that followed left the cache empty.
        let active = Memory.activeMemory
        let cached = Memory.cacheMemory
        print("[unload-smoke] after a deferred drain: active=\(active) B cache=\(cached) B")
        XCTAssertLessThan(active, resident / 4, "the weights are no longer held: \(active) B active")
        XCTAssertLessThan(cached, 64 << 20, "the deferred drain must empty MLX's cache")
    }

    /// The cancelled twin of the test above: after the unload, the stream is
    /// cancelled instead of left to finish. mlx-swift-lm's worker checks for
    /// cancellation only between decode steps and lets go of the model after
    /// it synchronizes the GPU stream, so the engine must wait for the worker
    /// before it counts the run as over, or the drain runs a step too early
    /// and the weights land in MLX's cache after it (#136).
    func testMellum2UnloadDuringACancelledStreamDrainsAfterTheWorkerLetsGo() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_MELLUM_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_MELLUM_SMOKE=1 to run the Mellum 2 real-weights smoke test")
        }

        let modelID =
            ProcessInfo.processInfo.environment["MACMLX_MELLUM_MODEL"]
            ?? "Mellum2-12B-A2.5B-Thinking-4bit"
        let directory = modelDirectory(modelID)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Mellum 2 model dir not found: \(directory.path)")
        }

        let engine = MLXSwiftEngine()
        let drains = OSAllocatedUnfairLock<Int>(initialState: 0)
        await engine.setReleaseCachedBuffers {
            // Counted after the drain, so a read that follows a poll on the
            // count cannot land while the drain is still running.
            EngineMemory.releaseCachedBuffers()
            drains.withLock { $0 += 1 }
        }
        try await engine.load(localModel(id: modelID, directory: directory))
        let resident = Memory.activeMemory

        let request = GenerateRequest(
            model: modelID,
            messages: [ChatMessage(role: .user, content: "Count from one to two hundred, one number per line.")],
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 400, stream: true)
        )
        let chunks = OSAllocatedUnfairLock<Int>(initialState: 0)
        let stream = Task {
            var text = ""
            for try await chunk in engine.generate(request) {
                text += chunk.text
                chunks.withLock { $0 += 1 }
            }
            return text
        }
        let firstChunkDeadline = Date().addingTimeInterval(60)
        while chunks.withLock({ $0 }) == 0, Date() < firstChunkDeadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertGreaterThan(chunks.withLock { $0 }, 0, "the stream must have started")

        try await engine.unload()
        XCTAssertEqual(drains.withLock { $0 }, 0, "nothing can be drained while the stream holds the weights")
        stream.cancel()
        _ = await stream.result

        let drainDeadline = Date().addingTimeInterval(60)
        while drains.withLock({ $0 }) == 0, Date() < drainDeadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertEqual(drains.withLock { $0 }, 1, "the run ends, and drains, once the worker has let go")
        // The worker had stepped tokens nobody received, so the cache no
        // longer matched the key a cancelled run would have recorded under.
        let recorded = await engine.promptCacheResidentCount
        XCTAssertEqual(recorded, 0, "a cancelled response records no prompt cache entry")
        // The worker must let go of the weights (must happen: polled on
        // active memory), and the drain must have come after it: a drain
        // that ran first leaves the cache empty for a moment while the
        // weights are still active, then receives them for good once the
        // worker finishes, which the cache check below would then see.
        let releaseDeadline = Date().addingTimeInterval(60)
        while Memory.activeMemory >= resident / 4, Date() < releaseDeadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        let active = Memory.activeMemory
        let cached = Memory.cacheMemory
        print("[unload-smoke] after a cancelled stream's deferred drain: active=\(active) B cache=\(cached) B")
        XCTAssertLessThan(active, resident / 4, "the worker let go of the weights: \(active) B still active")
        XCTAssertLessThan(cached, 64 << 20, "the drain ran after the worker let go of the weights: \(cached) B in MLX's cache")
        XCTAssertLessThan(
            chunks.withLock { $0 }, 100,
            "the consumer received few chunks before its cancel (what the worker did afterwards is what the wait above checks)")
    }
}
