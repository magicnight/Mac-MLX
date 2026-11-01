// Copyright © 2026 macMLX. English comments only.

import AppKit
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MacMLXCore
// MLXLMCommon has a `JSONValue` of its own; the scoped import makes ours win
// (see the same note in PromptCacheStoreTests).
import enum MacMLXCore.JSONValue

/// Gated, real-weights image gates a family passes before it joins
/// ``FullPathVLM/servedFamilies`` (gates 4 and 5 of
/// `docs/superpowers/specs/2026-10-31-gemma4-qwen35-vision-design.md`; the
/// text gates are in `FullPathVLMParitySmokeTests`). The load is the full
/// path, through the forcing seam.
///
/// Gate 4, over `POST /v1/chat/completions` on the real `HummingbirdServer`:
/// - an image of rendered text is read back;
/// - the same image with a tool returns a parsed call carrying that text;
/// - with replayed reasoning and a tool result earlier in the conversation,
///   the image is still read (the reasoning render passes the images on);
/// - an image request records nothing, so a text request after it with the
///   same prefix reuses nothing;
/// - and, on the engine, image cache step 1: an image request after a long
///   text turn, warm against cold and each against a one-shot prefill, has
///   |Δ log p| within gate 2's bound, KL below δ²/2 and the same top token
///   on its first token. Compared on the top 10 logprobs the engine reports
///   (its `top_logprobs` cap), |Δ log p| only above a log p of −10
///   (``stepOneLogProbFloor``), with δ from the cold prefill's top-1 logit
///   and the noise from the same request prefilled in 128-token chunks. A
///   checkpoint whose image requests run cold (Gemma 4 with bidirectional
///   vision attention) must reuse nothing instead;
/// - also on the engine: the image is read with `kv_bits` 8 and 4, and a
///   cancel during a long image prefill streams nothing and leaves the
///   entry it reused intact, and the next image request is read (and
///   reuses that entry, unless the checkpoint's image requests run cold).
///
/// Gate 5: five turns of an agent conversation that reaches about 16k tokens,
/// with an image in turn 1 and without it, recording each turn's prompt,
/// reuse, time to the first token and peak memory. Every turn of both must
/// complete; the numbers are printed for the record.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_FULL_PATH_IMAGE_GATE=1`, and
///   3. the model directory exists (env `MACMLX_FULL_PATH_IMAGE_MODEL`
///      overrides the directory name under `~/.mac-mlx/models`; default
///      `gemma-4-e4b-it-8bit`).
///
/// `MACMLX_FULL_PATH_COLD_TIER=1` gives the engines a cold tier, as in
/// `FullPathVLMParitySmokeTests`.
///
/// Run:
///   MACMLX_RUN_FULL_PATH_IMAGE_GATE=1 TEST_RUNNER_MACMLX_RUN_FULL_PATH_IMAGE_GATE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/FullPathVLMImageGateSmokeTests
final class FullPathVLMImageGateSmokeTests: XCTestCase {

    private static let anchor = "MACMLX 2026"
    private static let readQuestion = "What text is written in this image? Output only the text."

    /// Room for a thinking template (Qwen3.x thinks by default; chat
    /// completions take no per-request switch) to reach its answer, which
    /// `content` carries apart from the reasoning.
    private static let answerTokens = 1024

    // MARK: - Gate 4 over the API

    func testImagesWorkOnTheChatCompletionsPath() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let image = try Self.renderTextImage(Self.anchor)
        defer { try? FileManager.default.removeItem(at: image) }
        let imagePart = try Self.imagePart(image)
        let engine = try await fullPathEngine(modelID, directory)
        let server = HummingbirdServer(engine: engine)
        let port = try await server.start(preferredPort: 22_640)
        addTeardownBlock { await server.stop() }

        // Read the image.
        let read = try await post(port, [
            "model": modelID, "temperature": 0, "max_tokens": Self.answerTokens,
            "messages": [["role": "user", "content": [["type": "text", "text": Self.readQuestion], imagePart]]],
        ])
        let readText = try Self.content(read)
        print("[image-gate] \(modelID) read: \(readText.debugDescription)")
        XCTAssertTrue(Self.reads(readText), "gate 4: the image is not read: \(readText)")

        // The same image with a tool.
        let called = try await post(port, [
            "model": modelID, "temperature": 0, "max_tokens": Self.answerTokens,
            "tools": [Self.recordTextTool],
            "messages": [["role": "user", "content": [
                ["type": "text", "text": "Read the text in this image and pass it to the record_text tool."],
                imagePart,
            ]]],
        ])
        let call = try Self.firstToolCall(called)
        print("[image-gate] \(modelID) tool call: \(call.name) \(call.arguments)")
        XCTAssertEqual(call.name, "record_text", "gate 4: no parsed tool call")
        XCTAssertTrue(Self.reads(call.arguments), "gate 4: the tool call does not carry the image's text")

        // Replayed reasoning and a tool result before the image.
        let history = try await post(port, [
            "model": modelID, "temperature": 0, "max_tokens": Self.answerTokens,
            "tools": [Self.bashToolJSON],
            "messages": [
                ["role": "user", "content": "How many files are in Sources?"],
                [
                    "role": "assistant", "content": "",
                    "reasoning_content": "I should list Sources before answering.",
                    "tool_calls": [[
                        "id": "call_1", "type": "function",
                        "function": ["name": "Bash", "arguments": #"{"command": "ls Sources"}"#],
                    ]],
                ],
                ["role": "tool", "tool_call_id": "call_1", "content": "a.swift\nb.swift\nc.swift"],
                [
                    "role": "assistant", "content": "Sources has three files.",
                    "reasoning_content": "The listing shows three files.",
                ],
                ["role": "user", "content": [["type": "text", "text": Self.readQuestion], imagePart]],
            ],
        ])
        let historyText = try Self.content(history)
        print("[image-gate] \(modelID) after reasoning and a tool result: \(historyText.debugDescription)")
        XCTAssertTrue(Self.reads(historyText), "gate 4: the image is lost behind replayed reasoning: \(historyText)")

        // An image request records nothing for a text request to find.
        await engine.clearPromptCache()
        let system = Self.systemPrompt()
        _ = try await post(port, [
            "model": modelID, "temperature": 0, "max_tokens": 4,
            "messages": [
                ["role": "system", "content": system],
                ["role": "user", "content": [["type": "text", "text": Self.readQuestion], imagePart]],
            ],
        ])
        let entriesAfterImage = await engine.promptCacheResidentCount + engine.promptCacheColdCount
        let text = try await post(port, [
            "model": modelID, "temperature": 0, "max_tokens": 4,
            "messages": [["role": "system", "content": system], ["role": "user", "content": Self.readQuestion]],
        ])
        let cached = Self.cachedTokens(text)
        print("[image-gate] \(modelID) text after image: entries=\(entriesAfterImage) cached=\(cached ?? -1)")
        XCTAssertEqual(entriesAfterImage, 0, "gate 4: the image request left a cache entry")
        XCTAssertEqual(cached, 0, "gate 4: a text request reused an image request's KV")
    }

    // MARK: - Gate 4, step 1 on logits

    func testAWarmImageRequestPredictsAsAColdOneDoes() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let image = try Self.renderTextImage(Self.anchor)
        defer { try? FileManager.default.removeItem(at: image) }
        let engine = try await fullPathEngine(modelID, directory)

        var parameters = GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 1, stream: true)
        parameters.logprobs = true
        parameters.topLogprobs = 10
        let question = ChatMessage(role: .user, content: "I will send you an image next. Are you ready?")
        let turn1 = GenerateRequest(
            model: modelID, messages: [question], systemPrompt: Self.systemPrompt(), parameters: parameters)
        let turn2 = GenerateRequest(
            model: modelID,
            messages: [
                question,
                ChatMessage(role: .assistant, content: "Yes, send it."),
                ChatMessage(role: .user, content: Self.readQuestion,
                            images: [ImageAttachment(fileURL: image, mimeType: "image/png")]),
            ],
            systemPrompt: Self.systemPrompt(), parameters: parameters)

        await engine.clearPromptCache()
        _ = try await run(engine, turn1)
        let warm = try await run(engine, turn2)
        await engine.clearPromptCache()
        let cold = try await run(engine, turn2)
        let coldLogits = try await engine.coldNextTokenLogits(turn2)
        let top1 = try XCTUnwrap(coldLogits.logits.max())
        let delta = 4 * FullPathVLMParitySmokeTests.LogitComparison.bf16ULP(top1)
        // Gate 2's bound: δ, or this load's own difference on the same image
        // request prefilled in other chunks, with the margin, if larger.
        let rechunked = try await engine.coldNextTokenLogits(
            turn2, prefill: FullPathVLMParitySmokeTests.noiseFloorPrefill)
        let noise = try XCTUnwrap(FullPathVLMParitySmokeTests.LogitComparison(
            reference: coldLogits.logits, candidate: rechunked.logits, top: 10))
        let bound = FullPathVLMParitySmokeTests.logProbBound(delta: delta, noise: noise.maxLogProbDelta)

        let warmTop = try XCTUnwrap(warm.logprobs.first?.topLogprobs)
        let coldTop = try XCTUnwrap(cold.logprobs.first?.topLogprobs)
        let warmByToken = Dictionary(warmTop.map { ($0.token, Double($0.logprob)) }, uniquingKeysWith: max)
        var maxDelta = 0.0
        var missing: [String] = []
        for alternative in coldTop {
            guard let other = warmByToken[alternative.token] else { missing.append(alternative.token); continue }
            guard Double(alternative.logprob) > Self.stepOneLogProbFloor else { continue }
            maxDelta = max(maxDelta, abs(Double(alternative.logprob) - other))
        }
        let shared = coldTop.compactMap { a in warmByToken[a.token].map { (Double(a.logprob), $0) } }
        let pNorm = FullPathVLMParitySmokeTests.LogitComparison.logSumExp(shared.map(\.0))
        let qNorm = FullPathVLMParitySmokeTests.LogitComparison.logSumExp(shared.map(\.1))
        let kl = shared.reduce(0.0) { sum, pair in
            let lp = pair.0 - pNorm
            return sum + exp(lp) * (lp - (pair.1 - qNorm))
        }
        print("[image-gate] \(modelID) step 1: reused=\(warm.cachedPromptTokens ?? 0) "
            + String(format: "δ=%.4f max|Δlogp|=%.5f KL=%.2e (bound %.2e) ", delta, maxDelta, kl, delta * delta / 2)
            + String(format: "rechunked max|Δlogp|=%.5f |Δ log p| bound %.5f ", noise.maxLogProbDelta, bound)
            + "missing=\(missing) warm=\(warm.text.debugDescription) cold=\(cold.text.debugDescription)")
        XCTAssertEqual(cold.cachedPromptTokens ?? 0, 0, "the reference run is cold")
        guard try Self.reusesPrefixBeforeMedia(directory) else {
            // Bidirectional vision attention: image requests run cold by
            // design, so the warm run is a second cold one.
            XCTAssertEqual(warm.cachedPromptTokens ?? 0, 0, "step 1: an image request reused the cache")
            return
        }
        XCTAssertGreaterThan(warm.cachedPromptTokens ?? 0, 0, "step 1: the warm image request reused nothing")
        XCTAssertTrue(missing.isEmpty, "gate 4: the warm run's top 10 lacks \(missing)")
        XCTAssertLessThanOrEqual(maxDelta, bound, "gate 4: warm against cold, |Δ log p|")
        XCTAssertLessThan(kl, delta * delta / 2, "gate 4: warm against cold, KL")

        // Both runs above prepare the block on a cached prefix: the warm one
        // on its hit, the cold one after prefilling the text before the
        // block itself (`FullPathVLM.textPrefillBeforeMedia`), and both
        // split after it (`FullPathVLM.mediaPrefillSplit`). The logits read
        // prefills the whole prompt in one `prepare`, the reference that
        // catches a block whose attention a cached prefix changes. Each run
        // against it over its top 10, by token text.
        let loaded = try await HuggingFaceTokenizerLoader().load(from: directory)
        let tokenizer = try XCTUnwrap(loaded as? TokenizerBridge)
        let oneShot = FullPathVLMParitySmokeTests.LogitComparison.logSoftmax(coldLogits.logits)
        let oneShotTop = oneShot.indices.sorted { oneShot[$0] > oneShot[$1] }.prefix(10)
        let oneShotFirst = try XCTUnwrap(oneShotTop.first.map {
            tokenizer.decode(tokenIds: [$0], skipSpecialTokens: false)
        })
        for (label, top) in [("cold", coldTop), ("warm", warmTop)] {
            let byToken = Dictionary(top.map { ($0.token, Double($0.logprob)) }, uniquingKeysWith: max)
            var splitDelta = 0.0
            var splitMissing: [String] = []
            var shared: [(oneShot: Double, split: Double)] = []
            for id in oneShotTop {
                let text = tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
                guard let split = byToken[text] else { splitMissing.append(text); continue }
                shared.append((oneShot[id], split))
                guard oneShot[id] > Self.stepOneLogProbFloor else { continue }
                splitDelta = max(splitDelta, abs(split - oneShot[id]))
            }
            let pNorm = FullPathVLMParitySmokeTests.LogitComparison.logSumExp(shared.map(\.oneShot))
            let qNorm = FullPathVLMParitySmokeTests.LogitComparison.logSumExp(shared.map(\.split))
            let splitKL = shared.reduce(0.0) { sum, pair in
                let lp = pair.oneShot - pNorm
                return sum + exp(lp) * (lp - (pair.split - qNorm))
            }
            print("[image-gate] \(modelID) \(label) run against one shot: "
                + String(format: "max|Δlogp|=%.5f KL=%.2e ", splitDelta, splitKL)
                + "top=\(top.first?.token.debugDescription ?? "-") missing=\(splitMissing)")
            // The tenth place can swap with the eleventh between two prefills.
            XCTAssertTrue(splitMissing.count <= 1, "gate 4: the \(label) run's top 10 lacks \(splitMissing)")
            XCTAssertLessThanOrEqual(splitDelta, bound, "gate 4: \(label) against one-shot prefill, |Δ log p|")
            XCTAssertLessThan(splitKL, delta * delta / 2, "gate 4: \(label) against one-shot prefill, KL")
            XCTAssertEqual(top.first?.token, oneShotFirst, "gate 4: \(label) against one-shot prefill, top token")
        }
    }

    // MARK: - Gate 4, on the engine

    /// `kv_bits` with an image: the split head prefills into a plain cache,
    /// and the quantization plan the tail's iterator applies after its
    /// prefill then covers every full-attention layer's whole length, head
    /// included (sliding-window layers are not quantized). The answer must
    /// still read the image. Whether the cache was quantized is not
    /// observable here.
    func testAnImageIsReadWithAQuantizedKVCache() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let image = try Self.renderTextImage(Self.anchor)
        defer { try? FileManager.default.removeItem(at: image) }
        let engine = try await fullPathEngine(modelID, directory)

        for kvBits in [nil, 8, 4] as [Int?] {
            let request = Self.readRequest(
                modelID, image: image,
                parameters: GenerationParameters(
                    temperature: 0, topP: 1.0, maxTokens: Self.answerTokens, stream: true,
                    kvBits: kvBits, quantizedKVStart: kvBits == nil ? nil : 0))
            let result = try await run(engine, request)
            let answer = Self.answer(result.text)
            let label = kvBits.map(String.init) ?? "none"
            print("[image-gate] \(modelID) kv_bits=\(label): \(answer.debugDescription)")
            XCTAssertNotNil(result.promptTokens, "kv_bits=\(label): the request did not complete")
            XCTAssertTrue(Self.reads(answer), "kv_bits=\(label): the image is not read: \(answer)")
        }
    }

    /// A cancel that lands while a long image request prefills: the stream
    /// ends without a token, and the engine serves the next requests as
    /// before. The request reuses a text turn's entry up to its image, then
    /// prefills the image in one shot (not interruptible) and the long text
    /// after it in chunks; upstream stops those at the next chunk once
    /// cancelled (`PrefillParameters.forEachChunk` checks). A cache hit is a
    /// copy, so the stopped prefill leaves the entry it reused intact: the
    /// text turn is reused and answered the same afterwards. The text turn's
    /// wall time is printed: no more than its cold run if the prefill
    /// stopped, the rest of a 22k-token prefill if it ran on. A cancel that
    /// lands before the cache lookup passes too, without testing the
    /// prefill; it landed after it here (removing the copy on a hit fails
    /// this test).
    func testCancellingDuringAnImagePrefillLeavesTheEngineServing() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let image = try Self.renderTextImage(Self.anchor)
        defer { try? FileManager.default.removeItem(at: image) }
        let engine = try await fullPathEngine(modelID, directory)

        // The text turn records a snapshot of its conversation, which the
        // image request, a later turn of it, starts with.
        let text = Self.textTurn(modelID)
        let firstStart = Date()
        let before = try await run(engine, text)
        let beforeSeconds = Date().timeIntervalSince(firstStart)
        let entries = await engine.promptCacheResidentCount + engine.promptCacheColdCount
        XCTAssertGreaterThan(entries, 0, "the text turn recorded nothing")

        let long = Self.imageTurnAfter(text, image: image)
        let cancelled = Task { () -> (streamed: String, completed: Bool) in
            var streamed = ""
            var completed = false
            do {
                for try await chunk in engine.generate(long) {
                    streamed += chunk.text
                    if chunk.usage != nil { completed = true }
                }
            } catch {}
            return (streamed, completed)
        }
        // About 22k tokens after the image: one second in, the request is
        // still before its first token.
        try await Task.sleep(for: .seconds(1))
        cancelled.cancel()
        let outcome = await cancelled.value
        print("[image-gate] \(modelID) cancelled in prefill: text=\(outcome.streamed.debugDescription) "
            + "completed=\(outcome.completed)")
        XCTAssertEqual(outcome.streamed, "", "a cancelled image request streamed text")
        XCTAssertFalse(outcome.completed, "a cancelled image request reported usage")

        // The text turn waits until the stopped prefill lets go of the model,
        // then finds its entry as it left it.
        let start = Date()
        let after = try await run(engine, text)
        print("[image-gate] \(modelID) text turn after the cancel: "
            + String(format: "%.2f s (%.2f s before) ", Date().timeIntervalSince(start), beforeSeconds)
            + "reused=\(after.cachedPromptTokens ?? -1) same text: \(after.text == before.text)")
        XCTAssertGreaterThan(after.cachedPromptTokens ?? 0, 0, "after a cancel, the text turn reused nothing")
        XCTAssertEqual(after.text, before.text, "after a cancel, the text turn answers differently")

        // The same image request, run through, reuses the text turn (so the
        // cancelled one did too) and reads the image. A checkpoint whose image
        // requests run cold reuses nothing here, by design.
        var whole = long
        whole.parameters.maxTokens = 1
        let served = try await run(engine, whole)
        var read = Self.imageTurnAfter(text, image: image, rounds: 0)
        read.parameters.maxTokens = Self.answerTokens
        let answer = Self.answer(try await run(engine, read).text)
        print("[image-gate] \(modelID) image turn after the cancel: prompt=\(served.promptTokens ?? -1) "
            + "reused=\(served.cachedPromptTokens ?? -1) read: \(answer.debugDescription)")
        if try Self.reusesPrefixBeforeMedia(directory) {
            XCTAssertGreaterThan(served.cachedPromptTokens ?? 0, 0, "the image turn did not reuse the text turn")
        } else {
            XCTAssertEqual(served.cachedPromptTokens ?? 0, 0, "an image request reused the cache")
        }
        XCTAssertTrue(Self.reads(answer), "after a cancel, the image is not read: \(answer)")
    }

    /// Whether the checkpoint's image requests reuse the cache before their
    /// first media block (off for Gemma 4 with bidirectional vision
    /// attention, which runs them cold).
    private static func reusesPrefixBeforeMedia(_ directory: URL) throws -> Bool {
        let configURL = directory.appending(path: "config.json")
        return try XCTUnwrap(FullPathVLM.resolve(configURL: configURL, forced: true)).reusesPrefixBeforeMedia
    }

    /// A first turn: a question, with the session's system prompt and tool.
    private static func textTurn(_ modelID: String) -> GenerateRequest {
        GenerateRequest(
            model: modelID,
            messages: [ChatMessage(role: .user, content: "I will send you a screenshot of the build next.")],
            systemPrompt: systemPrompt(),
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 16, stream: true),
            tools: [bashTool])
    }

    /// `text`'s conversation continued: an answer, the image with the read
    /// question, then `rounds` tool calls with long results.
    private static func imageTurnAfter(_ text: GenerateRequest, image: URL, rounds: Int = 4) -> GenerateRequest {
        var request = text
        request.messages += [
            ChatMessage(role: .assistant, content: "Send it."),
            ChatMessage(
                role: .user, content: readQuestion,
                images: [ImageAttachment(fileURL: image, mimeType: "image/png")]),
        ]
        for round in 0..<rounds {
            request.messages.append(ChatMessage(
                role: .assistant, content: "",
                toolCalls: [ToolCallRequest(
                    id: "call_\(round)", name: "Bash", arguments: ["command": .string("ls -la Module\(round)")])]))
            request.messages.append(ChatMessage(
                role: .tool, content: listing("Module\(round)", count: 140), toolCallID: "call_\(round)"))
        }
        request.parameters.maxTokens = 256
        return request
    }

    private static func readRequest(
        _ modelID: String, image: URL, parameters: GenerationParameters
    ) -> GenerateRequest {
        GenerateRequest(
            model: modelID,
            messages: [ChatMessage(
                role: .user, content: readQuestion,
                images: [ImageAttachment(fileURL: image, mimeType: "image/png")])],
            parameters: parameters)
    }

    /// The text after a thinking template's reasoning, which the engine
    /// streams inline (the API splits it into `content`).
    private static func answer(_ text: String) -> String {
        guard let close = text.range(of: "</think>", options: .backwards) else { return text }
        return String(text[close.upperBound...])
    }

    // MARK: - Gate 5

    func testAFiveTurnAgentSessionWithAnImageCompletes() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let image = try Self.renderTextImage(Self.anchor)
        defer { try? FileManager.default.removeItem(at: image) }
        let engine = try await fullPathEngine(modelID, directory)

        for withImage in [false, true] {
            await engine.clearPromptCache()
            let label = withImage ? "with image" : "without image"
            for turn in 1...5 {
                let request = Self.sessionRequest(modelID, turn: turn, image: withImage ? image : nil)
                Memory.peakMemory = 0
                let start = Date()
                let result = try await run(engine, request)
                let seconds = Date().timeIntervalSince(start)
                print("[image-gate] \(modelID) session \(label) turn \(turn): prompt=\(result.promptTokens ?? -1) "
                    + "reused=\(result.cachedPromptTokens ?? 0) "
                    + String(format: "first token %.2f s ", result.firstTokenSeconds ?? seconds)
                    + "peak=\(Memory.peakMemory >> 20) MiB")
                XCTAssertNotNil(result.promptTokens, "gate 5: \(label) turn \(turn) did not complete")
            }
        }
    }

    /// Gate 5's other end: four text turns of the same session, then an
    /// image in a fifth. The text before the image prefills in chunks, as the
    /// turns before it did, so the image turn's peak stays within
    /// ``lateImagePeakAllowance`` of the text turn's before it. On a
    /// checkpoint whose image requests run cold the whole history up to the
    /// image prefills in one shot instead (88 GB at this size on Gemma 4
    /// 31B); there the turn must only complete.
    func testAnImageLateInALongSessionCompletes() async throws {
        let (modelID, directory) = try gateAndResolveModel()
        let image = try Self.renderTextImage(Self.anchor)
        defer { try? FileManager.default.removeItem(at: image) }
        let engine = try await fullPathEngine(modelID, directory)
        var textPeak = 0
        for turn in 1...5 {
            var request = Self.sessionRequest(modelID, turn: turn, image: nil)
            if turn == 5 {
                request.messages.append(ChatMessage(
                    role: .user, content: "Here is the screenshot.",
                    images: [ImageAttachment(fileURL: image, mimeType: "image/png")]))
            }
            Memory.peakMemory = 0
            let start = Date()
            let result = try await run(engine, request)
            let seconds = Date().timeIntervalSince(start)
            print("[image-gate] \(modelID) late image turn \(turn)\(turn == 5 ? " (image)" : ""): "
                + "prompt=\(result.promptTokens ?? -1) reused=\(result.cachedPromptTokens ?? 0) "
                + String(format: "first token %.2f s ", result.firstTokenSeconds ?? seconds)
                + "peak=\(Memory.peakMemory >> 20) MiB")
            XCTAssertNotNil(result.promptTokens, "gate 5: late-image turn \(turn) did not complete")
            guard turn == 5 else {
                textPeak = Memory.peakMemory
                continue
            }
            if try Self.reusesPrefixBeforeMedia(directory) {
                XCTAssertLessThanOrEqual(
                    Memory.peakMemory - textPeak, Self.lateImagePeakAllowance,
                    "gate 5: the late image turn's peak is far above the text turn's")
            }
        }
    }

    /// Step 1 compares |Δ log p| only on tokens whose reference log p is
    /// above this (a probability above about 4.5e-5); KL and the top token
    /// cover the tail below it. With the 3840×960 fixture Gemma 4 E4B puts
    /// nearly all mass on its first token, and the next ones sit near
    /// −10 to −20, where two prefills of the same prompt in other chunks move
    /// log p by 0.6 (rank 3, −10.78 against −10.18) at a KL of 5e-6.
    private static let stepOneLogProbFloor = -10.0

    /// How far above the text turn before it a late image turn may peak: the
    /// image block's activations and the KV of the turn's new tokens, not a
    /// one-shot prefill of the history.
    private static let lateImagePeakAllowance = 1 << 30

    // MARK: - Session

    /// The turns of an agent session: a question (with the image when there
    /// is one), then four rounds of a tool call and a long tool result, so
    /// turn 5 is about 16k tokens.
    private static func sessionRequest(_ modelID: String, turn: Int, image: URL?) -> GenerateRequest {
        var messages = [
            ChatMessage(
                role: .user, content: "Here is a screenshot of the build. Find the Swift files it mentions.",
                images: image.map { [ImageAttachment(fileURL: $0, mimeType: "image/png")] } ?? []),
        ]
        for round in 1..<turn {
            messages.append(ChatMessage(
                role: .assistant, content: "",
                toolCalls: [ToolCallRequest(
                    id: "call_\(round)", name: "Bash", arguments: ["command": .string("ls -la Module\(round)")])]))
            messages.append(ChatMessage(role: .tool, content: listing("Module\(round)", count: 140), toolCallID: "call_\(round)"))
        }
        return GenerateRequest(
            model: modelID, messages: messages, systemPrompt: systemPrompt(),
            parameters: GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 1, stream: true),
            tools: [bashTool])
    }

    private static func listing(_ directory: String, count: Int) -> String {
        (1...count).map { n in
            "-rw-r--r--  1 kevin  staff  \(1_000 + n * 37) Oct 30 12:\(String(format: "%02d", n % 60)) "
                + "\(directory)/Source\(n).swift"
        }.joined(separator: "\n")
    }

    private static let rules = [
        "You are a careful assistant that reads documents and images for the user.",
        "Answer only what the user asks, in as few words as the answer needs.",
        "When the user sends an image, look at the whole image before you answer.",
        "Copy text from an image exactly as it is written, keeping its case and digits.",
        "Never guess at text you cannot read; say that it is unreadable instead.",
        "Do not describe colors, fonts or layout unless the user asks for them.",
        "Keep each answer on a single line.",
        "If a question has no answer in the image, say so plainly.",
    ]

    /// About 1,600 tokens: past the snapshot minimum and Gemma 4's 512-token
    /// sliding window.
    private static func systemPrompt() -> String {
        (1...10).map { section in
            "Section \(section).\n" + rules.map { "- \($0)" }.joined(separator: "\n")
        }.joined(separator: "\n\n")
    }

    private static let bashTool: JSONValue = .object([
        "type": .string("function"),
        "function": .object([
            "name": .string("Bash"),
            "description": .string("Run a shell command and return its output."),
            "parameters": .object([
                "type": .string("object"),
                "properties": .object([
                    "command": .object(["type": .string("string"), "description": .string("The command to run.")])
                ]),
                "required": .array([.string("command")]),
            ]),
        ]),
    ])

    private static var bashToolJSON: [String: Any] { [
        "type": "function",
        "function": [
            "name": "Bash",
            "description": "Run a shell command and return its output.",
            "parameters": [
                "type": "object",
                "properties": ["command": ["type": "string", "description": "The command to run."]],
                "required": ["command"],
            ],
        ],
    ] }

    private static var recordTextTool: [String: Any] { [
        "type": "function",
        "function": [
            "name": "record_text",
            "description": "Record the text read from an image.",
            "parameters": [
                "type": "object",
                "properties": ["text": ["type": "string", "description": "The text, exactly as written."]],
                "required": ["text"],
            ],
        ],
    ] }

    // MARK: - Helpers

    private struct Run {
        var text = ""
        var promptTokens: Int?
        var cachedPromptTokens: Int?
        var logprobs: [TokenLogprob] = []
        var firstTokenSeconds: Double?
    }

    private func run(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> Run {
        var out = Run()
        let start = Date()
        for try await chunk in engine.generate(request) {
            if out.firstTokenSeconds == nil, !chunk.text.isEmpty {
                out.firstTokenSeconds = Date().timeIntervalSince(start)
            }
            out.text += chunk.text
            out.logprobs += chunk.logprobs ?? []
            if let usage = chunk.usage {
                out.promptTokens = usage.promptTokens
                out.cachedPromptTokens = usage.cachedPromptTokens
            }
        }
        return out
    }

    private func gateAndResolveModel() throws -> (id: String, directory: URL) {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_FULL_PATH_IMAGE_GATE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_FULL_PATH_IMAGE_GATE=1 to run the full-path image gate")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_FULL_PATH_IMAGE_MODEL"] ?? "gemma-4-e4b-it-8bit"
        let directory = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(modelID)", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Model dir not found: \(directory.path)")
        }
        return (modelID, directory)
    }

    private func fullPathEngine(_ modelID: String, _ directory: URL) async throws -> MLXSwiftEngine {
        let engine = MLXSwiftEngine(
            promptCache: PromptCacheConfig(coldEnabled: FullPathVLMParitySmokeTests.usesColdTier))
        await engine.setForcesFullPathVLM(true)
        try await engine.load(LocalModel(
            id: modelID, displayName: modelID, directory: directory, sizeBytes: 0,
            format: .mlxVLM, quantization: nil, parameterCount: nil, architecture: nil))
        addTeardownBlock { try await engine.unload() }
        // Runs first: teardown blocks run last-added first.
        if FullPathVLMParitySmokeTests.usesColdTier { addTeardownBlock { await engine.clearPromptCache() } }
        let served = await engine.servesFullPathVLM
        XCTAssertTrue(served, "the load is on the full path")
        return engine
    }

    private func post(_ port: Int, _ body: [String: Any]) async throws -> [String: Any] {
        let url = try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions"))
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 900
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200, "HTTP \(http.statusCode): \(String(decoding: data, as: UTF8.self))")
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func message(_ response: [String: Any]) throws -> [String: Any] {
        let choices = try XCTUnwrap(response["choices"] as? [[String: Any]])
        return try XCTUnwrap(choices.first?["message"] as? [String: Any])
    }

    private static func content(_ response: [String: Any]) throws -> String {
        try message(response)["content"] as? String ?? ""
    }

    private static func firstToolCall(_ response: [String: Any]) throws -> (name: String, arguments: String) {
        let calls = try XCTUnwrap(try message(response)["tool_calls"] as? [[String: Any]], "no tool_calls in \(response)")
        let function = try XCTUnwrap(calls.first?["function"] as? [String: Any])
        return (function["name"] as? String ?? "", function["arguments"] as? String ?? "")
    }

    private static func cachedTokens(_ response: [String: Any]) -> Int? {
        let usage = response["usage"] as? [String: Any]
        let details = usage?["prompt_tokens_details"] as? [String: Any]
        return details?["cached_tokens"] as? Int
    }

    private static func imagePart(_ url: URL) throws -> [String: Any] {
        let base64 = try Data(contentsOf: url).base64EncodedString()
        return ["type": "image_url", "image_url": ["url": "data:image/png;base64,\(base64)"]]
    }

    private static func reads(_ text: String) -> Bool {
        text.uppercased().filter { !$0.isWhitespace }.contains(anchor.filter { !$0.isWhitespace })
    }

    /// High-contrast text drawn to a temporary 3840×960-pixel PNG. The size
    /// is in pixels, not points, so it does not follow the display's backing
    /// scale. Qwen3.5/3.6 needs it this large: on the original Qwen3.6-27B
    /// weights, the reference implementation (transformers) reads the year
    /// as "2020" at 2560×640 and 3200×800, and reads it right at this size
    /// with the text moved by a few pixels or drawn in a regular weight.
    private static func renderTextImage(_ text: String) throws -> URL {
        let (width, height) = (3840, 960)
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8,
            samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
            bytesPerRow: 0, bitsPerPixel: 0),
            let context = NSGraphicsContext(bitmapImageRep: rep)
        else { throw XCTSkip("Could not render the test image") }
        rep.size = NSSize(width: width, height: height)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        NSColor.white.setFill()
        NSRect(x: 0, y: 0, width: width, height: height).fill()
        (text as NSString).draw(
            at: NSPoint(x: 144, y: 312),
            withAttributes: [.font: NSFont.systemFont(ofSize: 336, weight: .bold), .foregroundColor: NSColor.black])
        NSGraphicsContext.restoreGraphicsState()
        guard let png = rep.representation(using: .png, properties: [:])
        else { throw XCTSkip("Could not render the test image") }
        let url = FileManager.default.temporaryDirectory
            .appending(path: "full-path-image-gate-\(UUID().uuidString).png")
        try png.write(to: url)
        return url
    }
}
