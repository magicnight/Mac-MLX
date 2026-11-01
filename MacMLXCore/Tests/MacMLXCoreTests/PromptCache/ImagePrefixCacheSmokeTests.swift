// Copyright © 2026 macMLX. English comments only.

import AppKit
import Foundation
import MLXLMCommon
import XCTest

@testable import MacMLXCore
// MLXLMCommon has a `JSONValue` of its own; the scoped import makes ours win
// (see the same note in PromptCacheStoreTests).
import enum MacMLXCore.JSONValue

/// Gated, real-weights check of image cache step 1 on the full path
/// (``FullPathVLM/imageRequestLookupKey(_:)``): an image request after a long
/// text turn reuses the cache up to its first media block and still reads the
/// image the way a cold run does.
///
/// Turn 1 is a long system prompt and a text question. Its cache cannot be
/// trimmed (Qwen3.6's `MambaCache`; Gemma 4's sliding window, past its 512
/// tokens), so it ends with a history-boundary snapshot as its only entry.
/// Turn 2 replays turn 1 with a fixed answer and asks about an image of
/// rendered text, with thinking off so the first tokens read the image.
///
/// What it asserts, on the stock path and on the hand-built pipeline
/// (`logprobs`, the run the comparison is made on):
/// - turn 2 reuses exactly turn 1's history boundary, a strict prefix of its
///   capped key, and records nothing;
/// - turn 2 run cold reuses nothing and reads the image as the stock VLM
///   path does (its greedy text is the same; whether that text is the
///   rendered one depends on the model's OCR and is only printed);
/// - warm against cold, they match under ``NearTieRule``: the warm run's
///   prefix was prefilled in another piece, so they do not round alike.
///
/// The load is always the full path, through the forcing seam, whatever
/// `MACMLX_GATED_FULL_PATH_VLM` says: image requests take the text path only
/// there. Every engine keeps its cold tier off.
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_IMAGE_PREFIX_SMOKE=1`, and
///   3. the model directory exists (env `MACMLX_IMAGE_PREFIX_MODEL`
///      overrides the directory name under `~/.mac-mlx/models`; default
///      `Qwen3.6-27B-4bit`).
///
/// Run:
///   MACMLX_RUN_IMAGE_PREFIX_SMOKE=1 TEST_RUNNER_MACMLX_RUN_IMAGE_PREFIX_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/ImagePrefixCacheSmokeTests
/// and, for Gemma 4, add
///   TEST_RUNNER_MACMLX_IMAGE_PREFIX_MODEL=gemma-4-e4b-it-8bit
final class ImagePrefixCacheSmokeTests: XCTestCase {

    private static let anchor = "MACMLX 2026"

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

    /// About 160 tokens a section; ten clear the snapshot minimum and carry
    /// the prompt past Gemma 4's 512-token sliding window.
    private static func systemPrompt(sections: Int = 10) -> String {
        (1...sections).map { section in
            "Section \(section).\n" + rules.map { "- \($0)" }.joined(separator: "\n")
        }.joined(separator: "\n\n")
    }

    private struct Collected {
        var text = ""
        var promptTokens: Int?
        var cachedPromptTokens: Int?
        var logprobs: [TokenLogprob] = []
        var seconds: Double = 0
    }

    private func collect(_ engine: MLXSwiftEngine, _ request: GenerateRequest) async throws -> Collected {
        var out = Collected()
        let start = Date()
        for try await chunk in engine.generate(request) {
            out.text += chunk.text
            out.logprobs += chunk.logprobs ?? []
            if let usage = chunk.usage {
                out.promptTokens = usage.promptTokens
                out.cachedPromptTokens = usage.cachedPromptTokens
            }
        }
        out.seconds = Date().timeIntervalSince(start)
        return out
    }

    private static let templateKwargs: [String: JSONValue] = ["enable_thinking": .bool(false)]

    private func turns(
        modelID: String, image: URL, logprobs: Bool
    ) -> (turn1: GenerateRequest, turn2: GenerateRequest) {
        var parameters = GenerationParameters(temperature: 0, topP: 1.0, maxTokens: 4, stream: true)
        parameters.logprobs = logprobs
        // The other run's token has to be among the alternatives at a
        // near-tie.
        parameters.topLogprobs = logprobs ? 5 : nil
        let system = Self.systemPrompt()
        let question = ChatMessage(role: .user, content: "I will send you an image next. Are you ready?")
        var turn1 = GenerateRequest(
            model: modelID, messages: [question], systemPrompt: system, parameters: parameters)
        turn1.templateKwargs = Self.templateKwargs
        var turn2Parameters = parameters
        turn2Parameters.maxTokens = 16
        var turn2 = GenerateRequest(
            model: modelID,
            messages: [
                question,
                ChatMessage(role: .assistant, content: "Yes, send it."),
                ChatMessage(
                    role: .user, content: "What text is written in this image? Output only the text.",
                    images: [ImageAttachment(fileURL: image, mimeType: "image/png")]),
            ],
            systemPrompt: system, parameters: turn2Parameters)
        turn2.templateKwargs = Self.templateKwargs
        return (turn1, turn2)
    }

    /// Turn 1's chat rendered without the generation prompt, the way the
    /// engine renders its history boundary on this load.
    private func historyBoundary(
        _ request: GenerateRequest, generator: any MessageGenerator, tokenizer: TokenizerBridge
    ) throws -> Int {
        let input = UserInput(
            chat: request.allMessages.map { MLXSwiftEngine.upstreamChatMessage(from: $0, images: []) },
            additionalContext: ["enable_thinking": false])
        return try tokenizer.applyChatTemplate(
            messages: generator.generate(from: input), tools: nil,
            additionalContext: input.additionalContext, addGenerationPrompt: false
        ).count
    }

    private func modelDirectory() throws -> (id: String, url: URL) {
        try requireMLXRuntimeOrSkip()
        guard ProcessInfo.processInfo.environment["MACMLX_RUN_IMAGE_PREFIX_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_IMAGE_PREFIX_SMOKE=1 to run the image prefix prompt cache smoke")
        }
        let modelID = ProcessInfo.processInfo.environment["MACMLX_IMAGE_PREFIX_MODEL"] ?? "Qwen3.6-27B-4bit"
        let directory = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(modelID)", directoryHint: .isDirectory)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw XCTSkip("Model dir not found: \(directory.path)")
        }
        return (modelID, directory)
    }

    /// High-contrast text drawn to a temporary PNG, large enough to read.
    /// Drawn at 3840×960 pixels, as the full-path image gate draws its
    /// fixture: Qwen3.5/3.6 misreads this text at smaller sizes, in the
    /// reference implementation too.
    private func renderTextImage(_ text: String) throws -> URL {
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
            .appending(path: "image-prefix-smoke-\(UUID().uuidString).png")
        try png.write(to: url)
        return url
    }

    private static func localVLM(_ model: (id: String, url: URL)) -> LocalModel {
        LocalModel(
            id: model.id, displayName: model.id, directory: model.url, sizeBytes: 0,
            format: .mlxVLM, quantization: nil, parameterCount: nil, architecture: nil)
    }

    /// Turn 2 cold on the stock VLM path, the decoder every image request
    /// took before the full path: the reference for what the model reads in
    /// the image, whatever its OCR is worth. Unloaded before the full-path
    /// load, so the checkpoint is resident once.
    private func stockVLMText(model: (id: String, url: URL), image: URL) async throws -> String {
        let engine = MLXSwiftEngine(promptCache: PromptCacheConfig(coldEnabled: false))
        await engine.setKeepsStockVLMPath(true)
        try await engine.load(Self.localVLM(model))
        let servesFullPath = await engine.servesFullPathVLM
        XCTAssertFalse(servesFullPath, "the reference load is on the stock VLM path")
        let out = try await collect(engine, turns(modelID: model.id, image: image, logprobs: false).turn2)
        try await engine.unload()
        return out.text
    }

    private static func reads(_ text: String) -> Bool {
        text.uppercased().filter { !$0.isWhitespace }.contains(anchor.filter { !$0.isWhitespace })
    }

    private static func describe(_ run: Collected) -> String {
        let timing = String(format: "%.2f s", run.seconds)
        return "prompt=\(run.promptTokens ?? -1) reused=\(run.cachedPromptTokens ?? -1) \(timing) "
            + run.text.debugDescription
    }

    func testAnImageRequestReusesTheTextBeforeItAndReadsTheImageAsAColdRunDoes() async throws {
        let model = try modelDirectory()
        let image = try renderTextImage(Self.anchor)
        defer { try? FileManager.default.removeItem(at: image) }

        let reference = try await stockVLMText(model: model, image: image)
        print("[image-prefix] \(model.id) stock VLM path: \(reference.debugDescription)")

        let engine = MLXSwiftEngine(promptCache: PromptCacheConfig(coldEnabled: false))
        await engine.setForcesFullPathVLM(true)
        try await engine.load(Self.localVLM(model))
        addTeardownBlock { try await engine.unload() }
        let servesFullPath = await engine.servesFullPathVLM
        XCTAssertTrue(servesFullPath, "the forced load took the full path")
        let fullPath = try XCTUnwrap(
            FullPathVLM.resolve(configURL: model.url.appending(path: "config.json"), forced: true))
        let loaded = try await HuggingFaceTokenizerLoader().load(from: model.url)
        let tokenizer = try XCTUnwrap(loaded as? TokenizerBridge)

        var stockTexts: [String] = []
        for logprobs in [false, true] {
            let path = logprobs ? "custom pipeline" : "stock path"
            let (turn1, turn2) = turns(modelID: model.id, image: image, logprobs: logprobs)
            let boundary = try historyBoundary(
                turn1, generator: fullPath.messageGenerator, tokenizer: tokenizer)
            XCTAssertGreaterThanOrEqual(
                boundary, HistoryBoundary.minimumBoundarySnapshotTokens,
                "\(path): the system prompt clears the snapshot minimum")

            await engine.clearPromptCache()
            let first = try await collect(engine, turn1)
            let turn1Entries = await engine.promptCacheResidentCount
            let warm = try await collect(engine, turn2)
            let warmRecord = await engine.lastPromptCacheRecord
            await engine.clearPromptCache()
            let cold = try await collect(engine, turn2)

            let label = "[image-prefix] \(model.id) \(path):"
            print("\(label) boundary=\(boundary), hot-tier entries after turn 1: \(turn1Entries)")
            print("\(label) turn 1      \(Self.describe(first))")
            print("\(label) turn 2 warm \(Self.describe(warm))")
            print("\(label) turn 2 cold \(Self.describe(cold))")

            XCTAssertEqual(first.cachedPromptTokens ?? 0, 0, "\(path): turn 1 starts cold")
            XCTAssertEqual(turn1Entries, 1, "\(path): turn 1 left its history-boundary snapshot")
            XCTAssertEqual(warm.cachedPromptTokens, boundary, "\(path): turn 2 reuses turn 1's history boundary")
            XCTAssertNil(warmRecord, "\(path): the image request records nothing")
            XCTAssertEqual(cold.cachedPromptTokens ?? 0, 0, "\(path): the reference run is cold")
            XCTAssertEqual(warm.promptTokens, cold.promptTokens, "\(path): the warm turn reports the full prompt")
            XCTAssertEqual(
                cold.text, reference, "\(path): the cold run reads the image as the stock VLM decoder does")
            print("\(label) reads \(Self.anchor.debugDescription): \(Self.reads(warm.text))")

            let texts = [warm.text, cold.text]
            if logprobs {
                XCTAssertEqual(texts, stockTexts, "the runs decode alike on both paths")
                NearTieRule().assertMatches(
                    .init(name: "warm", text: warm.text, logprobs: warm.logprobs),
                    .init(name: "cold", text: cold.text, logprobs: cold.logprobs),
                    "turn 2 warm against cold")
            } else {
                stockTexts = texts
            }
        }
    }
}
