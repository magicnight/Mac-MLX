// Copyright © 2026 macMLX. English comments only.

import Foundation
import XCTest
import AppKit

@testable import MacMLXCore

/// Gated, real-weights smoke for GLM-OCR through macMLX's existing VLM path.
///
/// This is the FIRST end-to-end real-VLM smoke in the suite (the numeric-parity
/// and routing tests never load a real checkpoint). It proves that an OCR VLM
/// runs through the stock pipeline with no new model code:
///
///  a. **VLM routing** — a `.mlxVLM` `LocalModel` makes `MLXSwiftEngine.load`
///     go through `VLMModelFactory.shared.loadContainer`, which resolves
///     `config.json`'s `model_type: glm_ocr` to mlx-swift-lm's stock `GlmOcr`.
///  b. **image handoff** — the request carries an `ImageAttachment`; the engine
///     folds it into the `Chat.Message` image bag (`.url(fileURL)`), and GLM-OCR's
///     `UserInputProcessor` injects the image tokens.
///  c. **OCR generation** — greedy decode over a rendered, known text image must
///     read that text back. The image is drawn at runtime (no binary fixture).
///
/// GATED — never runs in CI. Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (real Metal, i.e. xcodebuild),
///   2. env `MACMLX_RUN_GLM_OCR_SMOKE=1`, and
///   3. the checkpoint is found on disk. Discovery order:
///        • env `MACMLX_GLM_OCR_MODEL_DIR` (a full snapshot-directory path), else
///        • `~/.mac-mlx/models/<MACMLX_GLM_OCR_MODEL>` (default `GLM-OCR-4bit`).
///
/// Run (with the ~1.2 GB checkpoint in `~/.mac-mlx/models/GLM-OCR-4bit`):
///   MACMLX_RUN_GLM_OCR_SMOKE=1 TEST_RUNNER_MACMLX_RUN_GLM_OCR_SMOKE=1 \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/GLMOCRSmokeTests/testGLMOCRReadsRenderedText
final class GLMOCRSmokeTests: XCTestCase {

    /// Distinctive, OCR-friendly text rendered into the test image. Uppercase +
    /// a digit run so a correct read is unambiguous and the substring check is
    /// robust to spacing/newline quirks in the model's output.
    private static let anchor = "MACMLX OCR 2026"

    private func resolveModelDirectory() -> URL? {
        let fm = FileManager.default
        func hasConfig(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appending(path: "config.json").path)
        }
        if let explicit = ProcessInfo.processInfo.environment["MACMLX_GLM_OCR_MODEL_DIR"] {
            let dir = URL(fileURLWithPath: explicit, isDirectory: true)
            return hasConfig(dir) ? dir : nil
        }
        let name = ProcessInfo.processInfo.environment["MACMLX_GLM_OCR_MODEL"] ?? "GLM-OCR-4bit"
        let local = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appending(path: ".mac-mlx/models/\(name)", directoryHint: .isDirectory)
        return hasConfig(local) ? local : nil
    }

    private func localVLM(id: String, directory: URL) -> LocalModel {
        // Format is `.mlxVLM` so `load` routes through `VLMModelFactory`. The
        // upgradeFormat detection that maps glm_ocr → .mlxVLM is covered by a
        // separate, model-free unit test.
        LocalModel(
            id: id,
            displayName: id,
            directory: directory,
            sizeBytes: 0,
            format: .mlxVLM,
            quantization: nil,
            parameterCount: nil,
            architecture: nil
        )
    }

    /// Draw high-contrast black-on-white text to a PNG in a temp file and return
    /// its URL. Rendered large so the OCR read is unambiguous.
    private func renderTextImage(_ text: String) throws -> URL {
        let size = NSSize(width: 640, height: 160)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 56, weight: .bold),
            .foregroundColor: NSColor.black,
        ]
        (text as NSString).draw(at: NSPoint(x: 24, y: 52), withAttributes: attrs)
        image.unlockFocus()

        guard let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:])
        else {
            throw XCTSkip("Could not render the OCR test image")
        }
        let url = FileManager.default.temporaryDirectory
            .appending(path: "glm-ocr-smoke-\(UUID().uuidString).png")
        try png.write(to: url)
        return url
    }

    func testGLMOCRReadsRenderedText() async throws {
        try requireMLXRuntimeOrSkip()

        guard ProcessInfo.processInfo.environment["MACMLX_RUN_GLM_OCR_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_GLM_OCR_SMOKE=1 to run the GLM-OCR real-weights smoke test")
        }
        guard let directory = resolveModelDirectory() else {
            throw XCTSkip("GLM-OCR checkpoint not found (MACMLX_GLM_OCR_MODEL_DIR / ~/.mac-mlx/models/GLM-OCR-4bit)")
        }
        let modelID = "GLM-OCR-4bit"

        let imageURL = try renderTextImage(Self.anchor)
        defer { try? FileManager.default.removeItem(at: imageURL) }

        let engine = MLXSwiftEngine()
        let parameters = GenerationParameters(
            temperature: 0, topP: 1.0, maxTokens: 128, stream: true)
        let request = GenerateRequest(
            model: modelID,
            messages: [
                ChatMessage(
                    role: .user,
                    content: "What text is written in this image? Output only the text.",
                    images: [ImageAttachment(fileURL: imageURL, mimeType: "image/png")]
                )
            ],
            parameters: parameters
        )

        var text = ""
        var completionTokens: Int?
        let start = Date()
        try await engine.load(localVLM(id: modelID, directory: directory))
        for try await chunk in engine.generate(request) {
            text += chunk.text
            if let usage = chunk.usage { completionTokens = usage.completionTokens }
        }
        let elapsed = Date().timeIntervalSince(start)

        // Echo the raw read first so a failed assertion still shows what the model saw.
        print("GLM_OCR_SMOKE_TEXT<<<\(text)>>>")
        if let completionTokens, elapsed > 0 {
            print(
                "GLM_OCR_SMOKE model=\(modelID) dir=\(directory.path) "
                    + "completionTokens=\(completionTokens) "
                    + "elapsed=\(String(format: "%.2f", elapsed))s "
                    + "tokPerSec=\(String(format: "%.1f", Double(completionTokens) / elapsed))")
        }

        XCTAssertFalse(text.isEmpty, "GLM-OCR must produce real output, not an early-exit stub")
        // The core claim: the OCR read contains the rendered anchor. Compare
        // case-insensitively and ignoring whitespace so spacing/newline quirks in
        // the model's formatting don't fail an otherwise-correct read.
        let normalized = text.uppercased().filter { !$0.isWhitespace }
        let anchorNormalized = Self.anchor.uppercased().filter { !$0.isWhitespace }
        XCTAssertTrue(
            normalized.contains(anchorNormalized),
            "GLM-OCR must read the rendered text '\(Self.anchor)' back — got: \(text)")
    }
}
