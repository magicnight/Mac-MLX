import Foundation
import HuggingFace
import XCTest

@testable import MacMLXCore

/// Gated, network smoke for #135: an audio model loaded through `AudioEngine`
/// must land in the app's cache and nowhere else. Six upstream speech-to-text
/// families (voxtral, cohere, canary, wav2vec2/mms, lasr, moonshine) resolved
/// their model a second time without the cache they were given, so the
/// weights they read came from a second full download into `HubCache.default`;
/// on the text-to-speech side Chatterbox and OmniVoice did the same for their
/// whole model, and the Llama and Qwen3 voices and Echo for the codec they
/// fetch from a second repo. The controlled fork passes the cache through;
/// this test watches the default cache stay empty.
///
/// Self-skips unless ALL hold:
///   1. `requireMLXRuntimeOrSkip()` passes (loading weights needs Metal),
///   2. env `MACMLX_RUN_AUDIO_CACHE_SMOKE=1`,
///   3. env `HF_HUB_CACHE` names the directory `HubCache.default` resolves to.
///      The test reads it and never clears it: point it at an empty temporary
///      directory, so "nothing landed there" means something.
///
/// `MACMLX_AUDIO_CACHE_SMOKE_MODEL` picks the repo; the default is the smallest
/// of the six families, `moonshine-ai/moonshine-tiny` (110 MB). Run it with
/// `mlx-community/SenseVoiceSmall` (936 MB; its `am.mvn` and `*.bpe.model`
/// come only through the loader's own patterns) to cover a loader that
/// fetches files the first resolve did not; `MACMLX_AUDIO_CACHE_SMOKE_FILES`
/// (comma-separated names) then lists the files the app snapshot must hold.
/// `MACMLX_AUDIO_CACHE_SMOKE_KIND=tts` loads a speech model instead; a voice
/// that fetches its codec from a second repo (`mlx-community/VyvoTTS-EN-Beta-4bit`,
/// a Qwen3 voice, fetches `mlx-community/snac_24khz`) must put that repo in
/// the app cache too, which `MACMLX_AUDIO_CACHE_SMOKE_CODEC` (the second
/// repo's id) makes the test check.
///
/// Downloads into `~/.mac-mlx/audio-models` and removes the entries a run
/// added there (snapshots, `models--*` repos, locks); files added inside an
/// entry that already existed stay. Run:
///   TEST_RUNNER_MACMLX_RUN_AUDIO_CACHE_SMOKE=1 \
///     TEST_RUNNER_HF_HUB_CACHE=$(mktemp -d) \
///     xcodebuild test -scheme MacMLXCore -destination 'platform=macOS' \
///     -skipPackagePluginValidation \
///     -only-testing:MacMLXCoreTests/AudioEngineCacheSmokeTests
final class AudioEngineCacheSmokeTests: XCTestCase {

    /// The identity of the snapshot's weights file (nil when there is none yet).
    private func weightsIdentifier(in snapshot: URL) throws -> AnyHashable? {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: snapshot.path)) ?? []
        guard let weights = files.sorted().first(where: { $0.hasSuffix(".safetensors") }) else { return nil }
        let values = try snapshot.appending(path: weights).resourceValues(forKeys: [.fileResourceIdentifierKey])
        return values.fileResourceIdentifier.flatMap { $0 as? NSObject }.map { AnyHashable($0) }
    }

    /// The entries under `directory`, to tell what a run added.
    private func entries(of directory: URL) -> Set<String> {
        Set((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? [])
    }

    func testAudioModelLandsOnlyInTheAppCache() async throws {
        try requireMLXRuntimeOrSkip()
        let env = ProcessInfo.processInfo.environment
        guard env["MACMLX_RUN_AUDIO_CACHE_SMOKE"] == "1" else {
            throw XCTSkip("Set MACMLX_RUN_AUDIO_CACHE_SMOKE=1 to run the audio cache smoke test")
        }
        guard let hubCachePath = env["HF_HUB_CACHE"] else {
            throw XCTSkip("Set HF_HUB_CACHE to an empty temporary directory so the default Hub cache can be watched")
        }
        let defaultCache = URL(fileURLWithPath: hubCachePath, isDirectory: true).standardizedFileURL
        guard HubCache.default.cacheDirectory.standardizedFileURL == defaultCache else {
            XCTFail("HubCache.default must follow HF_HUB_CACHE for this test to watch it; it resolves to \(HubCache.default.cacheDirectory.path)")
            return
        }

        let modelID = env["MACMLX_AUDIO_CACHE_SMOKE_MODEL"] ?? "moonshine-ai/moonshine-tiny"
        let speech = env["MACMLX_AUDIO_CACHE_SMOKE_KIND"] == "tts"
        let fm = FileManager.default
        let cacheRoot = AudioEngine.modelCacheDirectory
        let snapshots = cacheRoot.appending(path: "mlx-audio", directoryHint: .isDirectory)
        let locks = cacheRoot.appending(path: ".locks", directoryHint: .isDirectory)
        // Where the fork puts the snapshot it loads from. A model that fetches
        // a tokenizer or codec from a second repo adds that repo's snapshot,
        // blobs and locks beside it; everything a run adds is removed after.
        let appSnapshot = snapshots.appending(
            path: modelID.replacingOccurrences(of: "/", with: "_"), directoryHint: .isDirectory)
        let snapshotExisted = fm.fileExists(atPath: appSnapshot.path)
        let codec = env["MACMLX_AUDIO_CACHE_SMOKE_CODEC"].flatMap { $0.isEmpty ? nil : $0 }
        let codecSnapshot = codec.map {
            snapshots.appending(path: $0.replacingOccurrences(of: "/", with: "_"), directoryHint: .isDirectory)
        }
        let codecExisted = codecSnapshot.map { fm.fileExists(atPath: $0.path) } ?? false
        let before = (snapshots: entries(of: snapshots), repos: entries(of: cacheRoot), locks: entries(of: locks))
        defer {
            for added in entries(of: snapshots).subtracting(before.snapshots) {
                try? fm.removeItem(at: snapshots.appending(path: added))
            }
            for added in entries(of: cacheRoot).subtracting(before.repos) where added.hasPrefix("models--") {
                try? fm.removeItem(at: cacheRoot.appending(path: added))
            }
            for added in entries(of: locks).subtracting(before.locks) {
                try? fm.removeItem(at: locks.appending(path: added))
            }
        }

        let engine = AudioEngine()
        if speech {
            try await engine.prepareTTS(model: modelID)
        } else {
            try await engine.prepareSTT(model: modelID)
        }
        // The weights the prefetch put there must survive the load untouched:
        // a loader that asks for more fetches only the files it lacks.
        let weightsBefore = try weightsIdentifier(in: appSnapshot)
        XCTAssertNotNil(weightsBefore, "the prefetch must have written the weights")
        if speech {
            try await engine.loadTTS(modelID)
        } else {
            try await engine.loadSTT(modelID)
        }
        let weightsAfter = try weightsIdentifier(in: appSnapshot)
        XCTAssertEqual(weightsAfter, weightsBefore, "the loader must not replace the weights the prefetch wrote")

        let appFiles = try fm.contentsOfDirectory(atPath: appSnapshot.path)
        XCTAssertTrue(appFiles.contains("config.json"), "the app cache must hold the snapshot, got \(appFiles)")
        XCTAssertTrue(
            appFiles.contains { $0.hasSuffix(".safetensors") },
            "the app cache must hold the weights, got \(appFiles)")
        let required = (env["MACMLX_AUDIO_CACHE_SMOKE_FILES"] ?? "")
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        if !required.isEmpty {
            // Only a model not yet in the app cache proves anything about what
            // the loader fetched; with the snapshot already there the files
            // might as well have come from an earlier run.
            XCTAssertFalse(snapshotExisted, "MACMLX_AUDIO_CACHE_SMOKE_FILES needs a model not yet in the app cache")
            for file in required {
                XCTAssertTrue(
                    appFiles.contains(file),
                    "the loader's own patterns must land in the app cache: \(file) is missing from \(appFiles)")
            }
        }

        if let codec, let codecSnapshot {
            // As for the files: only a codec not yet in the app cache proves
            // this load put it there.
            XCTAssertFalse(codecExisted, "MACMLX_AUDIO_CACHE_SMOKE_CODEC needs a codec not yet in the app cache")
            let codecFiles = (try? fm.contentsOfDirectory(atPath: codecSnapshot.path)) ?? []
            XCTAssertTrue(
                codecFiles.contains { $0.hasSuffix(".safetensors") },
                "the codec fetched from \(codec) must land in the app cache, got \(codecFiles)")
        }

        let stray = (try? fm.subpathsOfDirectory(atPath: defaultCache.path)) ?? []
        XCTAssertTrue(stray.isEmpty, "nothing may land in the default Hub cache; found \(stray)")
    }
}
