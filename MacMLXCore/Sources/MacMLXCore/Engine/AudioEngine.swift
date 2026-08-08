import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioTTS

// MARK: - AudioEngine

/// In-process MLX speech-to-text and text-to-speech, backed by
/// `Blaizzy/mlx-audio-swift`. Powers `POST /v1/audio/transcriptions` and
/// `POST /v1/audio/speech`.
///
/// A deliberate sibling to ``RerankEngine`` and ``EmbeddingEngine``: an `actor`
/// (so concurrent calls serialize) that owns non-`Sendable` MLX models and
/// only ever hands `Sendable` values back across the isolation boundary —
/// every `MLXArray` is `eval()`'d and materialized to `[Float]` inside the
/// actor before returning. STT and TTS are cached independently, each
/// cold-swapping when a different model id is requested, exactly like
/// `ensureRerankerLoaded`.
///
/// **Model identity differs from the rest of macMLX.** The upstream loaders
/// resolve a Hugging Face repo id through a `HubCache`, not a scanned local
/// directory — `STT.loadModel` in particular accepts only `owner/name` and has
/// no bare-local-directory path. So these two endpoints take a repo id
/// (`openai/whisper-tiny`, `mlx-community/Kokoro-82M-4bit`, …) rather than a
/// `macmlx list` entry. Wiring them into `ModelLibraryManager` is deferred.
///
/// - Note: Architecture-faithful but NOT validated against a real checkpoint
///   in this environment — no weights were downloaded and no model was run.
public actor AudioEngine {

    // MARK: Results

    /// One timed span of a transcript, normalized out of the upstream
    /// `STTOutput.segments` dictionaries into a `Sendable` shape.
    public struct TranscriptionSegment: Sendable, Equatable {
        /// Zero-based position in the transcript.
        public let id: Int
        /// Span start, seconds from the beginning of the audio.
        public let start: Double
        /// Span end, seconds from the beginning of the audio.
        public let end: Double
        /// The text of this span.
        public let text: String

        public init(id: Int, start: Double, end: Double, text: String) {
            self.id = id
            self.start = start
            self.end = end
            self.text = text
        }
    }

    /// A completed transcription.
    public struct Transcription: Sendable {
        public let text: String
        /// The language the model reported, when it reported one.
        public let language: String?
        /// Duration of the decoded audio in seconds — measured from the
        /// decoded sample count, not claimed by the model.
        public let duration: Double
        public let segments: [TranscriptionSegment]
    }

    /// A completed synthesis: mono float samples plus the model's native rate.
    public struct Speech: Sendable {
        public let samples: [Float]
        public let sampleRate: Int

        /// Public so a caller outside this module can construct one — the GUI's
        /// playback tests script synthesis results without a model. Mirrors
        /// ``TranscriptionSegment``, which is public-init for the same reason.
        public init(samples: [Float], sampleRate: Int) {
            self.samples = samples
            self.sampleRate = sampleRate
        }
    }

    // MARK: Configuration

    /// Where audio model downloads land: `~/.mac-mlx/audio-models/`.
    ///
    /// The upstream default would be the shared Hugging Face cache
    /// (`~/.cache/huggingface`, or wherever `HF_HOME` points). macMLX keeps
    /// everything it manages under its own data root so the GUI, the CLI, and
    /// the uninstall story all see one directory — the same reasoning behind
    /// every other ``DataRoot`` consumer.
    public static let modelCacheDirectory: URL = DataRoot.macMLX("audio-models")

    /// Sample rate every supported STT model expects. `loadAudioArray`
    /// resamples to it, so callers may upload any rate AVFoundation can read.
    public static let sttSampleRate = 16_000

    /// The `HubCache` both loaders use — pinned to ``modelCacheDirectory``.
    private static var hubCache: HubCache { HubCache(cacheDirectory: modelCacheDirectory) }

    // MARK: State

    /// Repo id of the resident STT model, or `nil` before the first load.
    public private(set) var loadedSTTModelID: String?
    /// Repo id of the resident TTS model, or `nil` before the first load.
    public private(set) var loadedTTSModelID: String?

    /// The loaded STT model. Non-`Sendable` (an `AnyObject` protocol), but
    /// actor-isolated so that's safe — and its `generate` is synchronous, so
    /// it never leaves the actor's execution context.
    private var sttModel: (any STTGenerationModel)?

    /// The loaded TTS model, behind ``SpeechModelBox``.
    private var ttsModel: SpeechModelBox?

    /// Holds the synthesis model and performs the one `async` call it exposes.
    ///
    /// `SpeechGenerationModel.generate` is `async` AND the conforming models
    /// are non-`Sendable` classes, so invoking it directly from actor-isolated
    /// code is a strict-concurrency error: the model would be *sent* out of the
    /// actor's region into a nonisolated async context. Wrapping it here moves
    /// that call into a nonisolated context, where no isolation boundary is
    /// crossed, and hands the actor back only `Sendable` values.
    ///
    /// The `@unchecked` conformance is justified by construction, not by
    /// inspection: the boxed model is created inside ``AudioEngine``, stored
    /// only in its actor-isolated property, and never handed out — so the
    /// actor's serialization is the sole access path. `HummingbirdServer`
    /// additionally holds the generation lock across every call, which
    /// serializes this against all other MLX compute in the process.
    private struct SpeechModelBox: @unchecked Sendable {
        let model: any SpeechGenerationModel

        var sampleRate: Int { model.sampleRate }

        /// Synthesize and materialize to `Sendable` floats before returning.
        func synthesize(text: String, voice: String?, language: String?) async throws -> [Float] {
            let waveform = try await model.generate(
                text: text, voice: voice, refAudio: nil, refText: nil, language: language)
            // MUST eval + materialize here — MLXArray is not Sendable, so it
            // can never be the value that crosses back to the actor.
            waveform.eval()
            return waveform.asArray(Float.self)
        }
    }

    public init() {}

    // MARK: Loading

    /// Load an STT model, cold-swapping when a different repo id is requested.
    ///
    /// - Parameter modelID: A Hugging Face repo id (`owner/name`). Validated
    ///   before any network call so a malformed id fails fast and locally.
    /// - Throws: ``EngineError/invalidAudioModelID(reason:)`` for a malformed
    ///   id — raised here, with nothing sent over the network, which is why the
    ///   caller can answer 400 — or ``EngineError/modelLoadFailed(reason:)``
    ///   for an upstream load failure (unknown architecture, missing weights,
    ///   no network, …), which really is server-side.
    public func loadSTT(_ modelID: String) async throws {
        if loadedSTTModelID == modelID, sttModel != nil { return }
        guard Self.isHubRepoID(modelID) else {
            throw EngineError.invalidAudioModelID(reason: Self.repoIDHint(modelID, kind: "STT"))
        }
        // Drop the old model first: two resident audio models plus an LLM is
        // more memory than a swap needs to hold.
        sttModel = nil
        loadedSTTModelID = nil
        do {
            sttModel = try await STT.loadModel(modelRepo: modelID, cache: Self.hubCache)
            loadedSTTModelID = modelID
        } catch {
            sttModel = nil
            loadedSTTModelID = nil
            throw EngineError.modelLoadFailed(reason: error.localizedDescription)
        }
    }

    /// Load a TTS model, cold-swapping when a different repo id is requested.
    ///
    /// - Parameter modelID: A Hugging Face repo id (`owner/name`). Unlike STT,
    ///   the upstream TTS loader also accepts an absolute local directory
    ///   containing `config.json`; that path is left unvalidated here and
    ///   simply forwarded.
    /// - Throws: ``EngineError/invalidAudioModelID(reason:)`` for an id that is
    ///   neither shape, or ``EngineError/modelLoadFailed(reason:)`` for an
    ///   upstream load failure. Same split as ``loadSTT(_:)``.
    public func loadTTS(_ modelID: String) async throws {
        if loadedTTSModelID == modelID, ttsModel != nil { return }
        guard Self.isHubRepoID(modelID) || Self.looksLikeLocalDirectory(modelID) else {
            throw EngineError.invalidAudioModelID(reason: Self.repoIDHint(modelID, kind: "TTS"))
        }
        ttsModel = nil
        loadedTTSModelID = nil
        do {
            ttsModel = SpeechModelBox(
                model: try await TTS.loadModel(modelRepo: modelID, cache: Self.hubCache))
            loadedTTSModelID = modelID
        } catch {
            ttsModel = nil
            loadedTTSModelID = nil
            throw EngineError.modelLoadFailed(reason: error.localizedDescription)
        }
    }

    // MARK: Inference

    /// Transcribe a local audio file.
    ///
    /// Decodes through AVFoundation (`loadAudioArray`, which resamples to
    /// ``sttSampleRate``), runs one forward pass, and materializes a
    /// `Sendable` result. `duration` is derived from the decoded sample count
    /// so it describes the audio we actually fed the model.
    ///
    /// - Parameters:
    ///   - audioURL: A decodable audio file on disk. The caller owns its
    ///     lifetime; nothing here deletes it.
    ///   - language: ISO-639-1 hint forwarded to the model, or `nil` to let
    ///     the model detect.
    ///   - temperature: Sampling temperature, or `nil` for the model default.
    /// - Throws: ``EngineError/modelNotLoaded`` before ``loadSTT(_:)``, or
    ///   ``EngineError/audioProcessingFailed(reason:)`` when decoding fails.
    public func transcribe(
        audioURL: URL,
        language: String? = nil,
        temperature: Float? = nil
    ) async throws -> Transcription {
        guard let sttModel else { throw EngineError.modelNotLoaded }

        let audio: MLXArray
        let sampleCount: Int
        do {
            let (_, decoded) = try loadAudioArray(from: audioURL, sampleRate: Self.sttSampleRate)
            audio = decoded
            sampleCount = decoded.size
        } catch {
            throw EngineError.audioProcessingFailed(
                reason: "Could not decode the uploaded audio: \(error.localizedDescription)")
        }

        let defaults = sttModel.defaultGenerationParameters
        let parameters = STTGenerateParameters(
            maxTokens: defaults.maxTokens,
            temperature: temperature ?? defaults.temperature,
            topP: defaults.topP,
            topK: defaults.topK,
            verbose: false,
            language: language ?? defaults.language,
            chunkDuration: defaults.chunkDuration,
            minChunkDuration: defaults.minChunkDuration,
            repetitionPenalty: defaults.repetitionPenalty,
            repetitionContextSize: defaults.repetitionContextSize
        )
        let output = sttModel.generate(audio: audio, generationParameters: parameters)

        return Transcription(
            text: output.text,
            language: output.language,
            duration: Double(sampleCount) / Double(Self.sttSampleRate),
            segments: Self.segments(from: output.segments)
        )
    }

    /// Synthesize speech from text.
    ///
    /// - Parameters:
    ///   - text: What to say.
    ///   - voice: Model-specific voice id, or `nil` for the model default.
    ///   - language: Language hint, or `nil` for the model default.
    /// - Returns: Mono float samples plus the model's native sample rate. The
    ///   caller decides the container (see ``SpeechAudioFormat``).
    /// - Throws: ``EngineError/modelNotLoaded`` before ``loadTTS(_:)``, or
    ///   ``EngineError/audioProcessingFailed(reason:)`` when synthesis fails.
    public func synthesize(
        text: String,
        voice: String? = nil,
        language: String? = nil
    ) async throws -> Speech {
        guard let ttsModel else { throw EngineError.modelNotLoaded }
        do {
            let samples = try await ttsModel.synthesize(
                text: text, voice: voice, language: language)
            return Speech(samples: samples, sampleRate: ttsModel.sampleRate)
        } catch {
            throw EngineError.audioProcessingFailed(reason: error.localizedDescription)
        }
    }

    // MARK: Pure helpers (testable without weights)

    /// Normalize the upstream `STTOutput.segments` — an untyped
    /// `[[String: Any]]` whose entries carry `text` / `start` / `end` — into
    /// typed, `Sendable` ``TranscriptionSegment``s.
    ///
    /// Kept a `nonisolated static` PURE function so the shape the
    /// `verbose_json` response depends on is unit-testable with no model.
    /// Deliberately lenient about what a model emits: a numeric field may
    /// arrive as `Double`, `Int`, or a numeric `String` (models in this family
    /// are not consistent), and a missing timestamp becomes `0`. Entries with
    /// no `text` at all are dropped, since a segment with no words is not
    /// something a client can use.
    nonisolated static func segments(from raw: [[String: Any]]?) -> [TranscriptionSegment] {
        guard let raw else { return [] }
        var result: [TranscriptionSegment] = []
        result.reserveCapacity(raw.count)
        for entry in raw {
            guard let text = entry["text"] as? String else { continue }
            result.append(TranscriptionSegment(
                id: result.count,
                start: seconds(entry["start"]),
                end: seconds(entry["end"]),
                text: text
            ))
        }
        return result
    }

    /// Coerce an untyped timestamp to seconds, defaulting to `0`.
    private nonisolated static func seconds(_ value: Any?) -> Double {
        switch value {
        case let double as Double: return double
        case let int as Int: return Double(int)
        case let float as Float: return Double(float)
        case let number as NSNumber: return number.doubleValue
        case let string as String: return Double(string) ?? 0
        default: return 0
        }
    }

    /// Whether `id` is shaped like a Hugging Face repo id (`owner/name`).
    ///
    /// Checked locally so a typo fails immediately with a useful message
    /// instead of after a network round trip. Mirrors the Hub's own rules:
    /// exactly one `/`, both halves non-empty, no whitespace, and no path
    /// traversal (which would otherwise escape the cache directory).
    nonisolated static func isHubRepoID(_ id: String) -> Bool {
        let components = id.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 2 else { return false }
        for component in components {
            if component.isEmpty { return false }
            if component == "." || component == ".." { return false }
            if component.contains(where: { $0.isWhitespace }) { return false }
        }
        return true
    }

    /// Whether `id` points at an existing directory holding a `config.json` —
    /// the shape the upstream TTS loader accepts in place of a repo id.
    nonisolated static func looksLikeLocalDirectory(_ id: String) -> Bool {
        guard id.hasPrefix("/") || id.hasPrefix("~") else { return false }
        let expanded = (id as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory),
              isDirectory.boolValue
        else { return false }
        return FileManager.default.fileExists(
            atPath: URL(filePath: expanded).appending(path: "config.json").path)
    }

    /// The message a rejected model id gets. Split out so both loaders and
    /// their tests share one wording.
    nonisolated static func repoIDHint(_ id: String, kind: String) -> String {
        "'\(id)' is not a Hugging Face repo id. macMLX loads \(kind) models by repo id "
            + "(`owner/name`, e.g. `openai/whisper-tiny`), not by `macmlx list` name — audio "
            + "models are cached separately under ~/.mac-mlx/audio-models/."
    }
}
