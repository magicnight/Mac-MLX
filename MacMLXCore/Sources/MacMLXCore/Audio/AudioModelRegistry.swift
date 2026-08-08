import Foundation

/// Classifies an on-disk `mlx-audio-swift` checkpoint as speech-to-text or
/// text-to-speech, using the same signals the upstream loaders use.
///
/// This is the audio analogue of `ModelLibraryManager.upgradeFormat`, but it
/// lives in its own type because audio models are discovered from a DIFFERENT
/// on-disk layout than every other model macMLX knows about (see
/// ``cacheSubdirectory``) and are classified against a registry that belongs
/// to a third-party package rather than to mlx-swift-lm.
///
/// **Honest gate.** A `model_type` that appears in neither registry is
/// classified `nil` and the directory is not surfaced at all — the same
/// discipline `LocalModel.isOCR` applies (only badge what can actually load).
/// The one guess upstream makes is deliberately NOT reproduced here: both
/// loaders fall back to inferring a type from the *repo name* when
/// `config.json` yields nothing (`STT.inferModelType`, `TTS.inferModelType`).
/// That fallback is a last resort at load time, where the user has already
/// named the model; using it for classification would let a directory whose
/// config says nothing recognisable acquire a badge on the strength of its
/// name alone. Config-driven only.
///
/// - Note: The two key sets are verified DISJOINT (24 STT keys, 34 TTS keys,
///   empty intersection) against the pinned revision, which is what makes
///   ``format(forModelType:)`` a total function rather than a priority order.
///   ``sttAndTTSTypesAreDisjoint`` re-checks it so a dependency bump that
///   introduced an overlap fails a test instead of silently mis-routing.
public enum AudioModelRegistry {

    /// Directory `mlx-audio-swift` nests its downloads under, inside whatever
    /// `HubCache.cacheDirectory` it was handed.
    ///
    /// Source of truth: `ModelUtils.resolveOrDownloadModel`, which builds
    /// `cache.cacheDirectory / "mlx-audio" / <repoID with "/" replaced by "_">`.
    /// Note this is NOT the Hugging Face `models--<org>--<name>/snapshots/<sha>/`
    /// layout `ModelLibraryManager.scanHuggingFaceCache` handles, and NOT the
    /// managed `<modelDirectory>/<id>` layout `scan(_:)` handles — hence a
    /// third scan rather than a tweak to either.
    public static let cacheSubdirectory = "mlx-audio"

    /// `model_type` values `STT.loadModel` dispatches on.
    ///
    /// Source of truth: the `switch resolved` in
    /// `Sources/MLXAudioSTT/MLXAudioSTT.swift`. Refresh when bumping the
    /// mlx-audio-swift dependency. Stored lowercased — compared
    /// case-insensitively against `config.json`.
    static let sttModelTypes: Set<String> = [
        "canary",
        "cohere",
        "cohere_asr",
        "fire_red",
        "firered",
        "fireredasr2",
        "glm",
        "glmasr",
        "granite_speech",
        "lasr",
        "lasr_ctc",
        "mms",
        "moonshine",
        "moss_transcribe_diarize",
        "nemotron",
        "nemotron_asr",
        "parakeet",
        "qwen3_asr",
        "sensevoice",
        "voxtral",
        "voxtral_realtime",
        "wav2vec",
        "wav2vec2",
        "whisper",
    ]

    /// `model_type` values `TTS.loadModel` dispatches on.
    ///
    /// Source of truth: the `switch resolvedType` in
    /// `Sources/MLXAudioTTS/TTSModel.swift`. Same refresh discipline as
    /// ``sttModelTypes``.
    ///
    /// Several keys here are GENERIC decoder families (`qwen3`, `qwen`,
    /// `llama`, `llama3`) that a plain chat checkpoint also declares. That is
    /// safe ONLY because this registry is consulted for directories inside
    /// the mlx-audio cache — a location written exclusively by
    /// `ModelUtils.resolveOrDownloadModel`, i.e. only ever as the result of
    /// someone asking mlx-audio to load an audio model. A chat model in the
    /// managed model directory never reaches this code. Do not reuse this set
    /// to classify arbitrary directories.
    static let ttsModelTypes: Set<String> = [
        "chatterbox",
        "chatterbox_tts",
        "chatterbox_turbo",
        "csm",
        "echo",
        "echo_tts",
        "fish_qwen3_omni",
        "fish_speech",
        "index_tts",
        "indextts",
        "irodori",
        "irodori_tts",
        "kitten",
        "kitten_tts",
        "kokoro",
        "kokoro_tts",
        "llama",
        "llama3",
        "llama3_tts",
        "llama_tts",
        "moss_tts",
        "moss_tts_delay",
        "moss_tts_local",
        "moss_tts_nano",
        "omnivoice",
        "orpheus",
        "orpheus_tts",
        "pocket_tts",
        "qwen",
        "qwen3",
        "qwen3_tts",
        "sesame",
        "soprano",
        "soprano_tts",
    ]

    /// Whether the two registries share no key, so a resolved `model_type`
    /// can never be both. Exposed for the test that guards this property
    /// across dependency bumps.
    static var sttAndTTSTypesAreDisjoint: Bool {
        sttModelTypes.isDisjoint(with: ttsModelTypes)
    }

    /// The ``ModelFormat`` a resolved `model_type` maps to, or `nil` when
    /// neither loader would accept it.
    public static func format(forModelType modelType: String?) -> ModelFormat? {
        guard let modelType else { return nil }
        let normalized = modelType.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return nil }
        if sttModelTypes.contains(normalized) { return .audioSTT }
        if ttsModelTypes.contains(normalized) { return .audioTTS }
        return nil
    }

    /// Pull the effective `model_type` out of a decoded `config.json`.
    ///
    /// Mirrors `ModelUtils.resolveModelType`'s key order exactly —
    /// `model_type`, then `architecture`, then `model_version` — because a
    /// disagreement between what we classify and what the loader dispatches
    /// on is precisely the bug this whole path exists to avoid. Upstream's
    /// fourth fallback (guess from the repo name) is intentionally omitted;
    /// see the type-level note.
    public static func modelType(fromConfig config: [String: Any]) -> String? {
        (config["model_type"] as? String)
            ?? (config["architecture"] as? String)
            ?? (config["model_version"] as? String)
    }

    /// Reverse the cache folder naming (`<owner>_<repo>` for repo id
    /// `<owner>/<repo>`) back into the repo id the loaders take.
    ///
    /// This matters more than it looks: ``AudioEngine/loadSTT(_:)`` accepts
    /// ONLY an `owner/name` repo id — it has no bare-local-directory path —
    /// so the recovered repo id, not the directory path, is what a discovered
    /// audio model must carry as its `LocalModel.id` for the GUI to be able
    /// to load it.
    ///
    /// Splits on the FIRST `_`, because `resolveOrDownloadModel` replaces only
    /// the single `/` separator and a Hugging Face *owner* may not contain
    /// `_` while a repo *name* very often does (`mlx-community/Kokoro-82M-4bit`
    /// is unaffected, but e.g. `openai/whisper_large_v3` round-trips correctly
    /// only under first-separator semantics). Returns `nil` for a folder with
    /// no separator, or an empty half — neither can name a real repo.
    public static func repoID(fromCacheFolderName name: String) -> String? {
        guard let separator = name.firstIndex(of: "_") else { return nil }
        let owner = name[name.startIndex..<separator]
        let repo = name[name.index(after: separator)...]
        guard !owner.isEmpty, !repo.isEmpty else { return nil }
        return "\(owner)/\(repo)"
    }
}
