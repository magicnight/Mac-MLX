// SettingsManager.swift
// MacMLXCore
//
// Persistent user settings stored as JSON at ~/.mac-mlx/settings.json.
// Designed for Swift 6 strict concurrency: the actor serialises all
// reads and writes, and the Settings struct is Sendable.

import Foundation

// MARK: - Settings

/// The full set of user-configurable preferences for macMLX.
public struct Settings: Codable, Equatable, Sendable {
    /// Directory where downloaded model weights are stored.
    public var modelDirectory: URL

    /// Which inference engine to use by default.
    public var preferredEngine: EngineID

    /// Port the embedded OpenAI-compatible HTTP server listens on.
    public var serverPort: Int

    /// Start the HTTP server automatically on launch.
    public var autoStartServer: Bool

    /// Model identifier of the most recently loaded model.
    public var lastLoadedModel: String?

    /// Whether the first-run onboarding flow has been completed.
    public var onboardingComplete: Bool

    /// Path to a custom Python binary (e.g. inside a uv virtual env).
    public var pythonPath: String?

    /// Path to the SwiftLM binary for 100B+ MoE inference.
    public var swiftLMPath: String?

    /// Bearer token required on the HTTP server's `/v1/*` and `/api/*`
    /// routes. `nil` (default) leaves the localhost server open — the dev
    /// default. When set, requests must send `Authorization: Bearer <key>`.
    public var serverAPIKey: String?

    /// Generation stall watchdog timeout in seconds (SRV-4 / issue #29).
    /// If a live generation produces no new chunk for this long, the
    /// server cancels it, releases the generation lock, and fails loudly
    /// (504 non-streaming; an SSE/NDJSON error frame then close for
    /// streaming) instead of hanging the client forever. Stall-based
    /// (inter-chunk gap), NOT total duration — a long generation that
    /// keeps producing tokens is never killed. `0` disables the watchdog.
    /// Default 120s.
    public var generationStallTimeoutSeconds: Int

    /// Sparkle update channel — "release" or "beta".
    public var sparkleUpdateChannel: String

    /// How many days to retain log entries before pruning.
    public var logRetentionDays: Int

    /// Hugging Face Hub endpoint URL. Default: "https://huggingface.co".
    /// Users in regions where huggingface.co is slow or blocked can point
    /// this at a mirror like "https://hf-mirror.com" (#21).
    public var hfEndpoint: String

    /// Hot prompt-cache capacity in megabytes — in-memory only.
    ///
    /// MVP note: `PromptCacheStore`'s `hotCapacity` is an *entry* count,
    /// not a byte budget. We persist the MB value for forward-compat so
    /// a byte-accurate budget can land in v0.4.0.1 without a settings
    /// migration. Today the engine ignores this value and uses the
    /// default 8-entry cap.
    public var kvCacheHotMB: Int

    /// Cold prompt-cache disk cap in gigabytes.
    ///
    /// MVP note: automatic cold-tier pruning is not yet implemented —
    /// rely on Settings → "Clear All KV Caches" to reclaim space. Real
    /// enforcement lands in v0.4.0.1.
    public var kvCacheColdGB: Int

    /// ModelPool byte budget, expressed in gigabytes (Apple's 10^9 GB
    /// convention). When resident models' summed estimated footprint
    /// exceeds this, the pool LRU-evicts non-pinned entries. Default
    /// is 50% of the machine's physical RAM, clamped to a 4 GB floor
    /// for small-memory Macs.
    public var maxResidentMemoryGB: Int

    // MARK: - Speech I/O (v0.6+)

    /// Master toggle for speech features — `false` keeps mic capture
    /// + TTS playback completely off, mirrors the v0.6 first-run UX.
    public var audioEnabled: Bool

    /// Identifier of the STT model to load on demand
    /// (e.g. `whisper-small`, `whisper-medium`, `whisper-large-v3`,
    /// `fun-asr`). Nil means "user hasn't picked one" — the chat
    /// input's mic button surfaces a one-shot picker on first use.
    public var sttModel: String?

    /// Identifier of the TTS model
    /// (e.g. `marvis`, `chatterbox`, `cosyvoice2`). Nil = no TTS
    /// model picked.
    public var ttsModel: String?

    /// Voice id passed to the TTS model. Voice cloning works by
    /// pointing this at a `~/.mac-mlx/audio/voices/<name>.wav`
    /// reference clip. Nil = use the model's default voice.
    public var ttsVoice: String?

    /// Auto-speak completed assistant replies. False (default) keeps
    /// playback opt-in via the per-bubble speaker button.
    public var ttsAutoSpeak: Bool

    // MARK: - Hugging Face cache discovery (Track F)

    /// Opt-in toggle: when `true`, the Model Library also scans
    /// `huggingFaceCacheDirectories` for MLX-format models already cached
    /// by other HF tooling (`transformers`, `huggingface-cli`, …) and
    /// lists them alongside the app's own managed models — referencing
    /// the cache in place, never copying. Default `false`: scanning a
    /// directory the user didn't explicitly point macMLX at is an
    /// opt-in, not a surprise.
    public var scanHuggingFaceCache: Bool

    /// Cache root directories to scan when `scanHuggingFaceCache` is on.
    /// User-editable (Settings → Hugging Face Cache) so a custom
    /// `HF_HOME`/`HF_HUB_CACHE` location can be added. Seeded with the
    /// standard `~/.cache/huggingface/hub` path by default; persisted
    /// independent of the toggle so a customised list survives
    /// switching the toggle off and back on.
    public var huggingFaceCacheDirectories: [URL]

    // MARK: Factory

    /// Sensible out-of-the-box defaults — used when no settings file exists.
    ///
    /// See `DataRoot.macMLX` for the rationale behind the dotfile-exempt
    /// path choice under App Sandbox.
    public static let `default`: Settings = .init(
        modelDirectory: DataRoot.macMLX("models"),
        preferredEngine: .mlxSwift,
        serverPort: 8000,
        autoStartServer: false,
        lastLoadedModel: nil,
        onboardingComplete: false,
        pythonPath: nil,
        swiftLMPath: nil,
        serverAPIKey: nil,
        generationStallTimeoutSeconds: 120,
        sparkleUpdateChannel: "release",
        logRetentionDays: 7,
        hfEndpoint: "https://huggingface.co",
        kvCacheHotMB: 512,
        kvCacheColdGB: 20,
        maxResidentMemoryGB: max(4, Int(MemoryProbe.totalMemoryGB()) / 2),
        audioEnabled: false,
        sttModel: nil,
        ttsModel: nil,
        ttsVoice: nil,
        ttsAutoSpeak: false,
        scanHuggingFaceCache: false,
        huggingFaceCacheDirectories: [Self.defaultHuggingFaceCacheDirectory]
    )

    /// Standard Hugging Face Hub cache location — `~/.cache/huggingface/hub`
    /// (`HF_HOME`/hub). Not read from the `HF_HOME` environment variable:
    /// keeping this a fixed default (rather than following an env var that
    /// may or may not be set in the GUI app's process environment) keeps
    /// the seed predictable; a user with a custom `HF_HOME` adds it via the
    /// Settings directory editor.
    public static var defaultHuggingFaceCacheDirectory: URL {
        DataRoot.userHome.appending(
            path: ".cache/huggingface/hub",
            directoryHint: .isDirectory
        )
    }

    // MARK: Init

    public init(
        modelDirectory: URL,
        preferredEngine: EngineID,
        serverPort: Int,
        autoStartServer: Bool,
        lastLoadedModel: String?,
        onboardingComplete: Bool,
        pythonPath: String?,
        swiftLMPath: String?,
        serverAPIKey: String? = nil,
        generationStallTimeoutSeconds: Int = 120,
        sparkleUpdateChannel: String,
        logRetentionDays: Int,
        hfEndpoint: String = "https://huggingface.co",
        kvCacheHotMB: Int = 512,
        kvCacheColdGB: Int = 20,
        maxResidentMemoryGB: Int = max(4, Int(MemoryProbe.totalMemoryGB()) / 2),
        audioEnabled: Bool = false,
        sttModel: String? = nil,
        ttsModel: String? = nil,
        ttsVoice: String? = nil,
        ttsAutoSpeak: Bool = false,
        scanHuggingFaceCache: Bool = false,
        huggingFaceCacheDirectories: [URL] = [Settings.defaultHuggingFaceCacheDirectory]
    ) {
        self.modelDirectory = modelDirectory
        self.preferredEngine = preferredEngine
        self.serverPort = serverPort
        self.autoStartServer = autoStartServer
        self.lastLoadedModel = lastLoadedModel
        self.onboardingComplete = onboardingComplete
        self.pythonPath = pythonPath
        self.swiftLMPath = swiftLMPath
        self.serverAPIKey = serverAPIKey
        self.generationStallTimeoutSeconds = generationStallTimeoutSeconds
        self.sparkleUpdateChannel = sparkleUpdateChannel
        self.logRetentionDays = logRetentionDays
        self.hfEndpoint = hfEndpoint
        self.kvCacheHotMB = kvCacheHotMB
        self.kvCacheColdGB = kvCacheColdGB
        self.maxResidentMemoryGB = maxResidentMemoryGB
        self.audioEnabled = audioEnabled
        self.sttModel = sttModel
        self.ttsModel = ttsModel
        self.ttsVoice = ttsVoice
        self.ttsAutoSpeak = ttsAutoSpeak
        self.scanHuggingFaceCache = scanHuggingFaceCache
        self.huggingFaceCacheDirectories = huggingFaceCacheDirectories
    }

    // MARK: - Codable (backward-compat decode)

    /// Pre-v0.4 settings files don't have `kvCacheHotMB` /
    /// `kvCacheColdGB` — decode them as optionals and fall back to the
    /// defaults so existing installs keep working across upgrades.
    private enum CodingKeys: String, CodingKey {
        case modelDirectory
        case preferredEngine
        case serverPort
        case autoStartServer
        case lastLoadedModel
        case onboardingComplete
        case pythonPath
        case swiftLMPath
        case serverAPIKey
        case generationStallTimeoutSeconds
        case sparkleUpdateChannel
        case logRetentionDays
        case hfEndpoint
        case kvCacheHotMB
        case kvCacheColdGB
        case maxResidentMemoryGB
        case audioEnabled
        case sttModel
        case ttsModel
        case ttsVoice
        case ttsAutoSpeak
        case scanHuggingFaceCache
        case huggingFaceCacheDirectories
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelDirectory = try c.decode(URL.self, forKey: .modelDirectory)
        self.preferredEngine = try c.decode(EngineID.self, forKey: .preferredEngine)
        self.serverPort = try c.decode(Int.self, forKey: .serverPort)
        self.autoStartServer = try c.decode(Bool.self, forKey: .autoStartServer)
        self.lastLoadedModel = try c.decodeIfPresent(String.self, forKey: .lastLoadedModel)
        self.onboardingComplete = try c.decode(Bool.self, forKey: .onboardingComplete)
        self.pythonPath = try c.decodeIfPresent(String.self, forKey: .pythonPath)
        self.swiftLMPath = try c.decodeIfPresent(String.self, forKey: .swiftLMPath)
        self.serverAPIKey = try c.decodeIfPresent(String.self, forKey: .serverAPIKey)
        self.generationStallTimeoutSeconds =
            try c.decodeIfPresent(Int.self, forKey: .generationStallTimeoutSeconds) ?? 120
        self.sparkleUpdateChannel = try c.decode(String.self, forKey: .sparkleUpdateChannel)
        self.logRetentionDays = try c.decode(Int.self, forKey: .logRetentionDays)
        self.hfEndpoint = try c.decodeIfPresent(String.self, forKey: .hfEndpoint)
            ?? "https://huggingface.co"
        self.kvCacheHotMB = try c.decodeIfPresent(Int.self, forKey: .kvCacheHotMB) ?? 512
        self.kvCacheColdGB = try c.decodeIfPresent(Int.self, forKey: .kvCacheColdGB) ?? 20
        self.maxResidentMemoryGB =
            (try c.decodeIfPresent(Int.self, forKey: .maxResidentMemoryGB))
            ?? max(4, Int(MemoryProbe.totalMemoryGB()) / 2)
        // v0.6 audio fields — pre-v0.6 settings.json files don't carry
        // them. Fall back to "audio off" so existing installs upgrade
        // without surprise mic permission prompts.
        self.audioEnabled = try c.decodeIfPresent(Bool.self, forKey: .audioEnabled) ?? false
        self.sttModel = try c.decodeIfPresent(String.self, forKey: .sttModel)
        self.ttsModel = try c.decodeIfPresent(String.self, forKey: .ttsModel)
        self.ttsVoice = try c.decodeIfPresent(String.self, forKey: .ttsVoice)
        self.ttsAutoSpeak = try c.decodeIfPresent(Bool.self, forKey: .ttsAutoSpeak) ?? false
        // Track F — pre-existing settings.json files have neither key;
        // default to "off" with the standard cache path pre-seeded so
        // flipping the toggle on later has something sensible to scan.
        self.scanHuggingFaceCache =
            try c.decodeIfPresent(Bool.self, forKey: .scanHuggingFaceCache) ?? false
        self.huggingFaceCacheDirectories =
            try c.decodeIfPresent([URL].self, forKey: .huggingFaceCacheDirectories)
            ?? [Settings.defaultHuggingFaceCacheDirectory]
    }
}

// MARK: - SettingsManager

/// Reads and writes `Settings` to a JSON file on disk.
///
/// All mutations are serialised by the actor; callers are safe to call
/// `update` or `replace` from multiple concurrent contexts.
public actor SettingsManager {

    // MARK: - State

    /// Most recently loaded (or default) settings.
    public private(set) var current: Settings

    /// The URL of the JSON file we persist to.
    private let fileURL: URL

    // MARK: - Init

    /// Production initialiser — persists to `~/.mac-mlx/settings.json`.
    ///
    /// Uses `DataRoot.macMLX` (real `$HOME`, not the sandbox container)
    /// so GUI and CLI processes converge on the same `settings.json`.
    /// Before v0.3 this used `FileManager.default.homeDirectoryForCurrentUser`
    /// which under sandbox wrote to `~/Library/Containers/<bundle-id>/Data/…`,
    /// splitting settings between GUI and CLI surfaces.
    public init() {
        self.fileURL = DataRoot.macMLX.appending(
            path: "settings.json",
            directoryHint: URL.DirectoryHint.notDirectory
        )
        self.current = .default
    }

    /// Test / preview initialiser — caller controls the backing file URL.
    public init(fileURL: URL) {
        self.fileURL = fileURL
        self.current = .default
    }

    // MARK: - Bootstrap

    /// Must be called once after init to load settings from disk.
    ///
    /// Separated from init because actor initialisers cannot throw.
    ///
    /// Policy:
    /// - File missing → write defaults so the file exists next time.
    /// - File present but corrupt → use defaults in memory, but do **not**
    ///   overwrite the file (leave it for human inspection).
    ///
    /// - Returns: The loaded (or default) settings.
    @discardableResult
    public func load() async -> Settings {
        if FileManager.default.fileExists(atPath: fileURL.path) {
            // Present — attempt decode; on corruption keep the file intact.
            if let loaded = try? readFromDisk() {
                current = loaded
            } else {
                current = .default
            }
        } else {
            // Missing — persist defaults so the file exists next time.
            current = .default
            try? writeToDisk(current)
        }
        return current
    }

    // MARK: - Public API

    /// Re-read the file from disk and update `current`.
    ///
    /// Falls back to `.default` if the file is missing or corrupt.
    /// On corruption the file is **not** overwritten.
    @discardableResult
    public func reload() async throws -> Settings {
        if let loaded = try? readFromDisk() {
            current = loaded
        }
        // If readFromDisk() throws/returns nil, leave current as-is.
        return current
    }

    /// Mutate the current settings via a closure, then persist to disk.
    ///
    /// `mutate` is `@Sendable` so cross-actor callers (e.g. a `@MainActor`
    /// view model) can pass closures into this `actor` method without tripping
    /// Swift 6 strict concurrency.
    public func update(_ mutate: @Sendable (inout Settings) -> Void) throws {
        var copy = current
        mutate(&copy)
        try writeToDisk(copy)
        current = copy
    }

    /// Replace settings wholesale and persist to disk.
    public func replace(_ new: Settings) throws {
        try writeToDisk(new)
        current = new
    }

    // MARK: - Private helpers

    private func readFromDisk() throws -> Settings? {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return nil
        }
        let data = try Data(contentsOf: fileURL)
        return try JSONDecoder().decode(Settings.self, from: data)
    }

    private func writeToDisk(_ settings: Settings) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(settings)

        // Ensure parent directory exists.
        let dir = fileURL.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(
                at: dir,
                withIntermediateDirectories: true
            )
        }

        try data.write(to: fileURL, options: .atomic)
    }
}
