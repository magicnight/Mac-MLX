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
        sparkleUpdateChannel: "release",
        logRetentionDays: 7,
        hfEndpoint: "https://huggingface.co",
        kvCacheHotMB: 512,
        kvCacheColdGB: 20
    )

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
        sparkleUpdateChannel: String,
        logRetentionDays: Int,
        hfEndpoint: String = "https://huggingface.co",
        kvCacheHotMB: Int = 512,
        kvCacheColdGB: Int = 20
    ) {
        self.modelDirectory = modelDirectory
        self.preferredEngine = preferredEngine
        self.serverPort = serverPort
        self.autoStartServer = autoStartServer
        self.lastLoadedModel = lastLoadedModel
        self.onboardingComplete = onboardingComplete
        self.pythonPath = pythonPath
        self.swiftLMPath = swiftLMPath
        self.sparkleUpdateChannel = sparkleUpdateChannel
        self.logRetentionDays = logRetentionDays
        self.hfEndpoint = hfEndpoint
        self.kvCacheHotMB = kvCacheHotMB
        self.kvCacheColdGB = kvCacheColdGB
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
        case sparkleUpdateChannel
        case logRetentionDays
        case hfEndpoint
        case kvCacheHotMB
        case kvCacheColdGB
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
        self.sparkleUpdateChannel = try c.decode(String.self, forKey: .sparkleUpdateChannel)
        self.logRetentionDays = try c.decode(Int.self, forKey: .logRetentionDays)
        self.hfEndpoint = try c.decodeIfPresent(String.self, forKey: .hfEndpoint)
            ?? "https://huggingface.co"
        self.kvCacheHotMB = try c.decodeIfPresent(Int.self, forKey: .kvCacheHotMB) ?? 512
        self.kvCacheColdGB = try c.decodeIfPresent(Int.self, forKey: .kvCacheColdGB) ?? 20
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
