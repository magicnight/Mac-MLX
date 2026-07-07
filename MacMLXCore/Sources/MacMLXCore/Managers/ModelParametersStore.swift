// ModelParametersStore.swift
// MacMLXCore
//
// Per-model sampling & prompt overrides.
//
// v0.2 Batch 1 scope (issue #15):
// - MVP knobs: temperature, topP, maxTokens, systemPrompt — the four
//   parameters users actually change day-to-day.
// - JSON per model at `<directory>/{model-id}.json`. Model IDs can contain
//   slashes (e.g. "mlx-community/Qwen3-8B-4bit") so we URL-percent-encode
//   when deriving the filename.
// - Deferred: topK, minP, repetitionPenalty, contextLength, modelAlias,
//   ttlMinutes, pinInMemory, trustRemoteCode. Spec captures those under
//   .claude/features/parameters.md — land them in a follow-up issue.

import Foundation

// MARK: - ModelParameters

/// Per-model generation parameter overrides.
public struct ModelParameters: Codable, Hashable, Sendable {
    /// Sampling temperature. Range [0, 2]. 0.7 is a good default.
    public var temperature: Double
    /// Nucleus (top-p) sampling cutoff. Range (0, 1].
    public var topP: Double
    /// Maximum generation length in tokens.
    public var maxTokens: Int
    /// System prompt prepended to every generation.
    public var systemPrompt: String
    /// Optional LoRA adapter name (folder under
    /// `~/.mac-mlx/adapters/<name>/`) to apply on model load (v0.5+).
    /// Empty string is treated identically to `nil` and means "no
    /// adapter" — matches how the parameters inspector represents
    /// the "None" pick.
    public var adapterName: String?
    /// Optional user-facing alias for this model (v0.5.1). When set,
    /// `GET /v1/models` reports the alias as the model `id`, and
    /// chat / messages / completions requests may name the model by
    /// this alias in addition to its on-disk directory id. Empty
    /// string is treated identically to `nil` ("no alias").
    public var alias: String?
    /// Optional idle time-to-live in seconds (v0.5.1). When set, the
    /// model pool unloads this model once it has been idle longer than
    /// `ttlSeconds`, even while inside the byte budget (pinned models
    /// are exempt). `nil` means "never idle-unload".
    public var ttlSeconds: Int?
    /// Optional free-form chat-template kwargs (v0.5.1) forwarded to the
    /// Jinja chat template as `additionalContext` — e.g.
    /// `{"enable_thinking": true}` for Qwen3. `nil` means "no extra
    /// context".
    public var templateKwargs: [String: JSONValue]?

    public init(
        temperature: Double = 0.7,
        topP: Double = 0.95,
        maxTokens: Int = 2048,
        systemPrompt: String = "You are a helpful assistant.",
        adapterName: String? = nil,
        alias: String? = nil,
        ttlSeconds: Int? = nil,
        templateKwargs: [String: JSONValue]? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.systemPrompt = systemPrompt
        self.adapterName = adapterName
        self.alias = alias
        self.ttlSeconds = ttlSeconds
        self.templateKwargs = templateKwargs
    }

    /// Backwards-compatible decoder. Pre-v0.5 JSON has no
    /// `adapterName` key, and pre-v0.5.1 JSON has no `alias` /
    /// `ttlSeconds` / `templateKwargs` keys — default to nil so
    /// existing per-model override files load unchanged.
    private enum CodingKeys: String, CodingKey {
        case temperature, topP, maxTokens, systemPrompt, adapterName
        case alias, ttlSeconds, templateKwargs
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.temperature = try c.decode(Double.self, forKey: .temperature)
        self.topP = try c.decode(Double.self, forKey: .topP)
        self.maxTokens = try c.decode(Int.self, forKey: .maxTokens)
        self.systemPrompt = try c.decode(String.self, forKey: .systemPrompt)
        self.adapterName = try c.decodeIfPresent(String.self, forKey: .adapterName)
        self.alias = try c.decodeIfPresent(String.self, forKey: .alias)
        self.ttlSeconds = try c.decodeIfPresent(Int.self, forKey: .ttlSeconds)
        self.templateKwargs = try c.decodeIfPresent([String: JSONValue].self, forKey: .templateKwargs)
    }

    /// Factory for the factory defaults — handy in "Reset" buttons.
    public static let `default` = ModelParameters()

    /// Convenience adapter to the Stage-2 GenerateRequest parameter shape.
    public func asGenerationParameters() -> GenerationParameters {
        GenerationParameters(
            temperature: temperature,
            topP: topP,
            maxTokens: maxTokens,
            stream: true
        )
    }
}

// MARK: - Store

/// Actor-serialised load/save of `ModelParameters` keyed by HF-style
/// model ID ("mlx-community/Qwen3-8B-4bit" and similar).
public actor ModelParametersStore {

    private let directory: URL
    private let fileManager: FileManager

    /// Default directory: `~/.mac-mlx/model-params/` (real home, dotfile
    /// exemption applies under sandbox — see `DataRoot.macMLX`).
    public init(directory: URL? = nil, fileManager: FileManager = .default) {
        self.directory = directory ?? DataRoot.macMLX("model-params")
        self.fileManager = fileManager
    }

    // MARK: - Public API

    /// Load the stored overrides for `modelID`. Returns `.default` if no
    /// file exists or the file is unreadable / corrupt.
    public func load(for modelID: String) async -> ModelParameters {
        let url = fileURL(for: modelID)
        guard let data = try? Data(contentsOf: url) else {
            return .default
        }
        let decoder = JSONDecoder()
        return (try? decoder.decode(ModelParameters.self, from: data)) ?? .default
    }

    /// Save `parameters` for `modelID`. Atomic write; creates parent dir
    /// on demand.
    public func save(_ parameters: ModelParameters, for modelID: String) async throws {
        try ensureDirectory()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(parameters)
        try data.write(to: fileURL(for: modelID), options: .atomic)
    }

    /// Remove stored overrides for `modelID`. Idempotent.
    public func reset(for modelID: String) async {
        try? fileManager.removeItem(at: fileURL(for: modelID))
    }

    /// Scan the params directory for an override whose `alias` equals
    /// `alias`, returning that override's model ID (the decoded
    /// filename). Returns `nil` when no override declares this alias.
    /// Lets the server resolve a chat / messages / completions request
    /// that names a model by its user-facing alias (v0.5.1). An empty
    /// `alias` never matches — empty is "no alias".
    public func modelID(forAlias alias: String) async -> String? {
        guard !alias.isEmpty else { return nil }
        guard let files = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ) else {
            return nil
        }
        let decoder = JSONDecoder()
        for url in files where url.pathExtension == "json" {
            guard let data = try? Data(contentsOf: url),
                  let params = try? decoder.decode(ModelParameters.self, from: data),
                  params.alias == alias else {
                continue
            }
            // Recover the model ID from the percent-encoded filename.
            let encoded = url.deletingPathExtension().lastPathComponent
            return encoded.removingPercentEncoding ?? encoded
        }
        return nil
    }

    // MARK: - Private

    private func fileURL(for modelID: String) -> URL {
        // Model IDs contain "/" which would break the filesystem; encode.
        let encoded = modelID.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed.subtracting(.init(charactersIn: "/"))
        ) ?? modelID
        return directory.appending(path: "\(encoded).json", directoryHint: .notDirectory)
    }

    private func ensureDirectory() throws {
        if !fileManager.fileExists(atPath: directory.path) {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }
}
