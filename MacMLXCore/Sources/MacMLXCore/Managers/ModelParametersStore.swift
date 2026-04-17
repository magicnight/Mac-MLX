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

    public init(
        temperature: Double = 0.7,
        topP: Double = 0.95,
        maxTokens: Int = 2048,
        systemPrompt: String = "You are a helpful assistant."
    ) {
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.systemPrompt = systemPrompt
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
    /// exemption applies under sandbox).
    public init(directory: URL? = nil, fileManager: FileManager = .default) {
        if let directory {
            self.directory = directory
        } else {
            let home: URL = {
                if let path = NSHomeDirectoryForUser(NSUserName()) {
                    return URL(filePath: path, directoryHint: .isDirectory)
                }
                return URL(filePath: NSHomeDirectory(), directoryHint: .isDirectory)
            }()
            self.directory = home.appending(
                path: ".mac-mlx/model-params",
                directoryHint: .isDirectory
            )
        }
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
