import Foundation
import MacMLXCore

/// Bootstrapped per CLI invocation. Loads settings synchronously then exposes
/// the Core actors. Avoids the @MainActor AppState type since CLI doesn't
/// use SwiftUI.
public struct CLIContext: Sendable {
    public let settings: Settings
    public let library: ModelLibraryManager
    public let downloader: HFDownloader
    /// Shared per-model parameter overrides store (temperature, topP,
    /// maxTokens, systemPrompt). CLI commands layer explicit `--flag`
    /// values on top of these persisted values so a user who set
    /// `temperature=0.3` for `Qwen3-8B-4bit` in the GUI Parameters
    /// Inspector sees the same default in `macmlx run Qwen3-8B-4bit`.
    public let paramStore: ModelParametersStore

    public static func bootstrap() async throws -> CLIContext {
        let mgr = SettingsManager()
        let s = await mgr.load()
        let downloader = HFDownloader()
        // Honour the user's configured Hugging Face endpoint (#21). Without
        // this, `macmlx pull` hit huggingface.co even when the GUI had the
        // user pointed at a mirror like https://hf-mirror.com — a
        // regression surfaced by the v0.2 gap review.
        if let endpointURL = URL(string: s.hfEndpoint), endpointURL.host != nil {
            await downloader.setBaseURL(endpointURL)
        }
        return CLIContext(
            settings: s,
            library: ModelLibraryManager(),
            downloader: downloader,
            paramStore: ModelParametersStore()
        )
    }

    /// Create an engine matching the user's `settings.preferredEngine`.
    /// v0.3 only ships the MLX Swift engine; selecting a deferred engine
    /// (SwiftLM / Python mlx-lm) surfaces the same error the GUI shows
    /// — same behaviour in both surfaces so users don't get surprised.
    public func makeEngine() throws -> any InferenceEngine {
        switch settings.preferredEngine {
        case .mlxSwift:
            return MLXSwiftEngine()
        case .swiftLM:
            throw CLIError(
                "SwiftLM engine is not available in this build (deferred — see issue #12). Pick MLX Swift in the GUI Settings or remove the override."
            )
        case .pythonMLX:
            throw CLIError(
                "Python mlx-lm engine is not available in this build (deferred — see issue #13). Pick MLX Swift in the GUI Settings or remove the override."
            )
        }
    }

    /// Resolve the effective generation parameters for `modelID` by
    /// layering explicit CLI flags over the persisted `ModelParameters`
    /// for that model. A flag that's `nil` (i.e. the user didn't pass
    /// `--temperature` / `--max-tokens` / `--system`) falls back to the
    /// persisted value, which in turn falls back to `ModelParameters.default`.
    public func resolveParameters(
        for modelID: String,
        explicitTemperature: Double?,
        explicitMaxTokens: Int?,
        explicitSystem: String?,
        stream: Bool
    ) async -> (GenerationParameters, String?) {
        let persisted = await paramStore.load(for: modelID)
        let params = GenerationParameters(
            temperature: explicitTemperature ?? persisted.temperature,
            topP: persisted.topP,
            maxTokens: explicitMaxTokens ?? persisted.maxTokens,
            stream: stream
        )
        // System prompt: CLI flag wins; empty persisted string means "none"
        let system: String?
        if let explicitSystem {
            system = explicitSystem
        } else if persisted.systemPrompt.isEmpty {
            system = nil
        } else {
            system = persisted.systemPrompt
        }
        return (params, system)
    }
}

// MARK: - CLIError

/// Lightweight Error for CLI-side pre-flight checks. Prints the
/// `description` via ArgumentParser's default error path.
public struct CLIError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}
