import Foundation
import MacMLXCore

/// Bootstrapped per CLI invocation. Loads settings synchronously then exposes
/// the Core actors. Avoids the @MainActor AppState type since CLI doesn't
/// use SwiftUI.
public struct CLIContext: Sendable {
    public let settings: Settings
    public let library: ModelLibraryManager
    public let downloader: HFDownloader

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
            downloader: downloader
        )
    }
}
