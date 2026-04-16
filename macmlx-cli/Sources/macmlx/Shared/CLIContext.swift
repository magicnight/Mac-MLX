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
        return CLIContext(
            settings: s,
            library: ModelLibraryManager(),
            downloader: HFDownloader()
        )
    }
}
