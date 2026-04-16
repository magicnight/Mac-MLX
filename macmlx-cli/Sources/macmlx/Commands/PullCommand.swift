import ArgumentParser
import Foundation
import MacMLXCore

struct PullCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a model from Hugging Face into the model directory."
    )

    @Argument(help: "Hugging Face model ID (e.g. mlx-community/Qwen3-8B-4bit).")
    var modelID: String

    @Option(help: "Override destination directory (default: settings.modelDirectory).")
    var dir: String?

    func run() async throws {
        let ctx = try await CLIContext.bootstrap()
        let target: URL
        if let dirPath = dir {
            target = URL(filePath: dirPath)
        } else {
            target = ctx.settings.modelDirectory
        }

        // v0.1: always uses plain stdout (TUI deferred to v0.2).
        try await PullDashboard.run(
            modelID: modelID,
            target: target,
            downloader: ctx.downloader
        )
    }
}
