import ArgumentParser
import Foundation
import MacMLXCore

struct ListCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List downloaded models in the configured model directory."
    )

    @Flag(help: "Output as JSON for scripting.")
    var json = false

    func run() async throws {
        let ctx = try await CLIContext.bootstrap()
        let models: [LocalModel]
        do {
            models = try await ctx.library.scan(ctx.settings.modelDirectory)
        } catch {
            // Directory may not exist yet — treat as empty
            models = []
        }

        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(models)
            FileHandle.standardOutput.write(data)
            print()
        } else {
            printTable(models)
        }
    }

    // MARK: - Plain text table

    private func printTable(_ models: [LocalModel]) {
        if models.isEmpty {
            print("No models found in \(Self.modelDirDescription())")
            print("Run `macmlx pull <modelID>` to download a model.")
            return
        }

        // Column widths — at least the header width, grow with content
        let nameW = max(20, models.map { $0.displayName.count }.max() ?? 0)
        let sizeW = max(8, models.map { $0.humanSize.count }.max() ?? 0)
        let quantW = max(6, models.map { ($0.quantization ?? "-").count }.max() ?? 0)

        let header = String(format: "%-*s  %*s  %-*s",
            nameW, "NAME",
            sizeW, "SIZE",
            quantW, "QUANT")
        let divider = String(repeating: "-", count: header.count)
        print(header)
        print(divider)
        for m in models {
            let row = String(format: "%-*s  %*s  %-*s",
                nameW, m.displayName,
                sizeW, m.humanSize,
                quantW, m.quantization ?? "-")
            print(row)
        }
    }

    private static func modelDirDescription() -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let defaultDir = home.appending(path: "models")
        return defaultDir.path(percentEncoded: false)
    }
}
