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
            printTable(models, dir: ctx.settings.modelDirectory)
        }
    }

    // MARK: - Plain text table

    private func printTable(_ models: [LocalModel], dir: URL) {
        if models.isEmpty {
            // Display the **actual configured** model directory, not a
            // hard-coded "~/models" guess. Pre-v0.3 this always printed
            // the v0.1 default even if the user had moved the directory
            // in GUI Settings.
            let display = dir.path(percentEncoded: false)
                .replacingOccurrences(
                    of: FileManager.default.homeDirectoryForCurrentUser.path,
                    with: "~"
                )
            print("No models found in \(display)")
            print("Run `macmlx pull <modelID>` to download a model.")
            return
        }

        // Column widths — at least the header width, grow with content
        let nameW = max(20, models.map { $0.displayName.count }.max() ?? 0)
        let sizeW = max(8, models.map { $0.humanSize.count }.max() ?? 0)
        let quantW = max(6, models.map { ($0.quantization ?? "-").count }.max() ?? 0)

        // Use Swift-native padding instead of `String(format: "%-*s", …)` —
        // the C printf `%s` specifier expects a `const char *`, but passing
        // a Swift `String` bridges via NSString's UTF-16 buffer, which on
        // arm64 release builds segfaults non-deterministically (exit 139).
        // Reproduced pre-fix with `macmlx list` on any non-empty model set.
        let header = padLeft("NAME", nameW) + "  " + padRight("SIZE", sizeW) + "  " + padLeft("QUANT", quantW)
        let divider = String(repeating: "-", count: header.count)
        print(header)
        print(divider)
        for m in models {
            let row = padLeft(m.displayName, nameW) + "  "
                + padRight(m.humanSize, sizeW) + "  "
                + padLeft(m.quantization ?? "-", quantW)
            print(row)
        }
    }

    /// Left-align `s` to exactly `width` chars (pads with spaces; truncates
    /// if longer — shouldn't happen here since column widths are derived
    /// from content `.count`).
    private func padLeft(_ s: String, _ width: Int) -> String {
        s.padding(toLength: max(width, s.count), withPad: " ", startingAt: 0)
    }

    /// Right-align `s` to exactly `width` chars (pads with leading spaces).
    private func padRight(_ s: String, _ width: Int) -> String {
        if s.count >= width { return s }
        return String(repeating: " ", count: width - s.count) + s
    }

}
