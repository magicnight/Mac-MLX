import ArgumentParser
import MacMLXCore

@main
struct MacmlxCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "macmlx",
        abstract: "Native macOS LLM inference CLI.",
        version: coreVersion
    )

    /// Exposed for tests; mirrors `MacMLXCore.version`.
    static let coreVersion = MacMLXCore.version

    func run() async throws {
        // Stage 1 stub — real subcommands land in Stage 5.
        print("macmlx \(MacmlxCommand.coreVersion)")
        print("No subcommand wired yet. See `macmlx --help`.")
    }
}
