import ArgumentParser
import MacMLXCore

@main
struct MacmlxCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "macmlx",
        abstract: "Native macOS LLM inference CLI.",
        version: coreVersion,
        subcommands: [
            ServeCommand.self,
            PullCommand.self,
            RunCommand.self,
            ListCommand.self,
            PSCommand.self,
            StopCommand.self,
        ]
    )

    /// Exposed for tests; mirrors `MacMLXCore.version`.
    static let coreVersion = MacMLXCore.version
}
