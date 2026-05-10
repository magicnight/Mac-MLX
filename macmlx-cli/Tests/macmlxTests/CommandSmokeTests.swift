import Testing
@testable import macmlx
import MacMLXCore

@Test
func cliBundlesCoreVersion() {
    // The CLI must surface the same version string as Core.
    #expect(MacmlxCommand.coreVersion == MacMLXCore.version)
}

@Test
func cliConfigurationHasExpectedCommandName() {
    #expect(MacmlxCommand.configuration.commandName == "macmlx")
}

@Test
func cliHasSevenSubcommands() {
    // serve, pull, run, list, ps, stop, mcp (added in v0.4.0)
    #expect(MacmlxCommand.configuration.subcommands.count == 7)
}

@Test
func cliSubcommandsIncludeAllExpected() {
    let names = MacmlxCommand.configuration.subcommands.map { $0.configuration.commandName }
    #expect(names.contains("serve"))
    #expect(names.contains("pull"))
    #expect(names.contains("run"))
    #expect(names.contains("list"))
    #expect(names.contains("ps"))
    #expect(names.contains("stop"))
    #expect(names.contains("mcp"))
}

@Test
func listCommandName() {
    #expect(ListCommand.configuration.commandName == "list")
}

@Test
func psCommandName() {
    #expect(PSCommand.configuration.commandName == "ps")
}

@Test
func serveCommandName() {
    #expect(ServeCommand.configuration.commandName == "serve")
}

@Test
func stopCommandName() {
    #expect(StopCommand.configuration.commandName == "stop")
}

@Test
func pullCommandName() {
    #expect(PullCommand.configuration.commandName == "pull")
}

@Test
func runCommandName() {
    #expect(RunCommand.configuration.commandName == "run")
}

@Test
func mcpCommandName() {
    #expect(MCPCommand.configuration.commandName == "mcp")
}

@Test
func mcpHasServeSubcommand() {
    let names = MCPCommand.configuration.subcommands.map { $0.configuration.commandName }
    #expect(names.contains("serve"))
}
