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
