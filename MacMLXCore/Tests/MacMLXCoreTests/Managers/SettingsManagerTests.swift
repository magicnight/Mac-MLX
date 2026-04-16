// SettingsManagerTests.swift
// MacMLXCoreTests

import Foundation
import Testing
@testable import MacMLXCore

// MARK: - Helpers

private func makeTempSettingsURL() -> URL {
    FileManager.default.temporaryDirectory
        .appending(path: "settings-test-\(UUID().uuidString).json")
}

// MARK: - Tests

@Suite("SettingsManager")
struct SettingsManagerTests {

    // MARK: Missing file

    @Test("defaults loaded and file written when file is missing")
    func defaultsLoadedWhenFileMissing() async throws {
        let url = makeTempSettingsURL()
        // Confirm the file does not exist beforehand.
        #expect(!FileManager.default.fileExists(atPath: url.path))

        let manager = SettingsManager(fileURL: url)
        await manager.load()

        // After load, current should equal the defaults.
        let current = await manager.current
        #expect(current == Settings.default)

        // The file should now exist (defaults were persisted).
        #expect(FileManager.default.fileExists(atPath: url.path))
    }

    // MARK: Round-trip

    @Test("settings persist and reload correctly across instances")
    func roundTripPersistsAcrossInstances() async throws {
        let url = makeTempSettingsURL()

        // Manager A: mutate the port and persist.
        let managerA = SettingsManager(fileURL: url)
        await managerA.load()
        try await managerA.update { $0.serverPort = 9999 }

        // Manager B: read from the same file.
        let managerB = SettingsManager(fileURL: url)
        await managerB.load()
        let current = await managerB.current
        #expect(current.serverPort == 9999)
    }

    // MARK: Corrupt file

    @Test("corrupt file falls back to defaults without overwriting it")
    func corruptFileFallsBackToDefaults() async throws {
        let url = makeTempSettingsURL()

        // Write garbage to the file.
        let garbage = "{ this is not valid json !!!".data(using: .utf8)!
        try garbage.write(to: url)

        let manager = SettingsManager(fileURL: url)
        await manager.load()

        let current = await manager.current
        #expect(current == Settings.default)

        // The corrupt file should still be there, untouched.
        let stillGarbage = try Data(contentsOf: url)
        #expect(stillGarbage == garbage)
    }

    // MARK: Update closure

    @Test("update closure mutates settings and persists the change")
    func updateClosureMutatesAndPersists() async throws {
        let url = makeTempSettingsURL()

        let manager = SettingsManager(fileURL: url)
        await manager.load()

        try await manager.update { $0.onboardingComplete = true }

        // Decode the raw file to verify the field was persisted.
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode(Settings.self, from: data)
        #expect(decoded.onboardingComplete == true)

        // Also verify the in-memory snapshot was updated.
        let current = await manager.current
        #expect(current.onboardingComplete == true)
    }

    // MARK: Replace

    @Test("replace replaces settings wholesale")
    func replaceWritesNewSettings() async throws {
        let url = makeTempSettingsURL()

        let manager = SettingsManager(fileURL: url)
        await manager.load()

        var newSettings = Settings.default
        newSettings.serverPort = 7777
        newSettings.sparkleUpdateChannel = "beta"

        try await manager.replace(newSettings)

        let current = await manager.current
        #expect(current.serverPort == 7777)
        #expect(current.sparkleUpdateChannel == "beta")
    }
}
