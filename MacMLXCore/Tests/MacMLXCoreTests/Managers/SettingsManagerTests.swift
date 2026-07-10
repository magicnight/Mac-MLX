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

    @Test("serverAPIKey persists and reloads across instances")
    func serverAPIKeyRoundTrips() async throws {
        let url = makeTempSettingsURL()

        let managerA = SettingsManager(fileURL: url)
        await managerA.load()
        try await managerA.update { $0.serverAPIKey = "sk-test-123" }

        let managerB = SettingsManager(fileURL: url)
        await managerB.load()
        let current = await managerB.current
        #expect(current.serverAPIKey == "sk-test-123")
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

    // MARK: Hugging Face cache discovery (Track F)

    @Test("HF cache scanning defaults to off, seeded with the standard cache path")
    func hfCacheDefaults() async throws {
        let url = makeTempSettingsURL()
        let manager = SettingsManager(fileURL: url)
        let current = await manager.load()

        #expect(current.scanHuggingFaceCache == false)
        #expect(current.huggingFaceCacheDirectories == [Settings.defaultHuggingFaceCacheDirectory])
    }

    @Test("HF cache toggle and directory list persist and reload correctly")
    func hfCacheSettingsRoundTrip() async throws {
        let url = makeTempSettingsURL()

        let managerA = SettingsManager(fileURL: url)
        await managerA.load()
        let customDirs = [
            URL(filePath: "/Volumes/External/hf-cache", directoryHint: .isDirectory),
            Settings.defaultHuggingFaceCacheDirectory,
        ]
        try await managerA.update {
            $0.scanHuggingFaceCache = true
            $0.huggingFaceCacheDirectories = customDirs
        }

        let managerB = SettingsManager(fileURL: url)
        await managerB.load()
        let current = await managerB.current
        #expect(current.scanHuggingFaceCache == true)
        #expect(current.huggingFaceCacheDirectories == customDirs)
    }

    @Test("legacy settings.json missing the HF cache keys decodes with safe defaults")
    func hfCacheFieldsDefaultForLegacyFile() async throws {
        let url = makeTempSettingsURL()
        // A settings.json predating Track F — none of the HF cache keys.
        let legacy = """
        {
            "modelDirectory": "file:///Users/test/.mac-mlx/models/",
            "preferredEngine": "mlx-swift-lm",
            "serverPort": 8000,
            "autoStartServer": false,
            "onboardingComplete": true,
            "sparkleUpdateChannel": "release",
            "logRetentionDays": 7
        }
        """
        try Data(legacy.utf8).write(to: url)

        let manager = SettingsManager(fileURL: url)
        let current = await manager.load()
        // Confirms the *legacy JSON decoded successfully* (back-compat path)
        // rather than silently falling back to `Settings.default` wholesale —
        // `onboardingComplete: true` only survives via a real decode, since
        // `Settings.default.onboardingComplete` is `false`.
        #expect(current.onboardingComplete == true)
        #expect(current.scanHuggingFaceCache == false)
        #expect(current.huggingFaceCacheDirectories == [Settings.defaultHuggingFaceCacheDirectory])
    }
}
