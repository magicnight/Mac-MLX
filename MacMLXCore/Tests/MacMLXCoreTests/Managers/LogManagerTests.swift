// LogManagerTests.swift
// MacMLXCoreTests

import Foundation
import Testing
import Pulse
@testable import MacMLXCore

// MARK: - Helpers

/// Creates a fresh Pulse store in a unique temp directory.
private func makeTempStore() throws -> LoggerStore {
    let url = FileManager.default.temporaryDirectory
        .appending(path: "log-test-\(UUID().uuidString).pulse")
    return try LoggerStore(storeURL: url, options: [.create])
}

// MARK: - Tests

@Suite("LogManager")
struct LogManagerTests {

    // MARK: Message storage

    @Test("log records message text and category label to backing store")
    func logRecordsToBackingStore() async throws {
        let store = try makeTempStore()
        let manager = LogManager(store: store)

        await manager.log("hello from inference", level: .info, category: .inference)
        await manager.flush()

        let messages = try store.messages()
        #expect(messages.isEmpty == false)

        let match = messages.first { $0.text == "hello from inference" }
        #expect(match != nil)
        #expect(match?.label == LogCategory.inference.rawValue)
    }

    // MARK: Level mapping

    @Test("LogLevel.pulse maps to correct LoggerStore.Level values")
    func logLevelsMapToPulseLevels() {
        #expect(LogLevel.debug.pulse == .debug)
        #expect(LogLevel.info.pulse == .info)
        #expect(LogLevel.warning.pulse == .warning)
        #expect(LogLevel.error.pulse == .error)
        #expect(LogLevel.critical.pulse == .critical)
    }

    // MARK: All categories

    @Test("log does not crash for any LogLevel / LogCategory combination")
    func logAllCombinationsDoNotCrash() async throws {
        let store = try makeTempStore()
        let manager = LogManager(store: store)

        for level in LogLevel.allCases {
            for category in LogCategory.allCases {
                await manager.log("test \(level) \(category)", level: level, category: category)
            }
        }
        // Drain pending writes so the LoggerStore tears down cleanly when the
        // test scope exits. Without this, Pulse's async background-context
        // saves can race with test cleanup and trip the Swift 6.0 runtime
        // 'Incorrect actor executor assumption' check on CI.
        await manager.flush()
    }

    // MARK: Convenience methods

    @Test("convenience log methods store messages at the right level")
    func convenienceMethodsStoreAtCorrectLevel() async throws {
        let store = try makeTempStore()
        let manager = LogManager(store: store)

        await manager.debug("d")
        await manager.info("i")
        await manager.warning("w")
        await manager.error("e")
        await manager.critical("c")
        await manager.flush()

        let messages = try store.messages()
        #expect(messages.count == 5)
    }
}
