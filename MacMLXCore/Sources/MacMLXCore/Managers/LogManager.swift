// LogManager.swift
// MacMLXCore
//
// Thin actor shim over Pulse's LoggerStore.
// The GUI Logs tab (Stage 4) will layer PulseUI.ConsoleView on top of the
// same backing store via LoggerStore.shared.

import Foundation
import Pulse

// MARK: - LogLevel

/// Severity levels exposed to macMLX callers, mapped to Pulse's Level enum.
public enum LogLevel: String, Sendable, CaseIterable {
    case debug
    case info
    case warning
    case error
    case critical

    /// Maps to the corresponding `LoggerStore.Level` value.
    var pulse: LoggerStore.Level {
        switch self {
        case .debug:    return .debug
        case .info:     return .info
        case .warning:  return .warning
        case .error:    return .error
        case .critical: return .critical
        }
    }
}

// MARK: - LogCategory

/// Categorical labels for filtering in the Pulse console.
public enum LogCategory: String, Sendable, CaseIterable {
    case engine
    case inference
    case download
    case http
    case system
    case error
}

// MARK: - LogManager

/// App-wide logging gateway backed by Pulse's `LoggerStore`.
///
/// Use `LogManager.shared` from anywhere in the app. Tests supply their own
/// `LoggerStore` to keep log data isolated from the production store.
public actor LogManager {

    // MARK: Shared instance

    /// Production singleton — backed by `LoggerStore.shared`.
    public static let shared = LogManager()

    // MARK: State

    /// Underlying Pulse store. `nonisolated` so UI surfaces
    /// (PulseUI.ConsoleView in the Logs tab, #16) can grab it
    /// synchronously from @MainActor without hopping into the actor
    /// — `LoggerStore` is a Sendable reference type backed by its own
    /// internal concurrency, safe to hand out.
    public nonisolated let store: LoggerStore

    // MARK: Init

    /// Production initialiser — uses `LoggerStore.shared`.
    public init() {
        self.store = LoggerStore.shared
    }

    /// Test / preview initialiser — caller provides a custom store.
    public init(store: LoggerStore) {
        self.store = store
    }

    // MARK: Logging

    /// Record a message at the given level and category.
    public func log(
        _ message: String,
        level: LogLevel = .info,
        category: LogCategory = .system
    ) {
        store.storeMessage(
            label: category.rawValue,
            level: level.pulse,
            message: message,
            metadata: nil
        )
    }

    // MARK: Convenience shortcuts

    /// Log a debug-level message.
    public func debug(_ message: String, category: LogCategory = .system) {
        log(message, level: .debug, category: category)
    }

    /// Log an info-level message.
    public func info(_ message: String, category: LogCategory = .system) {
        log(message, level: .info, category: category)
    }

    /// Log a warning-level message.
    public func warning(_ message: String, category: LogCategory = .system) {
        log(message, level: .warning, category: category)
    }

    /// Log an error-level message.
    public func error(_ message: String, category: LogCategory = .error) {
        log(message, level: .error, category: category)
    }

    /// Log a critical-level message.
    public func critical(_ message: String, category: LogCategory = .error) {
        log(message, level: .critical, category: category)
    }

    // MARK: Flushing

    /// Synchronously flush any pending writes to disk.
    ///
    /// `storeMessage(...)` queues writes on Pulse's background context and
    /// returns immediately. Callers that need to read back logs (tests, app
    /// shutdown) must invoke `flush()` first to guarantee the writes have
    /// landed in the view context.
    public func flush() {
        store.backgroundContext.performAndWait {
            try? store.backgroundContext.save()
        }
        store.viewContext.performAndWait {
            store.viewContext.refreshAllObjects()
        }
    }
}
