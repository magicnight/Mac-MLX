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

    /// Production singleton — backed by a size-capped LoggerStore.
    public static let shared = LogManager()

    // MARK: State

    /// Underlying Pulse store. `nonisolated` so UI surfaces
    /// (PulseUI.ConsoleView in the Logs tab, #16) can grab it
    /// synchronously from @MainActor without hopping into the actor
    /// — `LoggerStore` is a Sendable reference type backed by its own
    /// internal concurrency, safe to hand out.
    public nonisolated let store: LoggerStore

    // MARK: Init

    /// Production initialiser — creates (or reopens) a size-capped
    /// LoggerStore at the standard macMLX data directory.
    ///
    /// Why not `LoggerStore.shared`: the default shared store has no
    /// user-tunable size limit, so months of chat-engine token traces
    /// can fill many GB. We keep a 100 MB FIFO cap (Pulse evicts the
    /// oldest entries once the cap is reached) — plenty for a week's
    /// worth of real-world events without unbounded disk growth.
    public init() {
        self.store = Self.makeCappedStore()
    }

    /// Test / preview initialiser — caller provides a custom store.
    public init(store: LoggerStore) {
        self.store = store
    }

    /// Build the production LoggerStore with a 100 MB size cap.
    /// Falls back to `LoggerStore.shared` if our explicit store fails
    /// to open (e.g. disk full, sandbox denied the directory) — better
    /// to log to the default than crash the app.
    private static func makeCappedStore() -> LoggerStore {
        let storeURL = DataRoot.macMLX("logs/macmlx.pulse")
        // Parent directory must exist before Pulse opens the store.
        try? FileManager.default.createDirectory(
            at: storeURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var configuration = LoggerStore.Configuration()
        // 100 MB FIFO. Pulse's SQLite-backed store auto-evicts oldest
        // entries once the file size approaches this cap, so logs
        // never grow unbounded even under heavy token-level tracing.
        configuration.sizeLimit = 100 * 1024 * 1024
        do {
            return try LoggerStore(
                storeURL: storeURL,
                options: [.create],
                configuration: configuration
            )
        } catch {
            return LoggerStore.shared
        }
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
