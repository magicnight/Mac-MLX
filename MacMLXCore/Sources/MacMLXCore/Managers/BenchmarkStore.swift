// BenchmarkStore.swift
// MacMLXCore
//
// Filesystem-backed benchmark history store (issue #22).
//
// Design mirrors ConversationStore: one JSON file per result at
// `<directory>/{uuid}.json`, atomic writes, corrupt files don't block
// other loads. History is sorted newest-first.

import Foundation

/// Benchmark history store. Every method is `async` because all I/O is
/// serialised on the actor.
public actor BenchmarkStore {

    private let directory: URL
    private let fileManager: FileManager

    /// Create a store backed by `<directory>/{uuid}.json`.
    ///
    /// Default directory is `~/.mac-mlx/benchmarks/` (real home — takes
    /// advantage of macOS App Sandbox's dotfile exemption, same as the
    /// rest of the `.mac-mlx` data root).
    public init(directory: URL? = nil, fileManager: FileManager = .default) {
        if let directory {
            self.directory = directory
        } else {
            // Use NSHomeDirectoryForUser to bypass the sandbox container
            // redirect (matches Settings.default.modelDirectory logic).
            let home: URL = {
                if let path = NSHomeDirectoryForUser(NSUserName()) {
                    return URL(filePath: path, directoryHint: .isDirectory)
                }
                return URL(filePath: NSHomeDirectory(), directoryHint: .isDirectory)
            }()
            self.directory = home.appending(
                path: ".mac-mlx/benchmarks",
                directoryHint: .isDirectory
            )
        }
        self.fileManager = fileManager
    }

    // MARK: - Public API

    /// Persist `result` to disk atomically. Creates the directory if missing.
    public func save(_ result: BenchmarkResult) async throws {
        try ensureDirectory()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(result)
        let url = fileURL(for: result.id)
        try data.write(to: url, options: .atomic)
    }

    /// Return all stored results sorted newest-first. Corrupt files are
    /// skipped silently — they don't block other loads.
    public func list() async throws -> [BenchmarkResult] {
        try ensureDirectory()
        let urls = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "json" }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var loaded: [BenchmarkResult] = []
        for url in urls {
            if let data = try? Data(contentsOf: url),
               let result = try? decoder.decode(BenchmarkResult.self, from: data) {
                loaded.append(result)
            }
        }
        return loaded.sorted { $0.timestamp > $1.timestamp }
    }

    /// Return the most-recently-run result, or nil if the store is empty.
    public func loadLatest() async throws -> BenchmarkResult? {
        try await list().first
    }

    /// Remove a result from disk. Idempotent — no error if missing.
    public func delete(id: UUID) async throws {
        let url = fileURL(for: id)
        try? fileManager.removeItem(at: url)
    }

    /// Wipe all results. Useful for a "Clear History" button.
    public func deleteAll() async throws {
        try ensureDirectory()
        let urls = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
        for url in urls {
            try? fileManager.removeItem(at: url)
        }
    }

    // MARK: - Private

    private func fileURL(for id: UUID) -> URL {
        directory.appending(path: "\(id.uuidString).json", directoryHint: .notDirectory)
    }

    private func ensureDirectory() throws {
        if !fileManager.fileExists(atPath: directory.path) {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }
}
