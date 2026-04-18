// ModelLibraryViewModel.swift
// macMLX
//
// @Observable view model for ModelLibraryView. Handles local library scanning
// and Hugging Face search, driving the coordinator for load/unload.

import Foundation
import MacMLXCore

@Observable
@MainActor
final class ModelLibraryViewModel {

    // MARK: - Tab

    enum Tab: String, CaseIterable, Identifiable {
        case local = "Local"
        case huggingFace = "Hugging Face"
        var id: String { rawValue }
    }

    // MARK: - State

    var selectedTab: Tab = .local
    var searchQuery: String = ""

    // Local
    var localModels: [LocalModel] = []
    var isLoadingLocal = false
    var localError: String? = nil

    // HF
    var hfModels: [HFModel] = []
    var isSearchingHF = false
    var hfError: String? = nil

    // Actions in flight
    var loadingModelID: String? = nil
    var downloadingModelIDs: Set<String> = []
    /// Latest download progress per model, keyed by HF modelID. Updated
    /// from the URLSession delegate via a @MainActor hop.
    var downloadProgress: [String: DownloadProgress] = [:]
    /// Outer Swift Tasks for in-flight downloads, keyed by HF modelID.
    /// Cancelling the Task cancels the underlying URLSession download
    /// (URLError.cancelled is thrown), wiring up issue #5's Cancel button.
    private var downloadTasks: [String: Task<Void, Never>] = [:]

    // MARK: - Private

    // Explicit dependencies rather than a back-reference to AppState —
    // AppState owns this VM, so holding AppState here would create a
    // retain cycle. Mirrors ChatViewModel's wiring.
    private let library: ModelLibraryManager
    private let coordinator: EngineCoordinator
    private let downloader: HFDownloader
    private let sizeCache: HFSizeCache
    /// Read the current model directory on demand. A closure (not a
    /// stored URL) so settings changes are observed live without this
    /// VM having to subscribe to SettingsManager.
    private let modelDirectoryProvider: @MainActor () -> URL
    private var searchTask: Task<Void, Never>? = nil
    /// Separately tracked so a follow-up `searchHF()` can cancel a
    /// still-running size enrichment before it races the new results.
    /// Without this, a stale enrichment pass could write a size into a
    /// row that happens to share an `id` with the superseded result set.
    private var enrichTask: Task<Void, Never>? = nil

    // MARK: - Init

    init(
        library: ModelLibraryManager,
        coordinator: EngineCoordinator,
        downloader: HFDownloader,
        sizeCache: HFSizeCache,
        modelDirectoryProvider: @escaping @MainActor () -> URL
    ) {
        self.library = library
        self.coordinator = coordinator
        self.downloader = downloader
        self.sizeCache = sizeCache
        self.modelDirectoryProvider = modelDirectoryProvider
    }

    // MARK: - Local

    /// Directory that `loadLocalModels()` scanned last — or the currently
    /// configured one if no scan has happened yet. Surfaced in the
    /// "No Local Models" empty-state so users can tell at a glance
    /// whether the app is looking at the directory they expect.
    var scanDirectory: URL {
        modelDirectoryProvider()
    }

    func loadLocalModels() async {
        isLoadingLocal = true
        localError = nil
        let dir = modelDirectoryProvider()
        await LogManager.shared.info(
            "Scanning local models at: \(dir.path(percentEncoded: false))",
            category: .system
        )
        do {
            let found = try await library.scan(dir)
            localModels = found
            await LogManager.shared.info(
                "Scan complete: \(found.count) local model(s) at \(dir.path(percentEncoded: false))",
                category: .system
            )
            if found.isEmpty {
                // List the raw subdirs so we can see whether scan is
                // looking at the right path but finding the wrong kind
                // of content (wrong format, empty dir, etc.)
                let raw = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
                await LogManager.shared.warning(
                    "Zero models found. Subdirs present: \(raw.prefix(20).joined(separator: ", "))",
                    category: .system
                )
            }
        } catch {
            await LogManager.shared.error(
                "Scan failed at \(dir.path(percentEncoded: false)): \(error.localizedDescription)",
                category: .system
            )
            localError = error.localizedDescription
        }
        isLoadingLocal = false
    }

    func loadModel(_ model: LocalModel) async {
        loadingModelID = model.id
        _ = await coordinator.load(model)
        loadingModelID = nil
    }

    func unloadModel() async {
        await coordinator.unload()
    }

    func deleteModel(_ model: LocalModel) {
        do {
            try FileManager.default.removeItem(at: model.directory)
            localModels.removeAll { $0.id == model.id }
        } catch {
            localError = "Delete failed: \(error.localizedDescription)"
        }
    }

    var loadedModelID: String? {
        coordinator.currentModel?.id
    }

    // MARK: - Search matching

    /// HF's `?search=` endpoint does fuzzy prefix matching — "gemma-4"
    /// returns gemma-3 and gemma-2. We post-filter so every token the
    /// user typed must appear in the repo name (not the org prefix).
    /// Tokens split on non-alphanumerics, so `gemma-4` → ["gemma", "4"],
    /// `qwen3 8b` → ["qwen3", "8b"].
    private static func tokenize(_ query: String) -> [String] {
        query
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map { $0.lowercased() }
            .filter { !$0.isEmpty }
    }

    private static func matches(_ model: HFModel, tokens: [String]) -> Bool {
        guard !tokens.isEmpty else { return true }
        let name = (model.id.split(separator: "/").last.map(String.init) ?? model.id).lowercased()
        return tokens.allSatisfy { name.contains($0) }
    }

    // MARK: - HF Search

    func searchHF() {
        searchTask?.cancel()
        // Cancel any in-flight enrichment BEFORE a new search assigns
        // `hfModels`, so a stale fetch can't write a size into a row
        // belonging to the superseded result set.
        enrichTask?.cancel()
        guard !searchQuery.trimmingCharacters(in: .whitespaces).isEmpty else {
            hfModels = []
            return
        }
        let query = searchQuery
        searchTask = Task {
            isSearchingHF = true
            hfError = nil
            do {
                let results = try await downloader.search(query: query, limit: 20)
                guard !Task.isCancelled else { return }
                let tokens = Self.tokenize(query)
                hfModels = results.filter { Self.matches($0, tokens: tokens) }
                // Kick off enrichment as its own tracked task so a
                // subsequent searchHF() can cancel it cleanly instead of
                // awaiting inline and blocking the search state machine.
                enrichTask = Task { [weak self] in
                    await self?.enrichSizes()
                }
            } catch {
                guard !Task.isCancelled else { return }
                hfError = error.localizedDescription
            }
            isSearchingHF = false
        }
    }

    /// Parallel size-fetch for the currently-listed HF results. Cap the
    /// concurrency so we don't hammer the Hub with a burst of 20
    /// simultaneous requests when the user types rapidly. The VM is
    /// @MainActor, so mutations into `hfModels[idx].sizeBytes` are race-free.
    private func enrichSizes() async {
        let ids = hfModels.map(\.id)
        let downloader = self.downloader
        let cache = self.sizeCache

        // Fast pass: apply any already-cached sizes synchronously before
        // we kick off network fetches. This makes re-searches feel instant.
        for id in ids {
            if let size = await cache.get(id),
               let idx = hfModels.firstIndex(where: { $0.id == id }) {
                hfModels[idx].sizeBytes = size
            }
        }

        // Only fetch sizes for models that are still unset.
        let missingIDs = hfModels.filter { $0.sizeBytes == nil }.map(\.id)
        guard !missingIDs.isEmpty else { return }

        await withTaskGroup(of: (String, Int64?).self) { group in
            var inflight = 0
            let maxInflight = 4
            var iterator = missingIDs.makeIterator()

            func enqueue() {
                guard let next = iterator.next() else { return }
                inflight += 1
                group.addTask {
                    let size = try? await downloader.sizeBytes(for: next)
                    return (next, size)
                }
            }
            while inflight < maxInflight { enqueue() }

            while let (id, size) = await group.next() {
                inflight -= 1
                if Task.isCancelled {
                    group.cancelAll()
                    break
                }
                if let size, size > 0 {
                    if let idx = hfModels.firstIndex(where: { $0.id == id }) {
                        hfModels[idx].sizeBytes = size
                    }
                    await cache.put(id, size: size)
                }
                if !Task.isCancelled { enqueue() }
            }
        }
    }

    func downloadModel(_ model: HFModel) {
        // Already downloading? Cancel-then-start would be confusing; no-op.
        guard downloadTasks[model.id] == nil else { return }

        downloadingModelIDs.insert(model.id)
        let dir = modelDirectoryProvider()

        // Bridge the URLSession delegate's @Sendable callback (background
        // queue) onto MainActor so SwiftUI observes the dictionary update.
        let modelID = model.id
        let handler: HFDownloader.ProgressHandler = { [weak self] progress in
            guard let self else { return }
            Task { @MainActor in
                self.downloadProgress[modelID] = progress
            }
        }

        let task = Task { @MainActor in
            defer {
                downloadingModelIDs.remove(modelID)
                downloadProgress.removeValue(forKey: modelID)
                downloadTasks.removeValue(forKey: modelID)
            }
            do {
                _ = try await downloader.download(
                    modelID: modelID,
                    to: dir,
                    progress: handler
                )
                await loadLocalModels()
            } catch is CancellationError {
                // User hit Cancel — silent, no error banner.
            } catch let err as URLError where err.code == .cancelled {
                // URLSession reports Task-level cancel as URLError.cancelled.
                // Treat it as a silent user cancel too.
                // Best-effort cleanup of partial files:
                cleanupPartialDirectory(for: modelID, under: dir)
            } catch {
                hfError = "Download failed: \(error.localizedDescription)"
            }
        }
        downloadTasks[modelID] = task
    }

    /// Cancel an in-flight download. No-op if the model isn't currently
    /// downloading. URLSession throws `URLError.cancelled` in response,
    /// which `downloadModel` catches and suppresses.
    func cancelDownload(_ model: HFModel) {
        downloadTasks[model.id]?.cancel()
    }

    /// Remove the half-downloaded model directory so a later retry starts
    /// fresh. Best-effort; ignores errors (the dir may not exist).
    private func cleanupPartialDirectory(for modelID: String, under parent: URL) {
        let modelName = modelID.split(separator: "/").last.map(String.init) ?? modelID
        let dir = parent.appending(path: modelName, directoryHint: .isDirectory)
        try? FileManager.default.removeItem(at: dir)
    }

    func isDownloaded(_ model: HFModel) -> Bool {
        let modelName = model.id.split(separator: "/").last.map(String.init) ?? model.id
        return localModels.contains { $0.id == modelName }
    }
}
