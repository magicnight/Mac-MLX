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

    private let appState: AppState
    private var searchTask: Task<Void, Never>? = nil

    // MARK: - Init

    init(appState: AppState) {
        self.appState = appState
    }

    // MARK: - Local

    /// Directory that `loadLocalModels()` scanned last — or the currently
    /// configured one if no scan has happened yet. Surfaced in the
    /// "No Local Models" empty-state so users can tell at a glance
    /// whether the app is looking at the directory they expect.
    var scanDirectory: URL {
        appState.currentSettings.modelDirectory
    }

    func loadLocalModels() async {
        isLoadingLocal = true
        localError = nil
        do {
            let dir = appState.currentSettings.modelDirectory
            localModels = try await appState.library.scan(dir)
        } catch {
            localError = error.localizedDescription
        }
        isLoadingLocal = false
    }

    func loadModel(_ model: LocalModel) async {
        loadingModelID = model.id
        _ = await appState.coordinator.load(model)
        loadingModelID = nil
    }

    func unloadModel() async {
        await appState.coordinator.unload()
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
        appState.coordinator.currentModel?.id
    }

    // MARK: - HF Search

    func searchHF() {
        searchTask?.cancel()
        guard !searchQuery.trimmingCharacters(in: .whitespaces).isEmpty else {
            hfModels = []
            return
        }
        let query = searchQuery
        searchTask = Task {
            isSearchingHF = true
            hfError = nil
            do {
                let results = try await appState.downloader.search(query: query, limit: 20)
                guard !Task.isCancelled else { return }
                hfModels = results
            } catch {
                guard !Task.isCancelled else { return }
                hfError = error.localizedDescription
            }
            isSearchingHF = false
        }
    }

    func downloadModel(_ model: HFModel) {
        // Already downloading? Cancel-then-start would be confusing; no-op.
        guard downloadTasks[model.id] == nil else { return }

        downloadingModelIDs.insert(model.id)
        let dir = appState.currentSettings.modelDirectory

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
                _ = try await appState.downloader.download(
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
