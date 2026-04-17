// ParametersViewModel.swift
// macMLX
//
// @Observable facade over ModelParametersStore (MacMLXCore). Holds the
// parameters for the currently-loaded model, auto-reloads when the model
// changes, and debounce-persists slider drags.

import Foundation
import MacMLXCore

@Observable
@MainActor
final class ParametersViewModel {

    // MARK: - State

    /// Parameters for the current model — driven by the Inspector UI.
    /// Default values until `loadForModel(_:)` fires on first model load.
    var parameters: ModelParameters = .default

    /// HF-style ID of the model `parameters` belongs to. Nil if no model
    /// is loaded; writes to `parameters` are not persisted in that case.
    var currentModelID: String?

    // MARK: - Private

    private let store: ModelParametersStore
    private var saveTask: Task<Void, Never>? = nil

    // MARK: - Init

    init(store: ModelParametersStore) {
        self.store = store
    }

    // MARK: - Public API

    /// Load the stored overrides for `modelID`, replacing `parameters`.
    /// No-op if `modelID` is already current (avoids clobbering an
    /// in-flight user edit during rapid model-load cycles).
    func loadForModel(_ modelID: String?) async {
        guard modelID != currentModelID else { return }
        currentModelID = modelID
        if let modelID {
            parameters = await store.load(for: modelID)
        } else {
            parameters = .default
        }
    }

    /// Persist `parameters` for the current model. Debounced by 300 ms
    /// so rapid slider drags collapse into a single disk write.
    func persist() {
        guard let modelID = currentModelID else { return }
        let snapshot = parameters
        saveTask?.cancel()
        saveTask = Task { [store] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            try? await store.save(snapshot, for: modelID)
        }
    }

    /// Reset to factory defaults and persist.
    func resetToDefaults() {
        parameters = .default
        persist()
    }
}
