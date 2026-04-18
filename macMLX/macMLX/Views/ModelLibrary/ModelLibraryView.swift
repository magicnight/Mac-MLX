// ModelLibraryView.swift
// macMLX
//
// Local + Hugging Face model browser. Replaces the Stage 4 Task 5 stub.

import SwiftUI
import MacMLXCore

struct ModelLibraryView: View {

    @Environment(AppState.self) private var appState

    var body: some View {
        // VM now lives on AppState so tab switches no longer tear down
        // the downloadProgress dictionary mid-download (issue #1
        // follow-up in v0.3.6).
        ModelLibraryContent(viewModel: appState.modelLibrary)
            // Force a view-identity change when scan results arrive so
            // SwiftUI re-renders even if the @Observable registrar
            // misses the async mutation on the hoisted VM.
            .id(appState.modelLibrary.localModels.count)
            .task {
                await appState.modelLibrary.loadLocalModels()
            }
            // Auto-rescan when the user changes the model directory in
            // Settings — pre-v0.3.1 user had to press Refresh manually
            // after Settings changes, which made it feel like Refresh
            // was broken.
            .onChange(of: appState.currentSettings.modelDirectory) { _, _ in
                Task { await appState.modelLibrary.loadLocalModels() }
            }
    }
}

// MARK: - ModelLibraryContent

private struct ModelLibraryContent: View {

    @Bindable var viewModel: ModelLibraryViewModel

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            tabContent
        }
        .navigationTitle("Model Library")
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 12) {
            // Search bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField(
                    viewModel.selectedTab == .local
                        ? "Filter local models…"
                        : "Search mlx-community…",
                    text: $viewModel.searchQuery
                )
                .textFieldStyle(.plain)
                .onSubmit {
                    if viewModel.selectedTab == .huggingFace {
                        viewModel.searchHF()
                    }
                }
                .onChange(of: viewModel.searchQuery) { _, _ in
                    if viewModel.selectedTab == .huggingFace {
                        viewModel.searchHF()
                    }
                }

                if !viewModel.searchQuery.isEmpty {
                    Button {
                        viewModel.searchQuery = ""
                        viewModel.hfModels = []
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))

            // Tab picker
            Picker("", selection: $viewModel.selectedTab) {
                ForEach(ModelLibraryViewModel.Tab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .fixedSize()

            // Refresh for local tab
            if viewModel.selectedTab == .local {
                Button {
                    Task { await viewModel.loadLocalModels() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .disabled(viewModel.isLoadingLocal)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: - Tab content

    @ViewBuilder
    private var tabContent: some View {
        switch viewModel.selectedTab {
        case .local:
            localTab
        case .huggingFace:
            hfTab
        }
    }

    // MARK: - Local tab

    @ViewBuilder
    private var localTab: some View {
        Group {
            if viewModel.isLoadingLocal {
                ProgressView("Scanning model directory…")
            } else if let err = viewModel.localError {
                errorView(message: err)
            } else {
                let filtered = viewModel.localModels.filter {
                    viewModel.searchQuery.isEmpty
                    || $0.displayName.localizedCaseInsensitiveContains(viewModel.searchQuery)
                }

                if filtered.isEmpty {
                    ContentUnavailableView(
                        "No Local Models",
                        systemImage: "tray",
                        description: Text(
                            viewModel.searchQuery.isEmpty
                                // Spell out the actual scanned path. Fixes the
                                // "I copied models but they don't show up"
                                // confusion when Settings points at a stale
                                // directory (e.g. leftover v0.1 `~/models`).
                                ? "No models found in \(Self.displayPath(viewModel.scanDirectory)).\nDownload from the Hugging Face tab, or set the directory in Settings."
                                : "No models match \"\(viewModel.searchQuery)\""
                        )
                    )
                } else {
                    List(filtered) { model in
                        LocalModelRow(
                            model: model,
                            isLoaded: viewModel.loadedModelID == model.id,
                            isLoading: viewModel.loadingModelID == model.id,
                            isPinned: viewModel.pinnedModelIDs.contains(model.id),
                            hasUpdateAvailable: viewModel.modelsWithUpdate.contains(model.id),
                            onLoad: {
                                Task { await viewModel.loadModel(model) }
                            },
                            onUnload: {
                                Task { await viewModel.unloadModel() }
                            },
                            onTogglePin: {
                                Task { await viewModel.togglePin(model) }
                            },
                            onDelete: {
                                viewModel.deleteModel(model)
                            }
                        )
                    }
                    .listStyle(.inset)
                    .scrollContentBackground(.hidden)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - HF tab

    @ViewBuilder
    private var hfTab: some View {
        Group {
            if viewModel.isSearchingHF {
                ProgressView("Searching Hugging Face…")
            } else if let err = viewModel.hfError {
                errorView(message: err)
            } else if viewModel.hfModels.isEmpty {
                ContentUnavailableView(
                    "Search for Models",
                    systemImage: "magnifyingglass",
                    description: Text("Type a model name to search mlx-community on Hugging Face.")
                )
            } else {
                List(viewModel.hfModels) { model in
                    HFModelRow(
                        model: model,
                        isDownloaded: viewModel.isDownloaded(model),
                        isDownloading: viewModel.downloadingModelIDs.contains(model.id),
                        progress: viewModel.downloadProgress[model.id],
                        onDownload: { viewModel.downloadModel(model) },
                        onCancel: { viewModel.cancelDownload(model) }
                    )
                }
                .listStyle(.inset)
                .scrollContentBackground(.hidden)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Path display

    /// Collapse the user's real home to `~` for terser rendering in the
    /// empty-state message.
    private static func displayPath(_ url: URL) -> String {
        let raw = url.path(percentEncoded: false)
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return raw.hasPrefix(home) ? "~" + raw.dropFirst(home.count) : raw
    }

    // MARK: - Error view

    private func errorView(message: String) -> some View {
        ContentUnavailableView(
            "Error",
            systemImage: "exclamationmark.triangle",
            description: Text(message)
        )
    }
}

#Preview {
    ModelLibraryView()
        .environment(AppState())
        .frame(width: 600, height: 500)
}
