// ModelLibraryView.swift
// macMLX
//
// Local + Hugging Face model browser. Replaces the Stage 4 Task 5 stub.

import SwiftUI
import MacMLXCore

struct ModelLibraryView: View {

    @Environment(AppState.self) private var appState
    @State private var viewModel: ModelLibraryViewModel?

    var body: some View {
        Group {
            if let vm = viewModel {
                ModelLibraryContent(viewModel: vm)
            } else {
                ProgressView("Loading…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task {
            let vm = ModelLibraryViewModel(appState: appState)
            viewModel = vm
            await vm.loadLocalModels()
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
        if viewModel.isLoadingLocal {
            ProgressView("Scanning model directory…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
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
                            ? "Download models from the Hugging Face tab, or add them to your model directory."
                            : "No models match \"\(viewModel.searchQuery)\""
                    )
                )
            } else {
                List(filtered) { model in
                    LocalModelRow(
                        model: model,
                        isLoaded: viewModel.loadedModelID == model.id,
                        isLoading: viewModel.loadingModelID == model.id,
                        onLoad: {
                            Task { await viewModel.loadModel(model) }
                        },
                        onUnload: {
                            Task { await viewModel.unloadModel() }
                        },
                        onDelete: {
                            viewModel.deleteModel(model)
                        }
                    )
                }
                .listStyle(.inset)
            }
        }
    }

    // MARK: - HF tab

    @ViewBuilder
    private var hfTab: some View {
        if viewModel.isSearchingHF {
            ProgressView("Searching Hugging Face…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
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
                    onDownload: {
                        Task { await viewModel.downloadModel(model) }
                    }
                )
            }
            .listStyle(.inset)
        }
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
