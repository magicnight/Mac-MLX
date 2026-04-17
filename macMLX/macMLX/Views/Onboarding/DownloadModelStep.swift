// DownloadModelStep.swift
// macMLX — Onboarding Step 4
//
// Memory-aware model recommendations. Skippable.
// If user already has MLX models in the selected directory, this step
// is skipped by OnboardingWindow before presenting.

import SwiftUI
import MacMLXCore

struct DownloadModelStep: View {

    @Bindable var state: OnboardingState
    @Environment(AppState.self) private var appState

    // MARK: - Model recommendations (from onboarding.md spec)

    private struct RecommendedModel: Identifiable {
        let id: String
        let name: String
        let sizeGB: Double
        let params: String
        let minMemoryGB: Double
        let description: String
    }

    private let allModels: [RecommendedModel] = [
        RecommendedModel(
            id: "mlx-community/Qwen3-1.7B-4bit",
            name: "Qwen3 1.7B",
            sizeGB: 1.1,
            params: "1.7B",
            minMemoryGB: 8,
            description: "Fastest · Great for coding tasks"
        ),
        RecommendedModel(
            id: "mlx-community/Qwen3-8B-4bit",
            name: "Qwen3 8B",
            sizeGB: 4.5,
            params: "8B",
            minMemoryGB: 8,
            description: "Best balance of speed and quality"
        ),
        RecommendedModel(
            id: "mlx-community/Qwen3-14B-4bit",
            name: "Qwen3 14B",
            sizeGB: 8.2,
            params: "14B",
            minMemoryGB: 16,
            description: "Higher quality · Needs 16GB+"
        ),
        RecommendedModel(
            id: "mlx-community/Qwen3-32B-4bit",
            name: "Qwen3 32B",
            sizeGB: 19.0,
            params: "32B",
            minMemoryGB: 32,
            description: "Near-frontier quality · Needs 32GB+"
        ),
    ]

    @State private var selectedModelID: String = ""
    @State private var downloadProgress: Double? = nil
    @State private var downloadStats: DownloadProgress? = nil
    @State private var isDownloading = false
    @State private var downloadError: String? = nil

    private var totalMemoryGB: Double { MemoryProbe.totalMemoryGB() }

    private var eligibleModels: [RecommendedModel] {
        allModels.filter { $0.minMemoryGB <= totalMemoryGB }
    }

    private var defaultSelection: RecommendedModel? {
        // Pick the largest model that fits
        eligibleModels.last
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            stepHeader(index: 3, title: "Download your first model")

            Text("Recommended for your Mac (\(String(format: "%.0f", totalMemoryGB)) GB):")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            modelList

            if isDownloading {
                if let stats = downloadStats {
                    // Current-file bar — honest about per-file progress.
                    if stats.currentFileTotalBytes > 0 {
                        ProgressView(value: stats.currentFileFraction)
                            .progressViewStyle(.linear)
                    } else {
                        ProgressView()
                            .progressViewStyle(.linear)
                    }
                    HStack(spacing: 6) {
                        if stats.currentFileTotalBytes > 0 {
                            Text(stats.currentFileHuman)
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                            Text("·").foregroundStyle(.tertiary)
                            Text(stats.currentFilePercent)
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                            Text("·").foregroundStyle(.tertiary)
                        } else {
                            Text("Starting…")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                            Text("·").foregroundStyle(.tertiary)
                        }
                        Text(stats.filesHuman)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    if let currentFile = stats.currentFileName {
                        Text(currentFile)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                } else {
                    ProgressView("Starting download…")
                        .progressViewStyle(.linear)
                }
            }

            if let err = downloadError {
                Text("Download failed: \(err)")
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            Text("Models exceeding your RAM are hidden.")
                .font(.caption2)
                .foregroundStyle(.secondary)

            Spacer()

            HStack {
                Button("Back") { state.goBack() }
                    .buttonStyle(.bordered)
                    .disabled(isDownloading)

                Spacer()

                Button("Skip") {
                    state.skipDownload = true
                    state.advance()
                }
                .buttonStyle(.bordered)
                .disabled(isDownloading)

                Button(isDownloading ? "Downloading…" : "Download \(selectedModelName)") {
                    Task { await startDownload() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(selectedModelID.isEmpty || isDownloading)
            }
        }
        .padding(40)
        .onAppear {
            selectedModelID = defaultSelection?.id ?? ""
        }
    }

    // MARK: - Model list

    private var modelList: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(eligibleModels) { model in
                HStack {
                    Image(systemName: selectedModelID == model.id
                          ? "largecircle.fill.circle"
                          : "circle")
                        .foregroundStyle(.tint)
                        .onTapGesture { selectedModelID = model.id }

                    VStack(alignment: .leading, spacing: 1) {
                        Text(model.name)
                            .font(.subheadline.bold())
                        Text(model.description)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    Text(String(format: "%.1f GB", model.sizeGB))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)

                    if model.id == defaultSelection?.id {
                        Text("★")
                            .foregroundStyle(.orange)
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture { selectedModelID = model.id }
            }
        }
        .padding(12)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
    }

    private var selectedModelName: String {
        eligibleModels.first { $0.id == selectedModelID }?.name ?? ""
    }

    // MARK: - Download

    private func startDownload() async {
        guard let dir = state.selectedModelDirectory,
              !selectedModelID.isEmpty else { return }
        isDownloading = true
        downloadError = nil
        downloadProgress = 0
        downloadStats = nil
        let modelID = selectedModelID

        // Bridge URLSession's background-queue callback to MainActor so
        // SwiftUI sees the @State updates.
        let handler: HFDownloader.ProgressHandler = { progress in
            Task { @MainActor in
                downloadStats = progress
                downloadProgress = progress.currentFileFraction
            }
        }

        do {
            _ = try await appState.downloader.download(
                modelID: modelID,
                to: dir,
                progress: handler
            )
            downloadProgress = 1.0
            // Give user a moment to see 100%
            try? await Task.sleep(for: .milliseconds(500))
            state.advance()
        } catch {
            downloadError = error.localizedDescription
        }
        isDownloading = false
    }
}

#Preview {
    DownloadModelStep(state: OnboardingState(appState: AppState()))
        .environment(AppState())
        .frame(width: 480, height: 500)
}
