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
                if let progress = downloadProgress, progress >= 1.0 {
                    ProgressView(value: 1.0)
                        .progressViewStyle(.linear)
                    Text("100%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    // Indeterminate — MacMLXCore v0.1 onProgress is not @Sendable
                    ProgressView("Downloading…")
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
        let modelID = selectedModelID
        do {
            // NOTE: onProgress closure is not @Sendable in MacMLXCore v0.1 API.
            // Pass nil and show indeterminate progress; real progress tracking is v0.2.
            _ = try await appState.downloader.download(
                modelID: modelID,
                to: dir,
                onProgress: nil
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
