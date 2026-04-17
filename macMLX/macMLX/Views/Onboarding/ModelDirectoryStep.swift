// ModelDirectoryStep.swift
// macMLX — Onboarding Step 2

import SwiftUI
import MacMLXCore

struct ModelDirectoryStep: View {

    @Bindable var state: OnboardingState

    private struct ScannedLocation: Identifiable {
        let id = UUID()
        let label: String
        let url: URL
        var modelCount: Int = 0
        var hasGGUF: Bool = false
    }

    @State private var scannedLocations: [ScannedLocation] = []
    @State private var isScanning = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            stepHeader(index: 1, title: "Where are your models?")

            Text("We found these locations on your Mac:")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            scannedList

            Divider()

            selectedDirectoryPicker

            Divider()

            HStack {
                Button("Create default: ~/.mac-mlx/models") {
                    createDefaultDirectory()
                }
                .buttonStyle(.bordered)

                Spacer()

                Button("Back") { state.goBack() }
                    .buttonStyle(.bordered)

                Button("Continue") {
                    Task { await advanceAfterSaving() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(state.selectedModelDirectory == nil)
            }
        }
        .padding(40)
        .task { await scanLocations() }
    }

    // MARK: - Scanned list

    @ViewBuilder
    private var scannedList: some View {
        if isScanning {
            ProgressView("Scanning…")
        } else if scannedLocations.isEmpty {
            Text("No model directories found on this Mac.")
                .foregroundStyle(.secondary)
                .font(.caption)
        } else {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(scannedLocations) { loc in
                    scannedRow(loc)
                }
            }
        }
    }

    private func scannedRow(_ loc: ScannedLocation) -> some View {
        HStack(alignment: .top, spacing: 8) {
            if loc.hasGGUF {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            } else if loc.modelCount > 0 {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else {
                Image(systemName: "circle")
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(loc.url.path(percentEncoded: false)
                    .replacingOccurrences(
                        of: FileManager.default.homeDirectoryForCurrentUser.path,
                        with: "~"
                    ))
                .font(.system(.caption, design: .monospaced))

                if loc.hasGGUF {
                    Text("GGUF format — not compatible with macMLX.")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                } else if loc.modelCount > 0 {
                    Text("\(loc.modelCount) MLX model\(loc.modelCount == 1 ? "" : "s") found")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            Button("Use") {
                state.selectedModelDirectory = loc.url
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(loc.hasGGUF)
        }
    }

    // MARK: - Directory picker

    private var selectedDirectoryPicker: some View {
        HStack {
            Text("Use:")
                .font(.subheadline)

            if let dir = state.selectedModelDirectory {
                Text(dir.path(percentEncoded: false)
                    .replacingOccurrences(
                        of: FileManager.default.homeDirectoryForCurrentUser.path,
                        with: "~"
                    ))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            } else {
                Text("None selected")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            }

            Spacer()

            Button("Browse…") { showOpenPanel() }
                .buttonStyle(.bordered)
        }
    }

    // MARK: - Actions

    private func scanLocations() async {
        isScanning = true
        let home = FileManager.default.homeDirectoryForCurrentUser
        // Fix #3: scan only macMLX's own canonical locations.
        // LM Studio (GGUF) and Ollama (packed GGUF) are format-incompatible
        // with mlx-swift-lm — reading their dirs would only produce a warning
        // we'd have to show the user. Focus on the Apple ecosystem path.
        let candidates: [(String, URL)] = [
            ("~/.mac-mlx/models", home.appending(path: ".mac-mlx/models")),
            ("~/models",          home.appending(path: "models")),
        ]

        var found: [ScannedLocation] = []
        let fm = FileManager.default

        for (label, url) in candidates {
            guard fm.fileExists(atPath: url.path) else { continue }
            var loc = ScannedLocation(label: label, url: url)

            if let contents = try? fm.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            ) {
                for item in contents {
                    let itemContents = (try? fm.contentsOfDirectory(
                        at: item,
                        includingPropertiesForKeys: nil,
                        options: .skipsHiddenFiles
                    ).map { $0.lastPathComponent }) ?? []
                    let fmt = ModelFormat.detect(in: itemContents)
                    if fmt == .mlx { loc.modelCount += 1 }
                    if fmt == .gguf { loc.hasGGUF = true }
                }
            }
            found.append(loc)
        }

        scannedLocations = found

        // Auto-select the best directory (most MLX models, no GGUF only)
        if state.selectedModelDirectory == nil {
            let best = found
                .filter { !$0.hasGGUF || $0.modelCount > 0 }
                .max(by: { $0.modelCount < $1.modelCount })
            state.selectedModelDirectory = best?.url
        }

        isScanning = false
    }

    private func createDefaultDirectory() {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appending(path: ".mac-mlx/models")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        state.selectedModelDirectory = dir
    }

    private func showOpenPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Choose Model Directory"
        if panel.runModal() == .OK, let url = panel.url {
            state.selectedModelDirectory = url
        }
    }

    private func advanceAfterSaving() async {
        if let dir = state.selectedModelDirectory {
            await state.confirmDirectory(dir)
        }
        state.advance()
    }
}

// MARK: - Shared helper

func stepHeader(index: Int, title: String) -> some View {
    VStack(alignment: .leading, spacing: 4) {
        Text("Step \(index) of 4")
            .font(.caption)
            .foregroundStyle(.secondary)
        Text(title)
            .font(.title2.bold())
    }
}

#Preview {
    ModelDirectoryStep(state: OnboardingState(appState: AppState()))
        .frame(width: 540, height: 500)
}
