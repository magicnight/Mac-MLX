// ModelCardView.swift
// macMLX
//
// Track F ModelCard upgrade: a three-tab (Overview / Parameters / Files)
// detail sheet for a local model. Presented from `ModelLibraryView`'s
// per-row info button. Read-only — editing generation parameters still
// happens in the Chat tab's `ParametersInspector`; this view is about the
// model itself (specs, quantization, files), not sampling knobs.

import SwiftUI
import MacMLXCore

struct ModelCardView: View {

    let model: LocalModel
    /// Every scanned LoRA adapter (`AppState.availableAdapters`) — filtered
    /// down to this model's compatible subset in `compatibleAdapters`.
    let allAdapters: [LocalAdapter]

    @Environment(\.dismiss) private var dismiss

    private enum Tab: String, CaseIterable, Identifiable {
        case overview = "Overview"
        case parameters = "Parameters"
        case files = "Files"
        var id: String { rawValue }
    }

    @State private var selectedTab: Tab = .overview
    @State private var configInfo: ModelConfigInfo?
    @State private var files: [FileEntry] = []

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()

            Picker("", selection: $selectedTab) {
                ForEach(Tab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(12)

            Divider()

            ScrollView {
                Group {
                    switch selectedTab {
                    case .overview: overviewTab
                    case .parameters: parametersTab
                    case .files: filesTab
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(width: 460, height: 440)
        .task {
            configInfo = ModelConfigInfo.read(from: model.directory)
            files = Self.listFiles(in: model.directory)
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(model.id)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            CopyIDButton(id: model.id)
            Button("Done") { dismiss() }
                .buttonStyle(.bordered)
                .keyboardShortcut(.defaultAction)
        }
        .padding(16)
    }

    // MARK: - Overview tab

    private var overviewTab: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 6) {
                badge(formatLabel, color: .green)
                if let quant = model.quantization {
                    badge(quant, color: .secondary)
                }
                if model.isExternalReference {
                    badge("HF Cache", color: .blue)
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                LabeledContent("Size", value: model.humanSize)
                if let params = model.parameterCount {
                    LabeledContent("Parameters", value: params)
                }
                if let architecture = model.architecture {
                    LabeledContent("Architecture", value: architecture)
                }
                LabeledContent("Location") {
                    Text(Self.displayPath(model.directory))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .multilineTextAlignment(.trailing)
                }
            }

            if model.isExternalReference {
                Text("This model is referenced from your Hugging Face cache, not copied into macMLX's model directory. It stays available as long as the cached files remain on disk.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
    }

    // MARK: - Parameters tab
    //
    // Model specs (quantization, context length, architecture) plus
    // compatible-LoRA cross-reference — NOT sampling parameters. Those
    // still live in the Chat tab's `ParametersInspector`; duplicating
    // temperature/topP/etc. here would just be two sources of truth for
    // the same knobs.

    private var parametersTab: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let info = configInfo, info.quantizationBits != nil || info.contextLength != nil || info.modelType != nil {
                VStack(alignment: .leading, spacing: 10) {
                    if let bits = info.quantizationBits {
                        LabeledContent("Quantization") {
                            Text("\(bits)-bit" + (info.quantizationGroupSize.map { ", group size \($0)" } ?? ""))
                        }
                    }
                    if let contextLength = info.contextLength {
                        LabeledContent("Context Length", value: "\(contextLength.formatted()) tokens")
                    }
                    if let modelType = info.modelType {
                        LabeledContent("Model Type", value: modelType)
                    }
                }
            } else {
                Text("No config.json details available for this model.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            if !compatibleAdapters.isEmpty {
                Divider()
                Text("Compatible LoRA Adapters")
                    .font(.headline)
                ForEach(compatibleAdapters) { adapter in
                    HStack(spacing: 6) {
                        Image(systemName: "paperclip")
                            .foregroundStyle(.secondary)
                        Text(adapter.name)
                        Spacer()
                        if let rank = adapter.rank {
                            Text("rank \(rank)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                Text("Pick an adapter for this model from the Chat tab's Parameters Inspector.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(16)
    }

    // MARK: - Files tab

    @ViewBuilder
    private var filesTab: some View {
        if files.isEmpty {
            Text("No files found.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .padding(16)
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(files) { file in
                    HStack {
                        Image(systemName: "doc")
                            .foregroundStyle(.secondary)
                        Text(file.name)
                            .font(.system(.callout, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Text(file.humanSize)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 6)
                    if file.id != files.last?.id {
                        Divider()
                    }
                }
            }
            .padding(16)
        }
    }

    // MARK: - Helpers

    /// Same three-way label as `LocalModelRow.formatLabel`.
    private var formatLabel: String {
        switch model.format {
        case .mlxVLM: return "Vision"
        case .embedder: return "Embed"
        default: return "MLX"
        }
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.12), in: Capsule())
    }

    /// LoRA adapters whose declared base model matches this card's model.
    ///
    /// Mirrors `ModelLibraryViewModel.isDownloaded(_:)`'s HF-id reconciliation:
    /// adapters record their target as a full HF id (`"org/name"`, from PEFT's
    /// `base_model_name_or_path`), while a normally-scanned `LocalModel.id` is
    /// just the bare directory name — so an exact match AND a last-path-
    /// component match both count. Exact match also covers Track F's
    /// HF-cache-discovered models, whose `id` already carries the full "org/name".
    private var compatibleAdapters: [LocalAdapter] {
        allAdapters.filter { adapter in
            guard let target = adapter.targetModel else { return false }
            if target == model.id { return true }
            let targetLeaf = target.split(separator: "/").last.map(String.init) ?? target
            return targetLeaf == model.id
        }
    }

    /// Collapse the user's real home to `~`, matching `ModelLibraryView`'s
    /// equivalent helper for the empty-state path.
    private static func displayPath(_ url: URL) -> String {
        let raw = url.path(percentEncoded: false)
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return raw.hasPrefix(home) ? "~" + raw.dropFirst(home.count) : raw
    }

    private struct FileEntry: Identifiable {
        var id: String { name }
        let name: String
        let sizeBytes: Int64

        /// Same base-10 formatting convention as `LocalModel.humanSize`.
        var humanSize: String {
            let bytes = Double(sizeBytes)
            if bytes >= 1_000_000_000 { return String(format: "%.2f GB", bytes / 1_000_000_000) }
            if bytes >= 1_000_000 { return String(format: "%.0f MB", bytes / 1_000_000) }
            if bytes >= 1_000 { return String(format: "%.0f KB", bytes / 1_000) }
            return "\(sizeBytes) B"
        }
    }

    private static func listFiles(in directory: URL) -> [FileEntry] {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return urls
            .map { url -> FileEntry in
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                return FileEntry(name: url.lastPathComponent, sizeBytes: Int64(size))
            }
            .sorted { $0.name.localizedCompare($1.name) == .orderedAscending }
    }
}

#Preview {
    ModelCardView(
        model: LocalModel(
            id: "mlx-community/Qwen3-8B-4bit",
            displayName: "Qwen3-8B-4bit",
            directory: URL(filePath: "/tmp/Qwen3-8B-4bit"),
            sizeBytes: 4_500_000_000,
            format: .mlx,
            quantization: "4bit",
            parameterCount: "8B",
            architecture: "qwen3"
        ),
        allAdapters: []
    )
}
