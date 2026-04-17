// HFModelRow.swift
// macMLX

import SwiftUI
import MacMLXCore

struct HFModelRow: View {

    let model: HFModel
    let isDownloaded: Bool
    let isDownloading: Bool
    /// Latest progress snapshot for this model. `nil` until the first
    /// chunk write callback fires after `onDownload` is invoked.
    let progress: DownloadProgress?
    let onDownload: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // HF icon
            Image(systemName: "arrow.down.circle")
                .foregroundStyle(isDownloaded ? .green : .secondary)
                .frame(width: 20)
                .padding(.top, 2)

            // Model info + (when downloading) progress section
            VStack(alignment: .leading, spacing: 4) {
                Text(modelName)
                    .font(.subheadline.bold())
                    .lineLimit(1)

                metadataRow

                if isDownloading {
                    progressSection
                        .padding(.top, 2)
                }
            }

            Spacer(minLength: 8)

            trailingAction
        }
        .padding(.vertical, 6)
        .contentShape(Rectangle())
    }

    // MARK: - Metadata row (author / downloads / likes)

    private var metadataRow: some View {
        HStack(spacing: 8) {
            if let author = model.author {
                Text(author)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let downloads = model.downloads {
                Label("\(downloads.formatted(.number))", systemImage: "arrow.down")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .labelStyle(.titleAndIcon)
            }
            if let likes = model.likes, likes > 0 {
                Label("\(likes)", systemImage: "heart")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .labelStyle(.titleAndIcon)
            }
        }
    }

    // MARK: - Progress section

    @ViewBuilder
    private var progressSection: some View {
        if let progress, progress.totalBytes > 0 {
            // Known-size path: bar + bytes-of-bytes + percent.
            VStack(alignment: .leading, spacing: 3) {
                ProgressView(value: progress.fractionCompleted)
                    .progressViewStyle(.linear)
                    .controlSize(.small)
                HStack(spacing: 8) {
                    Text(progress.humanProgress)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Text("·").foregroundStyle(.tertiary)
                    Text(progress.humanPercent)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(progress.completedFiles)/\(progress.totalFiles) files")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                if let currentFile = progress.currentFileName {
                    Text(currentFile)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        } else {
            // Either we haven't received the first callback yet, or HF didn't
            // expose file sizes — fall back to indeterminate bar.
            ProgressView()
                .progressViewStyle(.linear)
                .controlSize(.small)
                .padding(.top, 2)
        }
    }

    // MARK: - Trailing action (button / status)

    @ViewBuilder
    private var trailingAction: some View {
        if isDownloading {
            // Empty placeholder — progress is shown inline below metadata.
            EmptyView()
        } else if isDownloaded {
            Text("Downloaded")
                .font(.caption)
                .foregroundStyle(.green)
        } else {
            Button("Download", action: onDownload)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
        }
    }

    // MARK: - Helpers

    private var modelName: String {
        model.id.split(separator: "/").last.map(String.init) ?? model.id
    }
}

#Preview {
    let model = HFModel(
        id: "mlx-community/Qwen3-8B-4bit",
        author: "mlx-community",
        downloads: 2100,
        likes: 47,
        tags: ["mlx"],
        lastModified: nil
    )
    let inflightProgress = DownloadProgress(
        modelID: model.id,
        bytesDownloaded: 2_100_000_000,
        totalBytes: 4_500_000_000,
        completedFiles: 1,
        totalFiles: 4,
        currentFileName: "model-00002-of-00004.safetensors"
    )
    return List {
        HFModelRow(model: model, isDownloaded: false, isDownloading: false, progress: nil, onDownload: {})
        HFModelRow(model: model, isDownloaded: true, isDownloading: false, progress: nil, onDownload: {})
        HFModelRow(model: model, isDownloaded: false, isDownloading: true, progress: inflightProgress, onDownload: {})
        HFModelRow(model: model, isDownloaded: false, isDownloading: true, progress: nil, onDownload: {})
    }
    .frame(width: 560, height: 360)
}
