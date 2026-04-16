// HFModelRow.swift
// macMLX

import SwiftUI
import MacMLXCore

struct HFModelRow: View {

    let model: HFModel
    let isDownloaded: Bool
    let isDownloading: Bool
    let onDownload: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            // HF logo placeholder
            Image(systemName: "arrow.down.circle")
                .foregroundStyle(isDownloaded ? .green : .secondary)
                .frame(width: 20)

            // Model info
            VStack(alignment: .leading, spacing: 2) {
                Text(modelName)
                    .font(.subheadline.bold())
                    .lineLimit(1)

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

            Spacer()

            if isDownloading {
                ProgressView()
                    .scaleEffect(0.75)
                    .frame(width: 80)
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
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }

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
    return List {
        HFModelRow(model: model, isDownloaded: false, isDownloading: false, onDownload: {})
        HFModelRow(model: model, isDownloaded: true, isDownloading: false, onDownload: {})
        HFModelRow(model: model, isDownloaded: false, isDownloading: true, onDownload: {})
    }
    .frame(width: 500, height: 200)
}
