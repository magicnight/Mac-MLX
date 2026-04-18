// LocalModelRow.swift
// macMLX

import SwiftUI
import MacMLXCore

struct LocalModelRow: View {

    let model: LocalModel
    let isLoaded: Bool
    let isLoading: Bool
    let hasUpdateAvailable: Bool
    let onLoad: () -> Void
    let onUnload: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            // Status dot
            Circle()
                .fill(isLoaded ? Color.green : Color.clear)
                .stroke(isLoaded ? Color.green : Color.secondary, lineWidth: 1.5)
                .frame(width: 10, height: 10)

            // Model info
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .font(.subheadline.bold())
                    .lineLimit(1)

                HStack(spacing: 8) {
                    Text(model.humanSize)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if let quant = model.quantization {
                        Text(quant)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(.quaternary, in: Capsule())
                    }

                    Text("MLX")
                        .font(.caption2)
                        .foregroundStyle(.green)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Color.green.opacity(0.12), in: Capsule())

                    if hasUpdateAvailable {
                        Label("Update available", systemImage: "arrow.triangle.2.circlepath.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .labelStyle(.titleAndIcon)
                    }
                }
            }

            Spacer()

            // Actions
            if isLoading {
                ProgressView()
                    .scaleEffect(0.75)
                    .frame(width: 60)
            } else if isLoaded {
                Button("Unload", action: onUnload)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            } else {
                Button("Load", action: onLoad)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
            }

            Button(role: .destructive, action: onDelete) {
                Image(systemName: "trash")
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

#Preview {
    let model = LocalModel(
        id: "Qwen3-8B-4bit",
        displayName: "Qwen3-8B-4bit",
        directory: URL(filePath: "/tmp"),
        sizeBytes: 4_500_000_000,
        format: .mlx,
        quantization: "4bit",
        parameterCount: "8B",
        architecture: nil
    )
    return List {
        LocalModelRow(
            model: model,
            isLoaded: true,
            isLoading: false,
            hasUpdateAvailable: false,
            onLoad: {},
            onUnload: {},
            onDelete: {}
        )
        LocalModelRow(
            model: model,
            isLoaded: false,
            isLoading: false,
            hasUpdateAvailable: true,
            onLoad: {},
            onUnload: {},
            onDelete: {}
        )
    }
    .frame(width: 500, height: 200)
}
