// HuggingFaceCacheSection.swift
// macMLX
//
// Settings section for Track F's Hugging Face cache discovery: an opt-in
// toggle plus a user-editable list of cache root directories to scan.
// Scanning itself lives in MacMLXCore (`ModelLibraryManager.scanHuggingFaceCache`);
// this view is purely the switch + directory-list UI, mirroring the split
// already used by `KVCacheSection` / `ModelPoolSection`.

import SwiftUI
import MacMLXCore

struct HuggingFaceCacheSection: View {
    @Binding var enabled: Bool
    @Binding var directories: [URL]

    var body: some View {
        Section {
            Toggle("Scan Hugging Face cache for models", isOn: $enabled)

            if enabled {
                if directories.isEmpty {
                    Text("No directories configured — add one below.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(directories, id: \.path) { directory in
                        HStack {
                            Text(Self.displayPath(directory))
                                .font(.system(.caption, design: .monospaced))
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Button {
                                directories.removeAll { $0 == directory }
                            } label: {
                                Image(systemName: "minus.circle")
                            }
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                            .help("Stop scanning this directory")
                        }
                    }
                }

                Button("Add Directory…") { addDirectory() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        } header: {
            Text("Hugging Face Cache")
        } footer: {
            Text("Discovers MLX-format models already cached by other Hugging Face tools (transformers, huggingface-cli) without copying them — the library references the cache in place. Default: ~/.cache/huggingface/hub.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func addDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Choose Hugging Face Cache Directory"
        if panel.runModal() == .OK, let url = panel.url, !directories.contains(url) {
            directories.append(url)
        }
    }

    /// Collapse the user's real home to `~`, matching
    /// `ModelLibraryView`'s equivalent helper for the empty-state path.
    private static func displayPath(_ url: URL) -> String {
        let raw = url.path(percentEncoded: false)
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return raw.hasPrefix(home) ? "~" + raw.dropFirst(home.count) : raw
    }
}

#Preview {
    Form {
        HuggingFaceCacheSection(
            enabled: .constant(true),
            directories: .constant([Settings.defaultHuggingFaceCacheDirectory])
        )
    }
    .formStyle(.grouped)
    .frame(width: 500, height: 300)
}
