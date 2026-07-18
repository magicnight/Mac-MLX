// KVCacheSection.swift
// macMLX
//
// Settings section exposing the KV-cache-tiering knobs: hot (RAM)
// and cold (SSD) budget sliders plus a "Clear All KV Caches" button
// that drops both tiers.
//
// The sliders persist to `Settings.kvCacheHotMB` / `Settings.kvCacheColdGB`
// and are wired into the engine's `PromptCacheStore` via `PromptCacheConfig`:
// the hot slider is the primary in-RAM byte budget; the cold slider caps the
// on-disk directory, enforced automatically by `pruneCold` (mtime-LRU).

import SwiftUI
import MacMLXCore

struct KVCacheSection: View {
    @Binding var hotMB: Int
    @Binding var coldGB: Int
    @Binding var coldEnabled: Bool
    var onClearCache: () -> Void

    var body: some View {
        Section("KV Cache") {
            HStack {
                Text("Hot (RAM)")
                Spacer()
                Stepper(
                    value: $hotMB,
                    in: 128...8192,
                    step: 128
                ) {
                    Text(String(hotMB) + " MB")
                        .font(.system(.body, design: .monospaced))
                        .frame(minWidth: 80, alignment: .trailing)
                }
                .help("Primary in-RAM budget for reusable prompt-cache KV state. Evicted entries spill to the cold (SSD) tier. Applies to models loaded after the change.")
            }

            HStack {
                Text("Cold (SSD)")
                Spacer()
                Stepper(
                    value: $coldGB,
                    in: 1...500,
                    step: 1
                ) {
                    Text(String(coldGB) + " GB")
                        .font(.system(.body, design: .monospaced))
                        .frame(minWidth: 80, alignment: .trailing)
                }
                .help("On-disk cap for the cold KV-cache tier, pruned automatically (oldest first) once exceeded. Applies to models loaded after the change; use Clear All below to reclaim everything now.")
            }
            .disabled(!coldEnabled)

            Toggle("Spill to cold (SSD) tier", isOn: $coldEnabled)
                .help("When off, the KV cache stays in RAM only and nothing spills to disk. Existing cold files remain on disk until you Clear All.")

            HStack {
                Spacer()
                Button("Clear All KV Caches", action: onClearCache)
                    .foregroundStyle(.red)
            }
        }
    }
}
