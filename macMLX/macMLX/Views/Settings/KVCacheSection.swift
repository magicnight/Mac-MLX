// KVCacheSection.swift
// macMLX
//
// Settings section exposing the v0.4 KV-cache-tiering knobs: hot (RAM)
// and cold (SSD) budget sliders plus a "Clear All KV Caches" button
// that drops both tiers.
//
// MVP note: the sliders persist to `Settings.kvCacheHotMB` /
// `Settings.kvCacheColdGB` but are not yet wired into the engine's
// eviction logic. `PromptCacheStore` uses an 8-entry LRU today; a
// byte-accurate budget and automatic cold-tier pruning land in
// v0.4.0.1. See the `.help` strings below for the user-facing note.

import SwiftUI
import MacMLXCore

struct KVCacheSection: View {
    @Binding var hotMB: Int
    @Binding var coldGB: Int
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
                .help("Takes effect in v0.4.0.1 — currently capped at 8 cache entries regardless of this slider.")
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
                .help("Cold-tier cap is not enforced automatically in this MVP — use the Clear All button below to reclaim space. Automatic pruning lands in v0.4.0.1.")
            }

            HStack {
                Spacer()
                Button("Clear All KV Caches", action: onClearCache)
                    .foregroundStyle(.red)
            }
        }
    }
}
