// MenuBarPopoverView.swift
// macMLX
//
// SwiftUI content shown inside the NSPopover. Observes AppState /
// EngineCoordinator for live status updates.

import SwiftUI
import MacMLXCore

struct MenuBarPopoverView: View {

    @Environment(AppState.self) private var appState
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            statsGrid
            Divider()
            actionButtons
        }
        .padding(16)
        .frame(width: 280)
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Image(systemName: "cpu")
                .foregroundStyle(statusColor)
            Text("macMLX")
                .font(.headline)
            Spacer()
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
        }
    }

    // MARK: - Stats grid

    private var statsGrid: some View {
        VStack(alignment: .leading, spacing: 6) {
            statRow(label: "Status", value: statusLabel)
            statRow(label: "Model", value: modelLabel)
            statRow(label: "Memory", value: memoryLabel)
            statRow(label: "Tokens today", value: tokensLabel)
        }
    }

    private func statRow(label: String, value: String) -> some View {
        HStack {
            Text(label + ":")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 90, alignment: .leading)
            Text(value)
                .font(.caption)
                .foregroundStyle(.primary)
            Spacer()
        }
    }

    // MARK: - Action buttons

    private var actionButtons: some View {
        HStack(spacing: 8) {
            startStopButton
            openButton
        }
    }

    private var startStopButton: some View {
        let isRunning = appState.coordinator.status.isLoaded
        return Button(isRunning ? "Stop" : "Start") {
            Task {
                if isRunning {
                    await appState.coordinator.unload()
                }
                // "Start" requires a model selection — handled in main window.
            }
        }
        .frame(maxWidth: .infinity)
        .buttonStyle(.bordered)
    }

    private var openButton: some View {
        Button("Open") {
            NSApp.activate(ignoringOtherApps: true)

            // Look for an existing main-window instance we can just raise.
            // SwiftUI-managed windows have their group's identifier in
            // `windowNumber`-unrelated fields; we match by NSWindow kind +
            // visibility. If none exists (the user closed via red traffic
            // light under LSUIElement), re-create via openWindow(id:).
            let existing = NSApp.windows.first { win in
                win.isVisible && win.canBecomeMain && !win.title.isEmpty
            }
            if let existing {
                existing.makeKeyAndOrderFront(nil)
            } else {
                openWindow(id: "main")
            }
        }
        .frame(maxWidth: .infinity)
        .buttonStyle(.borderedProminent)
    }

    // MARK: - Computed labels

    private var statusColor: Color {
        switch appState.coordinator.status {
        case .idle:       return .gray
        case .loading:    return .orange
        case .ready:      return .green
        case .generating: return .green
        case .error:      return .red
        }
    }

    private var statusLabel: String {
        switch appState.coordinator.status {
        case .idle:                    return "Idle"
        case .loading(model: _):       return "Loading…"
        case .ready(model: _):         return "Ready"
        case .generating:              return "Generating…"
        case .error(let msg):          return "Error: \(msg)"
        }
    }

    private var modelLabel: String {
        appState.coordinator.currentModel?.displayName ?? "—"
    }

    private var memoryLabel: String {
        let total = MemoryProbe.totalMemoryGB()
        // Used memory is not trivially available at v0.1 — show total only.
        return String(format: "%.0f GB total", total)
    }

    private var tokensLabel: String {
        let count = appState.coordinator.tokensGeneratedTotal
        return count == 0 ? "0" : count.formatted(.number)
    }
}

#Preview {
    MenuBarPopoverView()
        .environment(AppState())
}
