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
            Divider()
            quitButton
        }
        .padding(16)
        .frame(width: 280)
    }

    // MARK: - Quit button (fix #17)
    //
    // LSUIElement apps hide the Dock icon, so ⌘Q from an unfocused state
    // doesn't reach us. The menu bar popover is the canonical escape hatch.
    private var quitButton: some View {
        Button {
            NSApp.terminate(nil)
        } label: {
            HStack {
                Image(systemName: "power")
                    .foregroundStyle(.secondary)
                Text("Quit macMLX")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("⌘Q")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .keyboardShortcut("q", modifiers: .command)
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
            statRow(label: "Server", value: serverLabel)
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
        // Button drives the HTTP server lifecycle (not the engine).
        // Engine load/unload is done from the Models tab in the main window.
        let isRunning = appState.server != nil
        let isToggling = appState.isServerToggling
        return Button {
            Task {
                if isRunning {
                    await appState.stopServer()
                } else {
                    await appState.startServer()
                }
            }
        } label: {
            HStack(spacing: 6) {
                if isToggling {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: isRunning ? "stop.circle.fill" : "play.circle.fill")
                }
                Text(isRunning ? "Stop Server" : "Start Server")
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .disabled(isToggling)
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
        // Green = server running, orange = toggling, gray = stopped,
        // red = any engine error. Engine state takes precedence only
        // when it's in the error state — a loaded-but-server-stopped
        // app is "gray/stopped" from the menu-bar glance perspective.
        if case .error = appState.coordinator.status { return .red }
        if appState.isServerToggling { return .orange }
        return appState.server != nil ? .green : .gray
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

    private var serverLabel: String {
        if let port = appState.serverPort {
            return "http://localhost:" + String(port)
        }
        return "Stopped"
    }
}

#Preview {
    MenuBarPopoverView()
        .environment(AppState())
}
