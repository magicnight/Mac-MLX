// MainWindowView.swift
// macMLX
//
// 3-column NavigationSplitView root following the macOS HIG (per
// .claude/ui-guidelines.md). Sidebar selects a feature tab; detail pane
// hosts the corresponding feature view.

import SwiftUI
import MacMLXCore

struct MainWindowView: View {

    @Environment(AppState.self) private var appState
    @State private var selectedTab: Tab = .library

    /// Sidebar tabs. Order matches the spec's reference apps (Reeder / Lasso / Sleeve).
    enum Tab: String, CaseIterable, Identifiable {
        case library   = "Models"
        case chat      = "Chat"
        case benchmark = "Benchmark"
        case settings  = "Settings"

        var id: String { rawValue }

        var systemImage: String {
            switch self {
            case .library:   "tray.full"
            case .chat:      "bubble.left.and.bubble.right"
            case .benchmark: "stopwatch"
            case .settings:  "gear"
            }
        }
    }

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .frame(minWidth: 800, minHeight: 500)
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(Tab.allCases, selection: $selectedTab) { tab in
            Label(tab.rawValue, systemImage: tab.systemImage)
                .tag(tab)
        }
        .listStyle(.sidebar)
        .navigationTitle("macMLX")
        .navigationSplitViewColumnWidth(min: 180, ideal: 200, max: 240)
        .safeAreaInset(edge: .bottom) {
            statusFooter
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
        }
    }

    /// Compact engine status indicator pinned at the bottom of the sidebar.
    /// Mirrors the menu bar icon's color semantics from `ui-guidelines.md`.
    private var statusFooter: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(statusLabel)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var statusColor: Color {
        switch appState.coordinator.status {
        case .idle:        return .gray
        case .loading:     return .orange
        case .ready:       return .green
        case .generating:  return .green
        case .error:       return .red
        }
    }

    private var statusLabel: String {
        switch appState.coordinator.status {
        case .idle:
            return "Idle"
        case .loading(model: let modelID):
            return "Loading \(modelID)…"
        case .ready(model: let modelID):
            return modelID
        case .generating:
            return "Generating…"
        case .error(let message):
            return "Error: \(message)"
        }
    }

    // MARK: - Detail

    @ViewBuilder
    private var detail: some View {
        switch selectedTab {
        case .library:   ModelLibraryView()
        case .chat:      ChatView()
        case .benchmark: BenchmarkView()
        case .settings:  SettingsView()
        }
    }
}

#Preview {
    MainWindowView()
        .environment(AppState())
}
