// LogsView.swift
// macMLX
//
// Native macOS log viewer for issue #16. Reads directly from Pulse's
// `LoggerStore` Core Data stack (the same store `LogManager` writes
// into) and renders with SwiftUI `Table`.
//
// Why not `PulseUI.ConsoleView`:
// The PulseUI 5.x ConsoleView is `#if !os(macOS)`-gated (iOS/tvOS/
// watchOS only). macOS users are expected to open exported `.pulse`
// bundles in the standalone "Pulse for Mac" app. That's not useful
// as an in-app logs tab — so we render our own, using Pulse's
// `LoggerMessageEntity` directly.

import SwiftUI
import CoreData
import Pulse
import MacMLXCore

struct LogsView: View {

    @Environment(AppState.self) private var appState

    @State private var messages: [LoggerMessageEntity] = []
    @State private var searchText: String = ""
    @State private var selectedLevel: LogLevel? = nil
    @State private var refreshTick: Int = 0

    private var store: LoggerStore { appState.logs.store }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            messageTable
        }
        .navigationTitle("Logs")
        .onAppear { refresh() }
        .onChange(of: searchText) { _, _ in refresh() }
        .onChange(of: selectedLevel) { _, _ in refresh() }
        // Poll the store every second so new log entries surface without
        // forcing LogManager to flush() on every write. 1 Hz is fine —
        // log traffic is bursty, not latency-sensitive.
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                refresh()
            }
        }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 12) {
            TextField("Search messages…", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 320)

            Picker("Level", selection: $selectedLevel) {
                Text("All").tag(LogLevel?.none)
                ForEach(LogLevel.allCases, id: \.self) { level in
                    Text(level.rawValue.capitalized).tag(LogLevel?.some(level))
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 140)

            Spacer()

            Text("\(messages.count.formatted()) entries")
                .font(.caption)
                .foregroundStyle(.secondary)

            Button {
                clearLogs()
            } label: {
                Label("Clear", systemImage: "trash")
            }
            .buttonStyle(.bordered)
            .help("Delete all stored log entries.")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: - Table

    private var messageTable: some View {
        Table(messages) {
            TableColumn("Time") { msg in
                Text(msg.createdAt.formatted(date: .omitted, time: .standard))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            .width(min: 90, ideal: 110, max: 130)

            TableColumn("Level") { msg in
                levelBadge(for: msg.level)
            }
            .width(min: 70, ideal: 80, max: 90)

            TableColumn("Category") { msg in
                Text(msg.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .width(min: 80, ideal: 100, max: 140)

            TableColumn("Message") { msg in
                Text(msg.text)
                    .lineLimit(3)
                    .textSelection(.enabled)
            }
        }
    }

    // MARK: - Level badge

    @ViewBuilder
    private func levelBadge(for raw: Int16) -> some View {
        let level = LoggerStore.Level(rawValue: raw)
        let (label, color): (String, Color) = {
            switch level {
            case .trace:    return ("TRACE",    .secondary)
            case .debug:    return ("DEBUG",    .secondary)
            case .info:     return ("INFO",     .blue)
            case .notice:   return ("NOTICE",   .teal)
            case .warning:  return ("WARN",     .orange)
            case .error:    return ("ERROR",    .red)
            case .critical: return ("CRIT",     .pink)
            case .none:     return ("?",        .gray)
            }
        }()
        Text(label)
            .font(.system(.caption2, design: .monospaced).bold())
            .foregroundStyle(.white)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color, in: RoundedRectangle(cornerRadius: 4))
    }

    // MARK: - Fetch

    private func refresh() {
        let request = NSFetchRequest<LoggerMessageEntity>(entityName: "LoggerMessageEntity")
        request.sortDescriptors = [
            NSSortDescriptor(keyPath: \LoggerMessageEntity.createdAt, ascending: false)
        ]
        // Cap to keep scroll responsive. Older entries still persisted on
        // disk — adjust as needed. Pulse's own ConsoleView paginates;
        // we're simpler and just fetch the newest 2000.
        request.fetchLimit = 2000

        var predicates: [NSPredicate] = []
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            predicates.append(NSPredicate(format: "text CONTAINS[cd] %@", trimmed))
        }
        if let level = selectedLevel {
            predicates.append(NSPredicate(format: "level == %d", level.pulse.rawValue))
        }
        if !predicates.isEmpty {
            request.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        }

        let context = store.viewContext
        // Pulse's store merges from background context periodically; force
        // a refresh so very recent writes (<1s old) become visible.
        context.refreshAllObjects()
        messages = (try? context.fetch(request)) ?? []
        _ = refreshTick  // silence "let but never read" if compiler nags
    }

    private func clearLogs() {
        let context = store.viewContext
        let delete = NSBatchDeleteRequest(
            fetchRequest: NSFetchRequest(entityName: "LoggerMessageEntity")
        )
        delete.resultType = .resultTypeObjectIDs
        context.performAndWait {
            _ = try? context.execute(delete)
            context.refreshAllObjects()
        }
        refresh()
    }
}

// MARK: - LogLevel.pulse helper (re-export for viewer)
// `LogLevel.pulse` is internal to MacMLXCore (see LogManager.swift). We
// duplicate the mapping here so the viewer's level filter lines up
// without expanding MacMLXCore's public API surface.
private extension LogLevel {
    var pulse: LoggerStore.Level {
        switch self {
        case .debug:    return .debug
        case .info:     return .info
        case .warning:  return .warning
        case .error:    return .error
        case .critical: return .critical
        }
    }
}

#Preview {
    LogsView()
        .environment(AppState())
        .frame(width: 900, height: 500)
}
