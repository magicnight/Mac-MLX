// ConversationSidebar.swift
// macMLX
//
// Conversation history sidebar for the Chat tab (v0.3.2).
//
// Why a hand-built sidebar rather than nesting a second
// `NavigationSplitView`: the outer `MainWindowView` already owns one
// NavigationSplitView for the app-level tabs (Models / Chat / Benchmark
// / Settings). Nesting NavSplitViews produces awkward double
// disclosure chevrons and confusing focus behaviour on macOS. A plain
// collapsible pane composed into Chat's layout gives us full control.

import SwiftUI
import MacMLXCore

struct ConversationSidebar: View {

    @Bindable var viewModel: ChatViewModel

    /// ID of the conversation currently being inline-renamed (nil = no
    /// rename in progress). Driving a TextField via this state instead of
    /// an NSAlert keeps the rename inline-and-fast.
    @State private var renamingID: UUID?
    /// Buffer for the inline rename TextField.
    @State private var renameDraft: String = ""
    /// Pending delete confirmation target.
    @State private var deletingID: UUID?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            list
        }
        .frame(minWidth: 200, idealWidth: 240)
        .background(Color(.windowBackgroundColor))
        // Attached to the root VStack (not to the List inside `list`)
        // so the modifier survives List rebuilds triggered by
        // selection changes + conversations re-sort on updatedAt. Fixes
        // the "left-click then right-click delete swallowed" bug.
        .confirmationDialog(
            "Delete this conversation?",
            isPresented: Binding(
                get: { deletingID != nil },
                set: { if !$0 { deletingID = nil } }
            ),
            presenting: deletingID
        ) { id in
            Button("Delete", role: .destructive) {
                Task {
                    await viewModel.deleteConversation(id)
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("Chat history will be permanently removed.")
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Text("Chats")
                .font(.headline)
            Spacer()
            Button {
                viewModel.createNew()
            } label: {
                Image(systemName: "square.and.pencil")
            }
            .buttonStyle(.plain)
            .help("New chat (⌘N)")
            .keyboardShortcut("n", modifiers: .command)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    // MARK: - List

    @ViewBuilder
    private var list: some View {
        if viewModel.conversations.isEmpty {
            ContentUnavailableView(
                "No chats yet",
                systemImage: "bubble.left.and.bubble.right",
                description: Text("Send a message to start your first conversation.")
            )
            .frame(maxHeight: .infinity)
        } else {
            List(selection: Binding(
                get: { viewModel.currentConversationID },
                set: { newID in
                    guard let newID else { return }
                    Task { await viewModel.switchTo(newID) }
                }
            )) {
                ForEach(viewModel.conversations) { convo in
                    row(convo)
                        .tag(convo.id)
                        .listRowSeparator(.hidden)
                }
            }
            .listStyle(.sidebar)
        }
    }

    // MARK: - Row

    @ViewBuilder
    private func row(_ convo: Conversation) -> some View {
        if renamingID == convo.id {
            // Inline rename TextField.
            TextField("Title", text: $renameDraft, onCommit: {
                commitRename(for: convo.id)
            })
            .textFieldStyle(.roundedBorder)
            .onExitCommand { cancelRename() }
            .padding(.vertical, 2)
        } else {
            VStack(alignment: .leading, spacing: 2) {
                Text(convo.title)
                    .font(.body)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text(convo.updatedAt, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
            .contextMenu {
                Button {
                    startRename(convo)
                } label: {
                    Label("Rename…", systemImage: "pencil")
                }
                Button(role: .destructive) {
                    deletingID = convo.id
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            }
            .onTapGesture(count: 2) {
                startRename(convo)
            }
        }
    }

    // MARK: - Rename helpers

    private func startRename(_ convo: Conversation) {
        renameDraft = convo.title
        renamingID = convo.id
    }

    private func cancelRename() {
        renamingID = nil
        renameDraft = ""
    }

    private func commitRename(for id: UUID) {
        let title = renameDraft
        renamingID = nil
        renameDraft = ""
        Task { await viewModel.rename(id, to: title) }
    }
}
