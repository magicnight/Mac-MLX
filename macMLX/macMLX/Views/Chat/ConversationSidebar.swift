// ConversationSidebar.swift
// macMLX
//
// Conversation history sidebar for the Chat tab.
//
// Manual VStack (no List(selection:)) because macOS List with a
// selection binding has a long-standing quirk: right-clicking an
// already-selected row stashes the contextMenu's Button action in a
// different event loop, swallowing state mutations like
// `deletingID = convo.id` before a confirmationDialog can present.
// Custom Button rows avoid that whole subsystem — each row handles
// its own click and contextMenu in a plain SwiftUI context.

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

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            list
        }
        .frame(minWidth: 200, idealWidth: 240)
        .background(Color(.windowBackgroundColor))
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
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            // Manual ScrollView + LazyVStack — `List.sidebar` style
            // outside a NavigationSplitView renders empty, and we
            // don't want List(selection:) which swallowed contextMenu
            // actions on focused rows. Plain Views + onTapGesture +
            // contextMenu on the row itself is the reliable pattern.
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(viewModel.conversations) { convo in
                        row(convo)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 4)
            }
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
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
        } else {
            conversationRow(convo)
        }
    }

    @ViewBuilder
    private func conversationRow(_ convo: Conversation) -> some View {
        let isSelected = convo.id == viewModel.currentConversationID

        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(convo.title)
                    .font(.body)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .foregroundStyle(isSelected ? Color.white : Color.primary)
                Text(convo.updatedAt, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(isSelected ? Color.white.opacity(0.85) : .secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.accentColor : Color.clear)
        )
        .contentShape(RoundedRectangle(cornerRadius: 6))
        // Single tap for row switch. Double-tap rename dropped — gesture
        // disambiguation with count:1/count:2 + contextMenu on macOS
        // SwiftUI was intermittently swallowing the contextMenu's Delete
        // button action when the row was currently selected. Rename is
        // now context-menu only (Finder convention).
        .onTapGesture {
            Task { await viewModel.switchTo(convo.id) }
        }
        .contextMenu {
            Button {
                startRename(convo)
            } label: {
                Label("Rename…", systemImage: "pencil")
            }
            Button(role: .destructive) {
                Task { await viewModel.deleteConversation(convo.id) }
            } label: {
                Label("Delete", systemImage: "trash")
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
