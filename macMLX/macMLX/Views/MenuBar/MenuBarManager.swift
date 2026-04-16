// MenuBarManager.swift
// macMLX
//
// Owns the NSStatusItem + NSPopover for the menu bar icon.
// Instantiated once by AppDelegate and kept alive for the app's lifetime.

import AppKit
import SwiftUI
import MacMLXCore

@MainActor
final class MenuBarManager {

    // MARK: - Private state

    private var statusItem: NSStatusItem?
    private var popover: NSPopover?
    private var appState: AppState?

    // MARK: - Setup

    func setup(appState: AppState) {
        self.appState = appState
        setupStatusItem()
        setupPopover(appState: appState)
    }

    // MARK: - Status item

    private func setupStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = item.button {
            // Template image so macOS auto-inverts for light/dark menu bar.
            button.image = NSImage(systemSymbolName: "cpu", accessibilityDescription: "macMLX")
            button.image?.isTemplate = true
            button.action = #selector(togglePopover(_:))
            button.target = self
        }

        statusItem = item
    }

    // MARK: - Popover

    private func setupPopover(appState: AppState) {
        let p = NSPopover()
        p.contentSize = NSSize(width: 280, height: 200)
        p.behavior = .transient
        p.animates = true
        p.contentViewController = NSHostingController(
            rootView: MenuBarPopoverView()
                .environment(appState)
        )
        popover = p
    }

    // MARK: - Toggle

    @objc private func togglePopover(_ sender: NSStatusBarButton) {
        guard let popover, let button = statusItem?.button else { return }
        if popover.isShown {
            popover.performClose(sender)
        } else {
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        }
    }
}
