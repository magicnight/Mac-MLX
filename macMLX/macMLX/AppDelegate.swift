// AppDelegate.swift
// macMLX
//
// NSApplicationDelegate: initialises MenuBarManager and (later) Sparkle.
// Bridged into the SwiftUI lifecycle via @NSApplicationDelegateAdaptor.

import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {

    // Hold a strong reference so the manager is never released.
    let menuBarManager = MenuBarManager()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // MenuBarManager.setup(appState:) is called from macMLXApp once
        // the AppState is available (see macMLXApp.swift).
        // Nothing else to do here yet — Sparkle wired in v0.2.
    }
}
