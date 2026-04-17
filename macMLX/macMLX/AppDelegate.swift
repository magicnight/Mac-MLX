// AppDelegate.swift
// macMLX
//
// NSApplicationDelegate: initialises MenuBarManager and (later) Sparkle.
// Bridged into the SwiftUI lifecycle via @NSApplicationDelegateAdaptor.

import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {

    // Hold a strong reference so the manager is never released.
    // Marking the class @MainActor lets this default value run in the
    // right isolation context (required by Swift 6.1+; 6.3 is more
    // lenient but CI runs 6.1).
    let menuBarManager = MenuBarManager()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // MenuBarManager.setup(appState:) is called from macMLXApp once
        // the AppState is available (see macMLXApp.swift).
        // Nothing else to do here yet — Sparkle wired in v0.2.
    }
}
