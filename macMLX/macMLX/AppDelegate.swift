// AppDelegate.swift
// macMLX
//
// NSApplicationDelegate: initialises MenuBarManager and (when Sparkle is
// wired via SPM) the auto-updater. Bridged into the SwiftUI lifecycle via
// @NSApplicationDelegateAdaptor.

import AppKit
import SwiftUI
#if canImport(Sparkle)
import Sparkle
#endif

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {

    // Hold a strong reference so the manager is never released.
    // Marking the class @MainActor lets this default value run in the
    // right isolation context (required by Swift 6.1+; 6.3 is more
    // lenient but CI runs 6.1).
    let menuBarManager = MenuBarManager()

    #if canImport(Sparkle)
    /// Sparkle's standard updater controller. Pulls config (SUFeedURL,
    /// SUPublicEDKey, SUEnableInstallerLauncherService, ...) from Info.plist.
    /// Activated by adding the Sparkle SPM dep to this Xcode target; when
    /// the module isn't present, the entire updater layer is compiled out.
    let updaterController: SPUStandardUpdaterController = .init(
        startingUpdater: true,
        updaterDelegate: nil,
        userDriverDelegate: nil
    )

    /// Exposed for the "Check for Updates…" menu command in macMLXApp.
    var updater: SPUUpdater { updaterController.updater }
    #endif

    func applicationDidFinishLaunching(_ notification: Notification) {
        // MenuBarManager.setup(appState:) is called from macMLXApp once
        // the AppState is available (see macMLXApp.swift).
    }
}
