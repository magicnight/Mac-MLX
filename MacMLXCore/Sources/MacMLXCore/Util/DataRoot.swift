import Foundation

/// Locations under the user's **real** `$HOME`, not the App Sandbox
/// container.
///
/// Why this exists as a single source of truth:
/// - Under App Sandbox, `FileManager.default.homeDirectoryForCurrentUser`
///   resolves to `~/Library/Containers/<bundle-id>/Data/`. Writes there
///   are invisible in Finder and don't line up with the CLI binary,
///   which runs outside the sandbox and sees the real `$HOME`.
/// - macOS's sandbox has a long-standing *dotfile exemption* — directories
///   under real `$HOME` whose name starts with `.` are writable from a
///   sandboxed app without needing `user-selected.read-write` entitlements
///   or security-scoped bookmarks. So `~/.mac-mlx/` both works under
///   sandbox AND is visible + shared with the CLI.
/// - `NSHomeDirectoryForUser(NSUserName())` consults Directory Service
///   for the logged-in user's `pw_dir`, which is `/Users/<name>` even
///   inside a sandboxed process — exactly what we want.
///
/// This helper replaces five inlined copies of the same
/// `NSHomeDirectoryForUser ?? NSHomeDirectory` dance that shipped in
/// v0.1–v0.2 (SettingsManager, ConversationStore, ModelParametersStore,
/// BenchmarkStore, HFDownloader). Also helps the app-target callers
/// (onboarding, settings view) do the right thing under sandbox.
public enum DataRoot {

    /// Real user home directory (`/Users/<name>`), bypassing the sandbox
    /// container redirect. Falls back to `NSHomeDirectory()` only if the
    /// Directory-Service lookup fails (shouldn't happen on a configured
    /// Mac).
    public static var userHome: URL {
        if let path = NSHomeDirectoryForUser(NSUserName()) {
            return URL(filePath: path, directoryHint: .isDirectory)
        }
        return URL(filePath: NSHomeDirectory(), directoryHint: .isDirectory)
    }

    /// `~/.mac-mlx/` — the app's data root. Dotfile-exempt under sandbox.
    public static var macMLX: URL {
        userHome.appending(path: ".mac-mlx", directoryHint: .isDirectory)
    }

    /// `~/.mac-mlx/<subpath>` — convenience for store-root construction.
    public static func macMLX(_ subpath: String) -> URL {
        macMLX.appending(path: subpath, directoryHint: .isDirectory)
    }
}
