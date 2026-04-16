import Darwin

/// Detects whether the CLI is running in an interactive terminal session.
///
/// Used by long-running commands (`serve`, `pull`, `run`) to choose between
/// a SwiftTUI dashboard and plain stdout output.
public enum TTYDetect {
    /// `true` when stdout is connected to an interactive terminal (TTY).
    ///
    /// Non-TTY contexts: pipes (`macmlx list | jq .`), cron jobs, SSH without
    /// pseudo-TTY, and unit tests (where stdout is a pipe).
    public static var isInteractive: Bool {
        isatty(STDOUT_FILENO) != 0
    }
}
