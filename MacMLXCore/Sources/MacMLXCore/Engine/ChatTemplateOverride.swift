import Foundation

/// Resolves a per-model chat-template override, injected BEFORE template
/// compilation so a checkpoint whose own `chat_template.jinja` swift-jinja
/// cannot parse — or that a power user simply wants to replace — still
/// tokenizes and generates.
///
/// Resolution order (first hit wins):
///   1. **User file** `<model dir>/macmlx.chat_template.jinja` — a per-model
///      power-user escape hatch; works for managed model directories and
///      HuggingFace-cache snapshots alike.
///   2. **Built-in override** registered by `config.json`'s `model_type`
///      (currently only `seed_oss`, see ``SeedOssChatTemplate``).
///   3. **None** — the checkpoint's own template is used unchanged (status quo).
///
/// A model with no matching override is byte-for-byte unaffected: ``resolve``
/// returns `nil`, and the exact same template string reaches the tokenizer as
/// it does today. The override, when present, is handed to swift-transformers as
/// a `ChatTemplateArgument.literal`, which the library prefers over the config
/// template — so the substitution happens before any Jinja compilation.
enum ChatTemplateOverride {
    /// Filename a power user drops next to a model to force a chat template.
    /// Chosen to be macMLX-specific so it never collides with a checkpoint's
    /// own `chat_template.jinja`.
    static let userOverrideFilename = "macmlx.chat_template.jinja"

    /// Built-in template overrides keyed by lowercased `model_type`.
    ///
    /// Keep this list SHORT and each entry justified in its template file's
    /// header — an override is a maintenance liability that must be removed once
    /// the underlying swift-jinja limitation is lifted.
    static let builtIns: [String: String] = [
        "seed_oss": SeedOssChatTemplate.template
    ]

    /// A resolved override plus a human-readable source description (for logging
    /// — no silent behavior).
    struct Resolved: Sendable, Equatable {
        let template: String
        let source: String
    }

    /// Full resolution outcome: the override to apply (if any) PLUS, when a user
    /// override file exists on disk but could not be used, the reason — so a
    /// present-but-broken file is logged instead of silently falling through to
    /// the next resolution source. "Never silent" applies to skipped files, not
    /// just applied overrides.
    struct Resolution: Sendable, Equatable {
        /// The override to apply, or `nil` when the checkpoint's own template
        /// should be used unchanged.
        let resolved: Resolved?
        /// Non-nil ONLY when `<model dir>/macmlx.chat_template.jinja` exists but
        /// was unusable: `"empty"`, `"unreadable"`, or `"not valid UTF-8"`. `nil`
        /// when the file is absent, or when it WAS usable — that success case is
        /// instead reflected by `resolved?.source == "user file ..."`.
        let skippedUserFileReason: String?
    }

    /// Resolve the chat-template override for a model directory, or `nil` when
    /// the checkpoint's own template should be used (the common case).
    ///
    /// Thin convenience wrapper around ``resolveDetailed(modelDirectory:fileManager:modelType:)``
    /// for callers that only need the override itself, not the skip diagnostics.
    ///
    /// - Parameters:
    ///   - modelDirectory: the model's on-disk directory (managed dir or HF-cache
    ///     snapshot) — the root that holds `config.json` and, optionally, a
    ///     user `macmlx.chat_template.jinja`.
    ///   - fileManager: injectable for tests.
    ///   - modelType: injectable lowercased `model_type`; when `nil` it is read
    ///     from `<modelDirectory>/config.json` via ``ModelConfigInfo``.
    static func resolve(
        modelDirectory: URL,
        fileManager: FileManager = .default,
        modelType: String? = nil
    ) -> Resolved? {
        resolveDetailed(
            modelDirectory: modelDirectory, fileManager: fileManager, modelType: modelType
        ).resolved
    }

    /// Same resolution as ``resolve(modelDirectory:fileManager:modelType:)``, plus
    /// the reason a PRESENT user override file was skipped, if any. The loader
    /// (`HuggingFaceTokenizerLoader`) uses this richer form so it can warn about a
    /// broken user file instead of silently falling through.
    static func resolveDetailed(
        modelDirectory: URL,
        fileManager: FileManager = .default,
        modelType: String? = nil
    ) -> Resolution {
        // 1. User file override — highest precedence, when usable. Distinguish
        // "absent" (silent — the common case) from "present but unusable" (must
        // be logged by the caller): unreadable (permissions/IO), not valid UTF-8,
        // or empty.
        let userFile = modelDirectory.appendingPathComponent(userOverrideFilename)
        var skippedUserFileReason: String?
        if fileManager.fileExists(atPath: userFile.path) {
            if let data = try? Data(contentsOf: userFile) {
                if let text = String(data: data, encoding: .utf8) {
                    // Trimmed emptiness: a whitespace-only file is as unusable
                    // as a zero-byte one — accepting it would render a blank
                    // prompt with no diagnostic instead of falling through.
                    if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        return Resolution(
                            resolved: Resolved(
                                template: text, source: "user file \(userOverrideFilename)"),
                            skippedUserFileReason: nil)
                    }
                    skippedUserFileReason = "empty"
                } else {
                    skippedUserFileReason = "not valid UTF-8"
                }
            } else {
                skippedUserFileReason = "unreadable"
            }
        }

        // 2. Built-in override keyed by model_type.
        let resolvedType =
            modelType
            ?? ModelConfigInfo.read(from: modelDirectory, fileManager: fileManager)?.modelType
        if let resolvedType, let builtIn = builtIns[resolvedType] {
            return Resolution(
                resolved: Resolved(template: builtIn, source: "built-in \(resolvedType)"),
                skippedUserFileReason: skippedUserFileReason)
        }

        // 3. Checkpoint default — no override.
        return Resolution(resolved: nil, skippedUserFileReason: skippedUserFileReason)
    }
}
