import Foundation

/// Scans the local filesystem for MLX model directories and maintains a cached list.
public actor ModelLibraryManager {

    // MARK: - Properties

    private let fileManager: FileManager

    /// Most recent scan result. `nil` if `scan(_:)` has not been called yet.
    public private(set) var lastScan: [LocalModel] = []

    // MARK: - Init

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    // MARK: - Public API

    /// Scan `directory` for subdirectories that look like MLX models.
    ///
    /// - Each top-level subdirectory is inspected by calling `ModelFormat.detect(in:)`.
    /// - MLX models are returned as `LocalModel` values sorted by `displayName`.
    /// - GGUF and unknown directories are silently skipped (with a `print` notice for GGUF).
    /// - Hidden directories (names starting with `.`) are always skipped.
    ///
    /// - Parameter directory: The root directory to scan.
    /// - Returns: Sorted array of discovered `LocalModel` values.
    @discardableResult
    public func scan(_ directory: URL) async throws -> [LocalModel] {
        let contents = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        var results: [LocalModel] = []

        for itemURL in contents {
            guard try isDirectory(itemURL) else { continue }

            let dirName = itemURL.lastPathComponent
            guard !dirName.hasPrefix(".") else { continue }

            let fileURLs = (try? fileManager.contentsOfDirectory(
                at: itemURL,
                includingPropertiesForKeys: [.fileSizeKey],
                options: [.skipsHiddenFiles]
            )) ?? []

            let fileNames = fileURLs.map { $0.lastPathComponent }
            let format = ModelFormat.detect(in: fileNames)

            switch format {
            case .mlx:
                // Peek `config.json` `model_type` — upgrade `.mlx` to
                // `.mlxVLM` (vision-language) or `.embedder` (text
                // embedding) when the directory's model_type says so.
                let upgradedFormat = upgradeFormat(directory: itemURL)
                let model = buildLocalModel(
                    dirName: dirName,
                    dirURL: itemURL,
                    fileURLs: fileURLs,
                    format: upgradedFormat
                )
                results.append(model)

            case .mlxVLM, .embedder, .reranker:
                // `ModelFormat.detect(in:)` never returns these directly
                // — they're set by `upgradeFormat` above. Reachable only
                // via tests that hand-craft a format. Fall through to the
                // same path as `.mlx`.
                let model = buildLocalModel(
                    dirName: dirName,
                    dirURL: itemURL,
                    fileURLs: fileURLs,
                    format: format
                )
                results.append(model)

            case .gguf:
                print("[ModelLibraryManager] Skipping GGUF directory: \(dirName)")

            case .unknown:
                break
            }
        }

        let sorted = results.sorted { $0.displayName.localizedCompare($1.displayName) == .orderedAscending }
        lastScan = sorted
        return sorted
    }

    /// Scans one or more Hugging Face Hub cache roots (conventionally
    /// `~/.cache/huggingface/hub`, i.e. `HF_HOME`/hub) for MLX-format model
    /// snapshots, WITHOUT copying anything — discovered entries reference
    /// the snapshot directory in place (`LocalModel.directory` points
    /// straight into the cache; `LocalModel.isExternalReference` is `true`).
    ///
    /// HF's on-disk cache layout is `<root>/models--<org>--<name>/snapshots/<revision>/`,
    /// with the actual weight files typically symlinked into a shared,
    /// content-addressed `blobs/` directory one level up. A single repo
    /// commonly has MULTIPLE cached snapshot revisions on disk at once
    /// (e.g. after a `git pull`-style re-download) — exactly one becomes a
    /// `LocalModel` per repo (see `currentSnapshotDirectory`), never one per
    /// revision, since duplicate `id`s in the same array is undefined
    /// behavior for SwiftUI's `List`/`ForEach(id:)`.
    ///
    /// Best-effort at every level: a root directory that doesn't exist or
    /// can't be read is silently skipped (rather than aborting the whole
    /// multi-root scan) since callers commonly keep the default HF path
    /// configured even for users who've never used `transformers` /
    /// `huggingface_hub`. Likewise, one unreadable `models--*` entry or
    /// snapshot revision is skipped rather than failing the entire scan —
    /// the same tolerance `scan(_:)` already shows for unreadable model
    /// subdirectories.
    ///
    /// - Parameter directories: Cache root directories to scan (user-
    ///   editable list in Settings; the default seed is the standard
    ///   `~/.cache/huggingface/hub` path).
    /// - Returns: Discovered models, sorted by `displayName`.
    @discardableResult
    public func scanHuggingFaceCache(directories: [URL]) async -> [LocalModel] {
        var results: [LocalModel] = []

        for root in directories {
            guard let modelDirs = try? fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else { continue }

            for modelDir in modelDirs {
                let folderName = modelDir.lastPathComponent
                guard folderName.hasPrefix("models--"),
                      (try? isDirectory(modelDir)) == true,
                      let repoID = Self.repoID(fromCacheFolderName: folderName)
                else { continue }

                let snapshotsDir = modelDir.appendingPathComponent("snapshots")
                guard let snapshots = try? fileManager.contentsOfDirectory(
                    at: snapshotsDir,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles]
                ) else { continue }

                let snapshotDirs = snapshots.filter { (try? isDirectory($0)) == true }
                guard let snapshotDir = currentSnapshotDirectory(
                    modelDir: modelDir, snapshotDirs: snapshotDirs
                ) else { continue }

                let fileURLs = (try? fileManager.contentsOfDirectory(
                    at: snapshotDir,
                    includingPropertiesForKeys: [.fileSizeKey],
                    options: [.skipsHiddenFiles]
                )) ?? []
                let fileNames = fileURLs.map { $0.lastPathComponent }
                guard ModelFormat.detect(in: fileNames) == .mlx else { continue }

                let upgradedFormat = upgradeFormat(directory: snapshotDir)
                let model = buildLocalModel(
                    dirName: repoID,
                    dirURL: snapshotDir,
                    fileURLs: fileURLs,
                    format: upgradedFormat,
                    isExternalReference: true
                )
                results.append(model)
            }
        }

        // De-duplicate by id (first configured root wins): a model resolvable under
        // two overlapping cache roots would otherwise yield two entries with the same
        // id, breaking SwiftUI List identity downstream.
        var seenIDs = Set<String>()
        let deduped = results.filter { seenIDs.insert($0.id).inserted }
        return deduped.sorted { $0.displayName.localizedCompare($1.displayName) == .orderedAscending }
    }

    /// Deletes `model`'s on-disk directory.
    ///
    /// This is the Core-side guardrail against deleting HF-cache external
    /// references: `LocalModelRow` hides the destructive action for these
    /// and `ModelLibraryViewModel.deleteModel` guards too, but both are
    /// GUI-layer defenses that any other call site (present or future)
    /// could bypass — this actor-level check is the one guard every caller
    /// necessarily goes through.
    ///
    /// - Throws: `ModelLibraryError.cannotDeleteExternalReference` if
    ///   `model.isExternalReference` — its `directory` points straight into
    ///   the user's shared Hugging Face cache, not an app-owned copy, and
    ///   deleting it would remove files other tools (`transformers`,
    ///   `huggingface-cli`) may still rely on.
    /// - Throws: whatever `FileManager.removeItem` throws on I/O failure.
    public func delete(_ model: LocalModel) throws {
        guard !model.isExternalReference else {
            throw ModelLibraryError.cannotDeleteExternalReference(id: model.id)
        }
        try fileManager.removeItem(at: model.directory)
    }

    /// Reverses HF Hub's cache-folder naming (`models--<org>--<name>` for
    /// repo id `<org>/<name>`) back into a slash-form repo id.
    ///
    /// Splits on the FIRST `--` occurrence after the `models--` prefix —
    /// correct because HF repo namespaces (the org/user segment) never
    /// contain `--`, only the remainder (the repo name, which commonly has
    /// single `-` separators) could in principle. Returns `nil` for
    /// anything not shaped like `models--<org>--<name>` (missing prefix,
    /// or no separator between org and name).
    static func repoID(fromCacheFolderName name: String) -> String? {
        guard name.hasPrefix("models--") else { return nil }
        let remainder = name.dropFirst("models--".count)
        guard let sepRange = remainder.range(of: "--") else { return nil }
        let org = remainder[remainder.startIndex..<sepRange.lowerBound]
        let rest = remainder[sepRange.upperBound...]
        guard !org.isEmpty, !rest.isEmpty else { return nil }
        return "\(org)/\(rest)"
    }

    // MARK: - Private Helpers

    private func isDirectory(_ url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey])
        return values.isDirectory == true
    }

    /// Resolves the single "current" snapshot directory for a cached repo,
    /// so `scanHuggingFaceCache` surfaces exactly one `LocalModel` per repo
    /// even when multiple revisions are cached side-by-side on disk.
    ///
    /// Prefers `<modelDir>/refs/main`'s content — a commit hash HF Hub
    /// writes to record which revision is currently checked out — to
    /// locate `snapshots/<hash>` directly. Falls back to the
    /// most-recently-modified snapshot directory when `refs/main` is
    /// absent/unreadable, or names a hash with no matching directory on
    /// disk (e.g. stale after manual cache surgery). Returns `nil` only
    /// when `snapshotDirs` is empty.
    private func currentSnapshotDirectory(modelDir: URL, snapshotDirs: [URL]) -> URL? {
        guard !snapshotDirs.isEmpty else { return nil }

        let refsMainURL = modelDir.appendingPathComponent("refs").appendingPathComponent("main")
        if let data = try? Data(contentsOf: refsMainURL),
           let hash = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
           !hash.isEmpty,
           let pinned = snapshotDirs.first(where: { $0.lastPathComponent == hash }) {
            return pinned
        }

        return snapshotDirs.max { modificationDate($0) < modificationDate($1) }
    }

    private func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            ?? .distantPast
    }

    private func buildLocalModel(
        dirName: String,
        dirURL: URL,
        fileURLs: [URL],
        format: ModelFormat = .mlx,
        isExternalReference: Bool = false
    ) -> LocalModel {
        // Sum all .safetensors files for reported size. Resolve symlinks
        // first: `URLResourceValues.fileSize` does NOT follow a symlink on
        // its own — querying it directly on a symlink URL returns the
        // link's own (near-zero) size, not the linked-to file's. Real HF
        // Hub cache layouts store weight files as symlinks into a shared
        // `blobs/` directory, so without this every HF-cache-discovered
        // model would report a bogus, tiny size instead of its real one.
        // A no-op for the common `scan(_:)` case of plain (non-symlinked)
        // files — resolving a regular file's path still names the same file.
        let sizeBytes: Int64 = fileURLs
            .filter { $0.pathExtension.lowercased() == "safetensors" }
            .compactMap { url -> Int64? in
                let resolved = url.resolvingSymlinksInPath()
                guard let values = try? resolved.resourceValues(forKeys: [.fileSizeKey]),
                      let size = values.fileSize else { return nil }
                return Int64(size)
            }
            .reduce(0, +)

        // Single config.json peek, reused for quantization + architecture.
        let configInfo = ModelConfigInfo.read(from: dirURL, fileManager: fileManager)
        let quantization = inferQuantization(dirName: dirName, configInfo: configInfo)
        let parameterCount = Self.inferParameterCount(from: dirName)

        return LocalModel(
            id: dirName,
            displayName: dirName,
            directory: dirURL,
            sizeBytes: sizeBytes,
            format: format,
            quantization: quantization,
            parameterCount: parameterCount,
            architecture: configInfo?.modelType,
            isExternalReference: isExternalReference
        )
    }

    /// `model_type` values mlx-swift-lm's `MLXVLM` library supports.
    ///
    /// Source of truth: `Libraries/MLXVLM/Models/*.swift` registry in
    /// the mlx-swift-lm checkout. Refresh this set when bumping the
    /// SPM dependency. Stored lowercased — comparisons are
    /// case-insensitive against `config.json`.
    private static let knownVLMTypes: Set<String> = [
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5_vl",
        "gemma3",
        "smolvlm",
        "smolvlm2",
        "paligemma",
        "pixtral",
        "idefics3",
        "fast_vlm",
        "lfm2_vl",
        "glm_ocr",
        "mistral3",
    ]

    /// `model_type` values that map to MLXEmbedders' encoder registry and
    /// are unambiguously text-embedding models.
    ///
    /// Source of truth: `Libraries/MLXEmbedders/ModelFactory.swift`
    /// `EmbedderTypeRegistry`. Deliberately ONLY the encoder-only families:
    /// the registry also maps decoder families (`qwen3`, `lfm2`, `gemma3`,
    /// `gemma3_text`, `gemma3n`) to embedder variants, but those share their
    /// `model_type` with generative LLM/VLM checkpoints — `model_type` alone
    /// can't tell a Qwen3 *embedder* from a Qwen3 *chat* model — so listing
    /// them here would mis-tag ordinary chat models as embedders and break
    /// generation. They stay `.mlx` / `.mlxVLM`; precise decoder-embedder
    /// detection (e.g. inspecting `1_Pooling/`) is a follow-up. Stored
    /// lowercased — comparisons are case-insensitive against config.json.
    private static let knownEmbedderTypes: Set<String> = [
        "bert",
        "roberta",
        "xlm-roberta",
        "distilbert",
        "nomic_bert",
    ]

    /// Peek `config.json`'s `model_type` and upgrade `.mlx` to a more
    /// specific format: `.mlxVLM` for a known vision-language family, or
    /// `.embedder` for a known text-embedding family. Vision-language wins
    /// when a `model_type` appears in both registries (e.g. `gemma3`).
    /// Returns `.mlx` when the type matches neither.
    ///
    /// Best-effort: any read or parse failure (missing file, malformed
    /// JSON, missing `model_type` key) falls back to `.mlx` — the scan
    /// must not blow up because of one unparseable config.
    private func upgradeFormat(directory: URL) -> ModelFormat {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return .mlx
        }
        // Reranker wins, checked FIRST — a cross-encoder reranker
        // (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) has `model_type` `bert`
        // or `xlm-roberta`, EXACTLY the encoders `knownEmbedderTypes` below
        // would tag `.embedder`. `model_type` therefore can't separate them;
        // the `*ForSequenceClassification` head in `architectures` (a single
        // relevance logit) is the discriminator. Take the LAST architecture —
        // HF lists the concrete task head there. Case-sensitive suffix match:
        // the HF class name is `…ForSequenceClassification`.
        //
        // A `*ForSequenceClassification` architecture alone is NOT sufficient:
        // a genuine multi-class classifier (e.g. a 5-label sentiment BERT)
        // carries the same architecture suffix but is NOT a reranker.
        // `RerankEngine` always builds a single-logit `Linear(hidden, 1)`
        // head, so a multi-label checkpoint's real `classifier.weight`
        // (`[N, hidden]`, `N > 1`) would fail `verify: [.all]` with a
        // cryptic load error rather than being cleanly routed elsewhere.
        // Gate on the label count: `num_labels` (or, absent that,
        // `id2label`'s entry count) must be `1` — or absent entirely, since
        // many reranker checkpoints omit `num_labels` and rely on the HF
        // default of `1`. Only an EXPLICIT `num_labels > 1` (with no
        // contradicting `id2label` of count 1) disqualifies.
        if let architectures = json["architectures"] as? [String],
           architectures.last?.hasSuffix("ForSequenceClassification") == true {
            let numLabels = json["num_labels"] as? Int
            let id2labelCount = (json["id2label"] as? [String: Any])?.count
            if numLabels == 1 || id2labelCount == 1 || numLabels == nil {
                return .reranker
            }
            // Explicit multi-label config — fall through to the model_type
            // checks below (typically lands as `.embedder`, since reranker
            // and embedder checkpoints share `model_type`).
        }
        guard let modelType = (json["model_type"] as? String)?.lowercased() else {
            return .mlx
        }
        if Self.knownVLMTypes.contains(modelType) {
            return .mlxVLM
        }
        if Self.knownEmbedderTypes.contains(modelType) {
            return .embedder
        }
        return .mlx
    }

    /// Extracts a quantization string from a directory name.
    ///
    /// Matches a trailing `-(\d+bit)` suffix, e.g. `Qwen3-8B-4bit` → `"4bit"`.
    private func extractQuantization(from name: String) -> String? {
        let pattern = #"-(\d+bit)$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return nil
        }
        let range = NSRange(name.startIndex..., in: name)
        guard let match = regex.firstMatch(in: name, options: [], range: range),
              let captureRange = Range(match.range(at: 1), in: name) else {
            return nil
        }
        return String(name[captureRange])
    }

    /// Infers a display quantization label ("4bit", "8bit", "bf16", …).
    ///
    /// Prefers `config.json`'s explicit `quantization.bits` — the most
    /// accurate source, since it reflects what's actually baked into the
    /// safetensors — falling back to the directory-name suffix convention
    /// (`extractQuantization`) when `config.json` has no `quantization`
    /// block (e.g. an unquantized bf16/fp16 export) or is missing /
    /// malformed. A final name-only pass catches the unquantized case via
    /// its own `-bf16`/`-fp16` naming convention.
    private func inferQuantization(dirName: String, configInfo: ModelConfigInfo?) -> String? {
        if let bits = configInfo?.quantizationBits {
            return "\(bits)bit"
        }
        if let fromName = extractQuantization(from: dirName) {
            return fromName
        }
        return Self.inferUnquantizedDtype(from: dirName)
    }

    /// Name-only heuristic for the common "no quantization block" case:
    /// mlx-community tags full-precision exports with a `-bf16` / `-fp16`
    /// suffix the same way quantized ones use `-4bit` / `-8bit`.
    private static func inferUnquantizedDtype(from name: String) -> String? {
        let lower = name.lowercased()
        if lower.hasSuffix("-bf16") { return "bf16" }
        if lower.hasSuffix("-fp16") || lower.hasSuffix("-f16") { return "fp16" }
        return nil
    }

    /// Infers a human parameter-count label ("8B", "0.5B", "70B", …) from
    /// the mlx-community naming convention
    /// (`<Family>-<Size>[BM](-<quant>)?`, e.g. `Qwen3-8B-4bit`,
    /// `Qwen2.5-0.5B-Instruct-bf16`).
    ///
    /// Name-only, unlike quantization: real parameter counts aren't a
    /// standard `config.json` field (computing one from architecture dims
    /// would need per-family formulas), so this heuristic is the only
    /// practical source. Best-effort — returns `nil` for an
    /// unconventional or user-renamed directory.
    static func inferParameterCount(from name: String) -> String? {
        let pattern = #"(?:^|[-_])(\d+(?:\.\d+)?[BM])(?:[-_]|$)"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return nil
        }
        let range = NSRange(name.startIndex..., in: name)
        guard let match = regex.firstMatch(in: name, options: [], range: range),
              let captureRange = Range(match.range(at: 1), in: name) else {
            return nil
        }
        return String(name[captureRange]).uppercased()
    }
}

/// Errors thrown by `ModelLibraryManager.delete(_:)`.
public enum ModelLibraryError: LocalizedError, Equatable, Sendable {
    /// Attempted to delete a `LocalModel` whose `isExternalReference` is
    /// `true` — an HF-cache-discovered entry whose `directory` points
    /// straight into the user's shared Hugging Face cache rather than an
    /// app-owned copy.
    case cannotDeleteExternalReference(id: String)

    public var errorDescription: String? {
        switch self {
        case .cannotDeleteExternalReference(let id):
            return "'\(id)' is referenced from your Hugging Face cache and can't be deleted "
                + "from macMLX. Manage it via Finder or huggingface-cli instead."
        }
    }
}
