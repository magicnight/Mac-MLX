// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Whether a checkpoint of a ``FullPathVLM`` family carries everything the
/// VLM factory needs to load it with its vision tower, read from the files
/// alone, without loading anything.
///
/// A family in ``FullPathVLM/servedFamilies`` is `.mlxVLM` only when this
/// holds. Otherwise it stays `.mlx` and loads as text through
/// `LLMModelFactory`, exactly as before the family flipped: a text-only
/// conversion that kept the multimodal `model_type` would otherwise go to
/// the VLM factory and fail, and a failed VLM load is never retried as text.
///
/// Four things must hold:
/// - `config.json` has a `vision_config` object;
/// - the weights hold the family's vision-tower tensors
///   (``FullPathVLM/Family/visionWeightPrefixes``, every one of them), as
///   listed in `model.safetensors.index.json`, or else in the headers of the
///   `.safetensors` files. A leading `model.` is dropped first, as upstream's
///   sanitizer does for raw Hugging Face layouts;
/// - the vision class loads every tensor name as it is
///   (``FullPathVLM/Family/visionClassLoads(_:config:)``);
/// - the processor metadata upstream reads declares the processor the
///   family's checkpoints load with (``FullPathVLM/Family/processorClass``).
///   Upstream prefers `preprocessor_config.json` to `processor_config.json`
///   and builds whatever `processor_class` the chosen file names, so that
///   file is the one read.
enum VisionCheckpoint {

    static func isComplete(
        directory: URL, config: [String: Any], family: FullPathVLM.Family,
        fileManager: FileManager = .default
    ) -> Bool {
        guard config["vision_config"] is [String: Any] else { return false }
        guard declaredProcessorClass(directory: directory, fileManager: fileManager)
            == family.processorClass
        else { return false }
        guard let keys = weightKeys(directory: directory, fileManager: fileManager) else {
            return false
        }
        let names = keys.map { $0.hasPrefix("model.") ? String($0.dropFirst("model.".count)) : $0 }
        return family.visionWeightPrefixes.allSatisfy { prefix in
            names.contains { $0.hasPrefix(prefix) }
        } && family.visionClassLoads(keys, config: config)
    }

    /// The `processor_class` of the processor metadata file upstream selects:
    /// `preprocessor_config.json` when it exists, else `processor_config.json`.
    /// Nil when neither exists, the chosen one does not parse, or it names
    /// no class.
    static func declaredProcessorClass(directory: URL, fileManager: FileManager = .default) -> String? {
        let candidates = ["preprocessor_config.json", "processor_config.json"]
            .map { directory.appending(component: $0) }
        guard let url = candidates.first(where: { fileManager.fileExists(atPath: $0.path) }),
            let data = try? Data(contentsOf: url),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return root["processor_class"] as? String
    }

    /// The tensor names of the checkpoint's weights: the index's `weight_map`
    /// keys when `model.safetensors.index.json` exists, else every name in
    /// the headers of its `.safetensors` files. Nil when neither can be read.
    static func weightKeys(directory: URL, fileManager: FileManager = .default) -> [String]? {
        let index = directory.appending(component: "model.safetensors.index.json")
        if fileManager.fileExists(atPath: index.path) {
            guard let data = try? Data(contentsOf: index),
                let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let map = root["weight_map"] as? [String: Any]
            else { return nil }
            return Array(map.keys)
        }
        guard let files = try? fileManager.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        else { return nil }
        let shards = files.filter { $0.pathExtension.lowercased() == "safetensors" }
        guard !shards.isEmpty else { return nil }
        var keys: [String] = []
        for shard in shards {
            guard let header = safetensorsHeaderKeys(shard) else { return nil }
            keys += header
        }
        return keys
    }

    /// Header sizes above this are refused rather than read; real headers
    /// are a few megabytes at most.
    static let maxHeaderBytes: UInt64 = 64 << 20

    /// The tensor names in a `.safetensors` file's header (an 8-byte
    /// little-endian length, then that many bytes of JSON), without the
    /// `__metadata__` entry. Nil when the header cannot be read.
    static func safetensorsHeaderKeys(_ url: URL) -> [String]? {
        guard let handle = try? FileHandle(forReadingFrom: url.resolvingSymlinksInPath()) else {
            return nil
        }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8), prefix.count == 8 else { return nil }
        let length = prefix.reversed().reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
        guard length > 0, length <= maxHeaderBytes,
            let body = try? handle.read(upToCount: Int(length)), body.count == Int(length),
            let root = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        else { return nil }
        return root.keys.filter { $0 != "__metadata__" }
    }
}
