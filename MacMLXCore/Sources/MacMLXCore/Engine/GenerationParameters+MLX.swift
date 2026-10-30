import MLXLMCommon

extension GenerationParameters {
    /// This request's parameters as mlx-swift-lm's `GenerateParameters`. The
    /// single-stream and the batched path both build theirs here, so the two
    /// cannot sample differently.
    ///
    /// temperature/topP: our values are Double, MLXLLM uses Float, and the
    /// temperature is `samplingTemperature`, so `top_p: 0` samples greedily.
    /// KV-cache quantization (Track E): nil defers to the upstream defaults
    /// (no quantization; group size 64; start 0). These are honored on the
    /// stock LLM path, the speculative path, and the VLM path — all of which
    /// build their `TokenIterator` from these parameters. They are NOT honored
    /// on the custom-processor path (logit_bias / XTC / logprobs / constrained),
    /// whose `TokenIterator(input:model:cache:processor:sampler:…)` initializer
    /// carries no quantization parameters; that combination is logged and the
    /// KV-quant request silently deferred (see `runLLMGeneration`). A request
    /// with `kvBits` is never batched, and the batched path builds its own
    /// caches either way.
    /// `chunking: .remainder` restores the legacy prefill stride and reserved
    /// tail that mlx-swift-lm 3.32.3 replaced with `.balanced` by default.
    /// Checked by reading against 3.31.4 for the generic `LLMModel` path and
    /// every VLM path; it does not cover Gemma3Text, whose prefill 3.32.3
    /// reworked on its own (it now chunks at 128 regardless), and no
    /// output-level comparison was possible here. Moving to `.balanced` is a
    /// measured change of its own, not something a dependency bump decides.
    /// The batched path prefills by hand in `ModelBatchInferenceCore` and
    /// never reads the prefill setting.
    var mlxGenerateParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize ?? 64,
            quantizedKVStart: quantizedKVStart ?? 0,
            temperature: Float(samplingTemperature),
            topP: Float(topP),
            prefill: PrefillParameters(chunking: .remainder)
        )
    }
}
