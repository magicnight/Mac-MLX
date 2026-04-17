import Testing
@testable import MacMLXCore

@Test
func engineStatusEqualityCoversAllCases() {
    #expect(EngineStatus.idle == .idle)
    #expect(EngineStatus.loading(model: "x") == .loading(model: "x"))
    #expect(EngineStatus.loading(model: "x") != .loading(model: "y"))
    #expect(EngineStatus.ready(model: "x") == .ready(model: "x"))
    #expect(EngineStatus.generating == .generating)
    #expect(EngineStatus.error("boom") == .error("boom"))
    #expect(EngineStatus.error("a") != .error("b"))
    #expect(EngineStatus.idle != .generating)
}

@Test
func engineStatusIsLoadedCoversReadyAndGenerating() {
    // A model is "loaded" from the UI's perspective whenever there is a
    // model in memory — both `.ready` (idle between turns) and
    // `.generating` (actively producing tokens). Pre-v0.3.1 this only
    // returned `true` for `.ready`, which caused the "No model loaded"
    // banner to flicker on for every send → first-token window.
    #expect(EngineStatus.idle.isLoaded == false)
    #expect(EngineStatus.loading(model: "x").isLoaded == false)
    #expect(EngineStatus.ready(model: "x").isLoaded == true)
    #expect(EngineStatus.generating.isLoaded == true)
    #expect(EngineStatus.error("x").isLoaded == false)
}
