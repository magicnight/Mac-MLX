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
func engineStatusIsLoadedReportsTrueOnlyForReady() {
    #expect(EngineStatus.idle.isLoaded == false)
    #expect(EngineStatus.loading(model: "x").isLoaded == false)
    #expect(EngineStatus.ready(model: "x").isLoaded == true)
    #expect(EngineStatus.generating.isLoaded == false)
    #expect(EngineStatus.error("x").isLoaded == false)
}
