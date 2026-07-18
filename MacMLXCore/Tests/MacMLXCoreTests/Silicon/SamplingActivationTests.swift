// Copyright © 2026 macMLX. English comments only.

import Testing

@testable import MacMLXCore

/// The reference-counted sampling gate: two surfaces (the Activity panel and a
/// benchmark run) can need sampling at once, and the loop must run while any of
/// them does. These cover the transition edges the app shell keys its timer off,
/// and the underflow guard.
struct SamplingActivationTests {

    @Test("The first activation turns sampling on; a second does not re-trigger it")
    func firstActivationTurnsOn() {
        var gate = SamplingActivation()
        #expect(gate.isActive == false)
        #expect(gate.activate() == true)    // 0 → 1: start the loop
        #expect(gate.isActive == true)
        #expect(gate.activate() == false)   // 1 → 2: already running
        #expect(gate.count == 2)
    }

    @Test("Two activations survive one deactivation; only the last stops sampling")
    func refcountKeepsSamplingUntilZero() {
        var gate = SamplingActivation()
        gate.activate()                      // panel
        gate.activate()                      // benchmark run (overlap)
        #expect(gate.isActive == true)

        // The benchmark run finishing must NOT stop sampling for the still-open panel.
        #expect(gate.deactivate() == false)  // 2 → 1: others still need it
        #expect(gate.isActive == true)
        #expect(gate.count == 1)

        // The panel closing is the last consumer → stop the loop.
        #expect(gate.deactivate() == true)   // 1 → 0
        #expect(gate.isActive == false)
    }

    @Test("A stray deactivation cannot drive the count negative")
    func deactivateDoesNotUnderflow() {
        var gate = SamplingActivation()
        #expect(gate.deactivate() == false)  // nothing was active
        #expect(gate.count == 0)
        #expect(gate.isActive == false)
        // A subsequent real activation still tracks correctly.
        #expect(gate.activate() == true)
        #expect(gate.isActive == true)
    }
}
