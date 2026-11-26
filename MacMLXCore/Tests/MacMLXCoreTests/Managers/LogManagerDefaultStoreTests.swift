// Copyright © 2026 macMLX. English comments only.

import Foundation
import Pulse
import Testing

@testable import MacMLXCore

/// Code under test logs through `LogManager.shared`, so in a test runner a
/// manager built the default way must not open the user's
/// `~/.mac-mlx/logs/macmlx.pulse`. Checks which store the managers hold;
/// nothing is logged.
@Suite("LogManager default store under a test runner")
struct LogManagerDefaultStoreTests {

    @Test("The shared store is in memory, not the user's log")
    func theSharedStoreIsNotTheUsersLog() {
        #expect(TestHost.isActive)
        let store = LogManager.shared.store
        // A store that cannot be opened in memory is replaced by Pulse's own
        // default, which is on disk, so check the one the manager ended up with.
        #expect(store.options.contains(.inMemory))
        #expect(!store.storeURL.standardizedFileURL.path.hasPrefix(DataRoot.macMLX.standardizedFileURL.path))
    }

    @Test("Every manager built the default way holds that one store")
    func defaultManagersShareTheStore() {
        #expect(LogManager().store === LogManager.shared.store)
    }
}
