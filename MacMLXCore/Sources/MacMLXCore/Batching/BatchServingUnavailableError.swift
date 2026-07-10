// Copyright © 2026 macMLX. English comments only.

import Foundation

/// Terminal error handed to any batched row that can no longer be served because
/// the resident model went away between admission and drive-loop start (H3).
///
/// Admission is a binding promise (``BatchGenerationServing`` clause 3): once a row
/// is enqueued the seam has already returned a stream, so there is no fall-back to
/// the legacy path left. If the drive loop then finds the container gone (an unload
/// or a failed swap raced the admission), the only correct outcome is to finish
/// every queued row's stream with a clear error rather than leave it dangling — a
/// dangling row would wedge the coordinator's `driving`/drain state and deadlock the
/// next ``BatchServingCoordinator/beginDrain()`` under the server's FIFO lock.
struct BatchServingUnavailableError: Error, Equatable, CustomStringConvertible {
    var description: String {
        "batched decode aborted: the resident model was unloaded or swapped before "
            + "the cohort could start — retry the request"
    }
}
