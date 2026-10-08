import Foundation

/// The two engines the server keeps beside the chat models (#130). Each has
/// its own slot on the server and its own key in the pool, so a reranker and
/// a chat model with the same id never share an entry: the library lists a
/// reranker checkpoint as one, but the server's load-by-path door can bring
/// the same checkpoint in as a chat model.
public enum AuxiliaryKind: String, Sendable {
    case embedder
    case reranker
}
