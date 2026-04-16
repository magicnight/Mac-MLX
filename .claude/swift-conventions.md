# Swift Conventions

## Language & SDK Requirements

- Swift 5.9+
- macOS 14.0+ deployment target
- Apple Silicon only (no x86_64 slices needed)
- Xcode 15+

## Concurrency

Always use Swift structured concurrency. No callbacks, no Combine for new code.

```swift
// ✅ Correct
func loadModels() async throws -> [LocalModel] {
    let urls = try await FileManager.default.contentsOfDirectory(...)
    return try await withThrowingTaskGroup(of: LocalModel?.self) { group in
        // ...
    }
}

// ❌ Wrong
func loadModels(completion: @escaping ([LocalModel]) -> Void) { }
```

Use `Actor` for shared mutable state:

```swift
actor DownloadQueue {
    private var activeDownloads: [String: DownloadTask] = [:]
    
    func add(_ task: DownloadTask) { ... }
    func cancel(id: String) { ... }
}
```

## State Management

Use `@Observable` macro exclusively. Never use `ObservableObject`.

```swift
// ✅ Correct
@Observable
final class ModelLibraryManager {
    var models: [LocalModel] = []
    var isScanning: Bool = false
}

// ❌ Wrong
class ModelLibraryManager: ObservableObject {
    @Published var models: [LocalModel] = []
}
```

## Error Handling

Define typed errors per domain:

```swift
enum InferenceServiceError: LocalizedError {
    case pythonNotFound
    case portAlreadyInUse(Int)
    case backendCrashed(exitCode: Int32)
    case modelNotLoaded(String)
    
    var errorDescription: String? {
        switch self {
        case .pythonNotFound:
            return "Python 3.10+ not found. Please install via Homebrew."
        // ...
        }
    }
}
```

Never use:
- `try!` in production code
- `!` force unwrap (use `guard let` or `if let`)
- `fatalError` except for programmer errors in debug builds

## Naming Conventions

- Types: `UpperCamelCase`
- Functions/variables: `lowerCamelCase`
- Constants: `lowerCamelCase` (Swift style, not `kConstant`)
- Files: match the primary type name exactly

```swift
// ✅
let maxConcurrentDownloads = 3
struct LocalModel { }
class InferenceServiceManager { }

// ❌
let MAX_CONCURRENT_DOWNLOADS = 3
let kMaxDownloads = 3
```

## SwiftUI Patterns

Keep views thin. Extract logic to `@Observable` managers.

```swift
// ✅ Thin view
struct ModelRowView: View {
    let model: LocalModel
    @Environment(ModelLibraryManager.self) var library
    
    var body: some View {
        HStack {
            Text(model.name)
            Spacer()
            Button("Load") { Task { await library.load(model) } }
        }
    }
}
```

Use `.task {}` for async work triggered by view lifecycle:

```swift
.task {
    await viewModel.fetchData()
}
```

## Process Management

Use `Foundation.Process` for Python subprocess:

```swift
final class PythonProcess {
    private var process: Process?
    
    func start(scriptPath: URL, port: Int) async throws {
        let process = Process()
        process.executableURL = try await findPython()
        process.arguments = [scriptPath.path, "--port", "\(port)"]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        self.process = process
    }
    
    func stop() {
        process?.terminate()
        process = nil
    }
}
```

## HTTP Client

Use `URLSession` with async/await. No third-party HTTP libraries.

```swift
func chatCompletion(_ request: ChatRequest) async throws -> ChatResponse {
    var urlRequest = URLRequest(url: baseURL.appending(path: "/v1/chat/completions"))
    urlRequest.httpMethod = "POST"
    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
    urlRequest.httpBody = try JSONEncoder().encode(request)
    
    let (data, response) = try await URLSession.shared.data(for: urlRequest)
    guard (response as? HTTPURLResponse)?.statusCode == 200 else {
        throw InferenceAPIError.unexpectedStatus
    }
    return try JSONDecoder().decode(ChatResponse.self, from: data)
}
```

For streaming (SSE), use `URLSession.bytes(for:)`.

## File Organization

One type per file. File name matches type name.
Group by feature, not by type:

```
Views/
  Chat/
    ChatView.swift
    ChatMessageView.swift
    ChatInputView.swift
  ModelLibrary/
    ModelLibraryView.swift
    ModelRowView.swift
```

## Dependencies

Allowed Swift packages (via SPM only):
- No third-party HTTP libraries (use URLSession)
- No third-party JSON libraries (use Codable)
- `swift-argument-parser` if CLI target is added

When adding any new dependency, add a comment explaining why
the system framework is insufficient.
