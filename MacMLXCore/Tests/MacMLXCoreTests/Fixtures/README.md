# Structured-output fixtures: provenance

The `.safetensors` and chat-template fixtures in this directory are documented
by the tests that load them. This note covers the JSON-schema fixtures used by
the structured-output tests (`Constraint/` and `Server/StructuredOutputServerTests`).

## Captured from Apple's framework

- `fm_generable_schemas_fixture.json` — the JSON schemas Apple's Foundation
  Models framework (macOS 27 SDK) emits for three `@Generable` test types,
  captured from the framework itself: `Person` (a nested type as `$defs` +
  `$ref`, bounded arrays, an array of arrays, string enums, a `const`, and an
  integer with `minimum`/`maximum` from a `.range` guide, which stays
  unsupported), `PersonNoRange` (the same without the `.range` property) and
  `Explicit` (only optional properties — a string, a `$ref` object and an
  integer array — none of them required).

## Copied from ml-explore/mlx-swift-lm (MIT)

- `fm_itinerary_production_fixture.json` — the TripPlanner `Itinerary` schema
  as Apple's framework emits it: `TestFixtures.itinerarySchemaProduction` in
  `Tests/MLXFoundationModelsTests/TestHelpers.swift`, copied verbatim.
- `schema_tier1_steps.json` … `schema_tier4_steps.json` — upstream's
  constrained-decoding goldens, copied byte for byte from
  `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Fixtures/goldens/`.
  The tests here use only each file's `schema` and `document`; the per-step
  masks belong to upstream's tokenizer and are kept so the files stay
  identical to their source.

Source: <https://github.com/ml-explore/mlx-swift-lm>, tag 3.32.3
(commit `3b339ad6e3b3f44c8121ecff5131c7fd55e075e6`). Licence:

```
MIT License

Copyright (c) 2024 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
