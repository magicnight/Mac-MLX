# Formula/macmlx.rb — Homebrew formula template for the `macmlx` CLI.
# Source of truth lives in the macMLX repo; scripts/render-formula.sh
# substitutes version + URL + sha256 on each release and the rendered
# file is published to magicnight/homebrew-mac-mlx. See
# .claude/distribution.md (Homebrew Tap section) for the full pipeline.
class Macmlx < Formula
  desc "Native macOS LLM inference CLI for Apple Silicon (powered by MLX)"
  homepage "https://github.com/magicnight/mac-mlx"
  url "@@URL@@"
  version "@@VERSION@@"
  sha256 "@@SHA256@@"
  license "Apache-2.0"

  # macmlx links against the dynamic Swift stdlib that ships with
  # macOS 14+ on Apple Silicon. Hard-fail on anything older or non-arm64
  # rather than producing a runtime crash.
  depends_on macos: :sonoma
  depends_on arch: :arm64

  def install
    bin.install "macmlx"
  end

  test do
    assert_match(/^macmlx /, shell_output("#{bin}/macmlx --version"))
  end
end
