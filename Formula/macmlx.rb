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
    # The mlx-swift resource bundle (default.metallib) must sit next to
    # the executable — MLX resolves its Metal library relative to the
    # binary, and a bare install aborts on the first inference with
    # "Failed to load the default metallib". Keep both in libexec and
    # expose a bin shim.
    libexec.install "macmlx", "mlx-swift_Cmlx.bundle"
    bin.write_exec_script libexec/"macmlx"
  end

  test do
    assert_match(/^macmlx /, shell_output("#{bin}/macmlx --version"))
  end
end
