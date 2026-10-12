# Homebrew formula for Assembly.
#
# This is a template: the `version` and each `sha256` must be set to a real,
# published release before the formula is usable. The release workflow uploads
# `assembly-<platform>.sha256` checksum files alongside the binaries; copy those
# values in here (or into a tap) when cutting a release.
class Assembly < Formula
  desc "Multi-model LLM router with a single-binary distribution"
  homepage "https://github.com/assembly-engine/assembly"
  version "0.1.0"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/assembly-engine/assembly/releases/download/v#{version}/assembly-macos-arm64"
      sha256 "REPLACE_WITH_MACOS_ARM64_SHA256"
    end

    on_intel do
      url "https://github.com/assembly-engine/assembly/releases/download/v#{version}/assembly-macos-x86_64"
      sha256 "REPLACE_WITH_MACOS_X86_64_SHA256"
    end
  end

  on_linux do
    url "https://github.com/assembly-engine/assembly/releases/download/v#{version}/assembly-linux-x86_64"
    sha256 "REPLACE_WITH_LINUX_SHA256"
  end

  def install
    bin.install Dir["assembly-*"].first => "assembly"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/assembly --version")
    assert_match "build", shell_output("#{bin}/assembly --help")
  end
end
