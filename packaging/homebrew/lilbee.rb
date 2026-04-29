class Lilbee < Formula
  desc "Local search engine and personal encyclopedia for your notes, code, and PDFs"
  homepage "https://github.com/tobocop2/lilbee"
  version "0.6.66b456"
  license "Elastic-2.0"

  on_macos do
    on_arm do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-macos-arm64"
      sha256 "6150740741616c46142fb6de1848bca979fabcc649d6e3183dacb4d9102fd73e"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-linux-x86_64"
      sha256 "e6dc28e49a9bd9158eb217a47a278552f42195cae2cd68642413bf382ed0c54c"
    end
  end

  def install
    binary = Dir["lilbee-*"].first
    bin.install binary => "lilbee"

    # The binary is unsigned (Apple charges $99/yr for signing). Clear the
    # com.apple.quarantine xattr Homebrew applied during download so Gatekeeper
    # does not block first launch. Mirrors what the Obsidian plugin does at runtime.
    system "xattr", "-cr", bin/"lilbee" if OS.mac?
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/lilbee --version")
  end
end
