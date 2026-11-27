class Lilbee < Formula
  desc "Whole local AI stack in one binary: models, cited search, crawler"
  homepage "https://github.com/tobocop2/lilbee"
  version "0.0.0"
  license "MIT"

  # The main Mac builds need macOS 15. Below that, each arch takes its compat
  # build. The renderer drops a compat block when the release has no such asset.
  on_macos do
    on_arm do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-macos-arm64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"

      on_sonoma :or_older do
        url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-compat-macos-arm64"
        sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      end
    end

    on_intel do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-macos-x86_64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"

      on_sonoma :or_older do
        url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-compat-macos-x86_64"
        sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      end
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-linux-x86_64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  # The first macOS release each Mac asset loads on, from .github/release-cells.yml
  # and .github/oldmac-cells.yml. The macOS blocks above match whole releases, so
  # the install checks the floor of the asset it got.
  MACOS_FLOORS = {
    "lilbee-macos-arm64"         => "15.0",
    "lilbee-macos-x86_64"        => "15.0",
    "lilbee-compat-macos-arm64"  => "12.3",
    "lilbee-compat-macos-x86_64" => "12.0",
  }.freeze

  def install
    binary = Dir["lilbee-*"].first
    floor = MACOS_FLOORS[binary]
    if OS.mac? && floor && MacOS.full_version < floor
      odie "lilbee needs macOS #{floor} or newer. This Mac runs #{MacOS.full_version}."
    end

    bin.install binary => "lilbee"

    # The binary is unsigned (Apple charges $99/yr for signing). Clear the
    # com.apple.quarantine xattr Homebrew applied during download so Gatekeeper
    # does not block first launch. Mirrors what the Obsidian plugin does at runtime.
    system "xattr", "-cr", bin/"lilbee" if OS.mac?
  end

  service do
    run [opt_bin/"lilbee", "serve", "--host", "127.0.0.1", "--port", "42697"]
    keep_alive true
    log_path var/"log/lilbee/serve.log"
    error_log_path var/"log/lilbee/serve.err.log"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/lilbee --version")
  end
end
