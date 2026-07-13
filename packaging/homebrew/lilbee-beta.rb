class LilbeeBeta < Formula
  desc "Whole local AI stack in one executable: models, search, crawler (beta channel)"
  homepage "https://github.com/tobocop2/lilbee"
  version "0.0.0"
  license "MIT"

  conflicts_with "tobocop2/lilbee/lilbee", because: "both install the lilbee binary"
  conflicts_with "tobocop2/lilbee/lilbee-cuda", because: "both install the lilbee binary"
  conflicts_with "tobocop2/lilbee/lilbee-compat", because: "both install the lilbee binary"

  on_macos do
    on_arm do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-macos-arm64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-linux-x86_64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  def install
    binary = Dir["lilbee-*"].first
    bin.install binary => "lilbee"
  end

  service do
    run [opt_bin/"lilbee", "serve", "--host", "127.0.0.1", "--port", "42697"]
    keep_alive true
    log_path var/"log/lilbee/serve.log"
    error_log_path var/"log/lilbee/serve.err.log"
  end

  def caveats
    <<~EOS
      lilbee-beta tracks the beta channel: the same binaries as `lilbee`, promoted
      ahead of the stable release. It installs the same `lilbee` command, so if you
      previously had `lilbee` installed, uninstall it first:

        brew uninstall lilbee
        brew install tobocop2/lilbee/lilbee-beta
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/lilbee --version")
  end
end
