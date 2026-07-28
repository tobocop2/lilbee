class LilbeeRocm < Formula
  desc "Whole local AI stack in one executable: models, search, crawler (AMD ROCm)"
  homepage "https://github.com/tobocop2/lilbee"
  version "0.0.0"
  license "MIT"

  conflicts_with "tobocop2/lilbee/lilbee", because: "both install the lilbee binary"
  conflicts_with "tobocop2/lilbee/lilbee-cuda", because: "both install the lilbee binary"
  conflicts_with "tobocop2/lilbee/lilbee-compat", because: "both install the lilbee binary"

  on_linux do
    on_intel do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-linux-x86_64-rocm"
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
      lilbee-rocm installs the same `lilbee` binary as the regular `lilbee` formula,
      built to offload inference to an AMD GPU through ROCm. The ROCm userspace ships
      inside it, so the host needs only the amdgpu kernel driver. If you previously had
      `lilbee` installed, uninstall it first:

        brew uninstall lilbee
        brew install tobocop2/lilbee/lilbee-rocm
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/lilbee --version")
  end
end
