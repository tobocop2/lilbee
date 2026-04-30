{
  # Wraps the same prebuilt frozen binary that the Docker image and AUR
  # PKGBUILD consume (built upstream with PyInstaller / Nuitka onefile).
  # The binary self-extracts on launch and dlopen's its bundled .so's, so
  # on Linux we run it inside a buildFHSEnv that exposes glibc / libgomp /
  # vulkan-loader at the conventional /lib paths the extracted libs expect.
  # Darwin uses @executable_path install names so no FHS dance is needed.
  #
  # Notes:
  #   - Models (GGUF, embeddings) download at runtime to the HuggingFace cache.
  #     First run needs network; thereafter it is fully local.
  #   - Tesseract is not bundled (OCR is optional). To enable:
  #       nix shell nixpkgs#tesseract --command nix run github:tobocop2/lilbee
  #   - CUDA / ROCm are not handled here. The binary uses Vulkan on Linux
  #     and Metal on macOS for GPU acceleration.
  #   - The version + sha256 fields below are stamped by
  #     packaging/tools/render_flake.sh on each release.

  description = "Local search engine and personal encyclopedia for your notes, code, and PDFs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      pkgDescription = "Local search engine and personal encyclopedia for your notes, code, and PDFs";

      version = "0.6.66b456"; # RENDERED:VERSION

      shas = {
        x86_64-linux = "e6dc28e49a9bd9158eb217a47a278552f42195cae2cd68642413bf382ed0c54c"; # RENDERED:SHA_LINUX
        aarch64-darwin = "6150740741616c46142fb6de1848bca979fabcc649d6e3183dacb4d9102fd73e"; # RENDERED:SHA_DARWIN
      };

      assets = {
        x86_64-linux = "lilbee-linux-x86_64";
        aarch64-darwin = "lilbee-macos-arm64";
      };

      systems = builtins.attrNames shas;
      forAllSystems = nixpkgs.lib.genAttrs systems;

      mkLilbee = system:
        let
          # Elastic 2.0 is source-available with restrictions, which nixpkgs
          # classifies as unfree. Scope the allow-unfree predicate to just
          # lilbee so users get nix run / nix profile install working out of
          # the box without lowering their global nixpkgs config.
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfreePredicate = pkg:
              builtins.elem (nixpkgs.lib.getName pkg) [ "lilbee" "lilbee-bin" ];
          };
          inherit (pkgs) lib;
          isLinux = lib.hasSuffix "linux" system;

          baseMeta = {
            description = pkgDescription;
            homepage = "https://github.com/tobocop2/lilbee";
            license = lib.licenses.elastic20;
            mainProgram = "lilbee";
            platforms = systems;
          };

          lilbee-bin = pkgs.stdenvNoCC.mkDerivation {
            pname = "lilbee-bin";
            inherit version;
            src = pkgs.fetchurl {
              url = "https://github.com/tobocop2/lilbee/releases/download/v${version}/${assets.${system}}";
              sha256 = shas.${system};
            };
            dontUnpack = true;
            installPhase = ''
              runHook preInstall
              install -Dm755 $src $out/bin/lilbee
              runHook postInstall
            '';
            meta = baseMeta;
          };

          linuxFHS = pkgs.buildFHSEnv {
            name = "lilbee";
            targetPkgs = ps: with ps; [
              stdenv.cc.cc.lib
              glibc
              zlib
              vulkan-loader
              libGL
            ];
            runScript = "${lilbee-bin}/bin/lilbee";
            meta = baseMeta;
          };
        in
          if isLinux then linuxFHS else lilbee-bin;
    in {
      packages = forAllSystems (system: {
        default = mkLilbee system;
      });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = nixpkgs.lib.getExe self.packages.${system}.default;
          meta = self.packages.${system}.default.meta;
        };
      });
    };
}
