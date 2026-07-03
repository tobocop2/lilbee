{
  description = "Run and manage local AI models and search your files, code, and crawled web pages, with cited answers";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { self, nixpkgs }:
    let
      sources = builtins.fromJSON (builtins.readFile ./sources.json);
      inherit (sources) version;

      systems = builtins.attrNames sources.systems;
      forAllSystems = nixpkgs.lib.genAttrs systems;

      mkPkgs = system: import nixpkgs { inherit system; };

      mkMeta = pkgs: {
        description = "Run and manage local AI models and search your files, code, and crawled web pages, with cited answers";
        homepage = "https://github.com/tobocop2/lilbee";
        license = pkgs.lib.licenses.mit;
        mainProgram = "lilbee";
        platforms = systems;
        sourceProvenance = [ pkgs.lib.sourceTypes.binaryNativeCode ];
      };

      mkBin =
        pkgs: system:
        let
          entry = sources.systems.${system};
        in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "lilbee-bin";
          inherit version;
          src = pkgs.fetchurl {
            url = "https://github.com/tobocop2/lilbee/releases/download/v${version}/${entry.asset}";
            inherit (entry) sha256;
          };
          dontUnpack = true;
          installPhase = ''
            runHook preInstall
            install -Dm755 $src $out/bin/lilbee
            runHook postInstall
          '';
          meta = mkMeta pkgs;
        };

      # The release binary is a Nuitka onefile: it self-extracts at launch
      # and dlopen's its bundled .so's via the standard ld.so path.
      # buildFHSEnv exposes glibc / libgomp / vulkan-loader at /lib so the
      # extracted libs resolve on bare NixOS. Darwin uses @executable_path
      # install names and needs no wrapper.
      mkLinuxFHS =
        pkgs: system:
        pkgs.buildFHSEnv {
          name = "lilbee";
          targetPkgs =
            ps: with ps; [
              stdenv.cc.cc.lib
              glibc
              zlib
              vulkan-loader
              libGL
            ];
          runScript = "${mkBin pkgs system}/bin/lilbee";
          meta = mkMeta pkgs;
        };

      mkLilbee =
        system:
        let
          pkgs = mkPkgs system;
          isLinux = pkgs.lib.hasSuffix "linux" system;
        in
        if isLinux then mkLinuxFHS pkgs system else mkBin pkgs system;

      # CUDA variant: lives in sources-cuda.json so the standard publish path
      # (which overwrites sources.json) cannot wipe it. Only present on
      # x86_64-linux and only when sources-cuda.systems.${system} is populated;
      # publish-cuda-packages fills it in after build-cuda-executables runs.
      cudaSources = builtins.fromJSON (builtins.readFile ./sources-cuda.json);
      cudaSystems = builtins.attrNames cudaSources.systems;
      hasCuda = system: builtins.elem system cudaSystems;

      mkCudaBin =
        pkgs: system:
        let
          entry = cudaSources.systems.${system};
        in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "lilbee-cuda";
          inherit version;
          src = pkgs.fetchurl {
            url = "https://github.com/tobocop2/lilbee/releases/download/v${version}/${entry.asset}";
            inherit (entry) sha256;
          };
          dontUnpack = true;
          installPhase = ''
            runHook preInstall
            install -Dm755 $src $out/bin/lilbee
            runHook postInstall
          '';
          meta = mkMeta pkgs;
        };

      # Like mkLinuxFHS but omits vulkan-loader. The CUDA build bundles its
      # own cudart/cublas via Nuitka onefile; the host only needs the NVIDIA
      # driver, which on NixOS is exposed via hardware.nvidia.* config.
      mkCudaLinuxFHS =
        pkgs: system:
        pkgs.buildFHSEnv {
          name = "lilbee-cuda";
          targetPkgs =
            ps: with ps; [
              stdenv.cc.cc.lib
              glibc
              zlib
              libGL
            ];
          runScript = "${mkCudaBin pkgs system}/bin/lilbee";
          meta = mkMeta pkgs;
        };

      mkLilbeeCuda =
        system:
        let
          pkgs = mkPkgs system;
        in
        mkCudaLinuxFHS pkgs system;

      # Pre-Haswell CPU variant: the same Vulkan binary built against the +compat
      # lancedb. Kept in sources-compat.json so the standard publish (which
      # overwrites sources.json) can't wipe it; present only on x86_64-linux and
      # only once publish-compat-packages fills sources-compat.systems.${system}.
      compatSources = builtins.fromJSON (builtins.readFile ./sources-compat.json);
      compatSystems = builtins.attrNames compatSources.systems;
      hasCompat = system: builtins.elem system compatSystems;

      mkCompatBin =
        pkgs: system:
        let
          entry = compatSources.systems.${system};
        in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "lilbee-compat";
          inherit version;
          src = pkgs.fetchurl {
            url = "https://github.com/tobocop2/lilbee/releases/download/v${version}/${entry.asset}";
            inherit (entry) sha256;
          };
          dontUnpack = true;
          installPhase = ''
            runHook preInstall
            install -Dm755 $src $out/bin/lilbee
            runHook postInstall
          '';
          meta = mkMeta pkgs;
        };

      # Same FHS surface as the default build -- the compat binary is still a
      # Vulkan onefile, only its bundled lancedb differs.
      mkCompatLinuxFHS =
        pkgs: system:
        pkgs.buildFHSEnv {
          name = "lilbee-compat";
          targetPkgs =
            ps: with ps; [
              stdenv.cc.cc.lib
              glibc
              zlib
              vulkan-loader
              libGL
            ];
          runScript = "${mkCompatBin pkgs system}/bin/lilbee";
          meta = mkMeta pkgs;
        };

      mkLilbeeCompat =
        system:
        let
          pkgs = mkPkgs system;
        in
        mkCompatLinuxFHS pkgs system;
    in
    {
      packages = forAllSystems (
        system:
        {
          default = mkLilbee system;
        }
        // nixpkgs.lib.optionalAttrs (hasCuda system) {
          lilbee-cuda = mkLilbeeCuda system;
        }
        // nixpkgs.lib.optionalAttrs (hasCompat system) {
          lilbee-compat = mkLilbeeCompat system;
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = nixpkgs.lib.getExe self.packages.${system}.default;
          inherit (self.packages.${system}.default) meta;
        };
      });

      formatter = forAllSystems (system: (mkPkgs system).nixfmt-rfc-style);

      nixosModules.lilbee =
        {
          config,
          lib,
          pkgs,
          ...
        }:
        let
          cfg = config.services.lilbee;
        in
        {
          options.services.lilbee = {
            enable = lib.mkEnableOption "lilbee HTTP server as a user-level systemd service";
            package = lib.mkOption {
              type = lib.types.package;
              default = self.packages.${pkgs.system}.default;
              defaultText = lib.literalExpression "lilbee.packages.\${pkgs.system}.default";
              description = "lilbee package to run.";
            };
          };
          config = lib.mkIf cfg.enable {
            systemd.user.services.lilbee = {
              description = "lilbee HTTP server";
              after = [ "network-online.target" ];
              wants = [ "network-online.target" ];
              wantedBy = [ "default.target" ];
              serviceConfig = {
                Type = "simple";
                ExecStart = "${cfg.package}/bin/lilbee serve --host 127.0.0.1 --port 42697";
                Restart = "on-failure";
                RestartSec = 5;
              };
            };
          };
        };
    };
}
