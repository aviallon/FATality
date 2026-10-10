{
  description = "FATality — scavenge everything recoverable from broken FAT filesystems";

  inputs = { nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in
    {
      devShells = forAllSystems (system:
        let pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = with pkgs; [
              uv
              python312
              # verification / cross-check tooling
              sleuthkit
              testdisk
              exiftool
              ffmpeg
              libjpeg # jpegtran/djpeg
              dosfstools # mkfs.fat for test fixtures
            ];
          };
        });
    };
}
