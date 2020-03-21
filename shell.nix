{ nixpkgs ? <nixpkgs> }:

let
  pkgs = import nixpkgs {};
in
  pkgs.mkShell {
    nativeBuildInputs = with pkgs; [ cmake git autoconf libtool automake ];
    buildInputs = with pkgs; [
      llvm.lib
      llvm
      glibc.static
      zlib.static
      (ncurses.override({ enableStatic = true; }))
      (lib.overrideDerivation hwloc (oldAttrs: rec {
        configureFlags = oldAttrs.configureFlags ++ [
          "--enable-static"
        ];
      })).lib
    ];

      #export CXX=${pkgs.clang}/bin/clang++
      #export CC=${pkgs.clang}/bin/clang
      #export LLVM_DIR="$(llvm-config --libdir --link-static)"
    shellHook = ''
      CC=${pkgs.clang}/bin/clang
      CXX=${pkgs.clang}/bin/clang++
      LLVM_DIR="$(llvm-config --libdir --link-static)"
    '';
  }
