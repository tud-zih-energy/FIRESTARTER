{ nixpkgs ? <nixpkgs> }:

let
  pkgs = import nixpkgs {};
in
  pkgs.mkShell {
    buildInputs = with pkgs; [
      llvm.lib
      llvm
      glibc.static
      zlib.static
      (ncurses.override({ enableStatic = true; }))
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
