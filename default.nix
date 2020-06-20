{ nixpkgs ? <nixpkgs> }:

let
  pkgs = import nixpkgs { overlays = [ (import ./overlay.nix) ]; };
in with pkgs; [
  (firestarter.override { "stdenv" = pkgs.clangStdenv; })
]
