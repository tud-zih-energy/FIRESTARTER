self: super: {
  firestarter = self.callPackage ./derivation.nix { };
  firestarter-cuda = self.callPackage ./derivation.nix { withCuda = true; };
}
