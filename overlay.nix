self: super: {
  firestarter = self.callPackage ./derivation.nix { };
  firestarter-static = self.callPackage ./derivation.nix { static = true; };
}
