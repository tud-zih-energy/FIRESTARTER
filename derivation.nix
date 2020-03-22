{ stdenv
, cmake
, llvm
, zlib
, ncurses
, glibc
, hwloc
, git
, lib
, pkgconfig
, static ? false
}:

stdenv.mkDerivation {
  name = "firestarter";
  version = "0.0";
  src = ./.;

  nativeBuildInputs = [ cmake git pkgconfig ];
  buildInputs = [
    llvm
    llvm.lib
  ] ++ (if static then [
    glibc.static
    zlib.static
    (ncurses.override { enableStatic = true; })
    (let pkg = lib.overrideDerivation hwloc (oldAttrs: rec {
      configureFlags = oldAttrs.configureFlags ++ [
        "--enable-static"
      ];
    }); in [ pkg.dev pkg.lib ])
  ] else [
    ncurses
    zlib
    hwloc.dev
    hwloc.lib
  ]);

  cmakeFlags = [
    (stdenv.lib.optional static "-DCMAKE_EXE_LINKER_FLAGS=\"-static\"")
  ];
  enableParalellBuilding = true;

  installPhase = ''
    mkdir -p $out/bin
    cp src/firestarter $out/bin/
  '';

}
