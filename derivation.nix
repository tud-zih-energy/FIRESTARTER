{ stdenv
, cmake
, llvm
, zlib
, ncurses
, glibc
, static ? false
}:

stdenv.mkDerivation {
  name = "firestarter";
  version = "0.0";
  src = ./.;

  nativeBuildInputs = [ cmake ];
  buildInputs = [
    llvm
    llvm.lib
  ] ++ (if static then [
    glibc.static
    zlib.static
    (ncurses.override { enableStatic = true; })
  ] else [
    ncurses
    zlib
  ]);

  cmakeFlags = [
#    (stdenv.lib.optional static "-DBUILD_STATIC=1")
    (stdenv.lib.optional static "-DCMAKE_EXE_LINKER_FLAGS=\"-static\"")
  ];
  enableParalellBuilding = true;

  installPhase = ''
    mkdir -p $out/bin
    cp src/firestarter $out/bin/
  '';

}
