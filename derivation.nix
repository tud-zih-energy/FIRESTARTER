{ stdenv
, cmake
, ncurses
, glibc
, git
, pkgconfig
}:

let
  llvm = stdenv.mkDerivation {
    name = "clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04";

    src = fetchTarball {
      url = https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz;
      sha256 = "1c51d1h46js96rla5g5dmg6ji7whpaggpj0frwmyr8597p5hgr78";
    };

    installPhase = ''
      mkdir -p $out
      cp -ar {lib,include} $out
    '';
  };

  hwloc = stdenv.mkDerivation {
    name = "hwloc";

    src = fetchTarball {
      url = https://download.open-mpi.org/release/hwloc/v2.2/hwloc-2.2.0.tar.gz;
      sha256 = "1ibw14h9ppg8z3mmkwys8vp699n85kymdz20smjd2iq9b67y80b6";
    };

    configureFlags = [
      "--enable-static"
      "--disable-libudev"
      "--disable-shared"
      "--disable-doxygen"
      "--disable-libxml2"
      "--disable-cairo"
      "--disable-io"
      "--disable-pci"
      "--disable-opencl"
      "--disable-cuda"
      "--disable-nvml"
      "--disable-gl"
      "--disable-libudev"
      "--disable-plugin-dlopen"
      "--disable-plugin-ltdl"
    ];

    nativeBuildInputs = [ pkgconfig ];

    enableParalellBuilding = true;

    outputs = [ "out" "lib" "dev" "doc" "man" ];
  };

in stdenv.mkDerivation {
  name = "firestarter";
  version = "0.0";
  src = ./.;

  nativeBuildInputs = [ cmake git ];

  buildInputs = [
    glibc.static
    (ncurses.override { enableStatic = true; })
  ];

  cmakeFlags = [
    "-DCMAKE_CXX_FLAGS=\"-DAFFINITY\""
    "-DLLVM_SOURCE_DIR=${llvm}"
    "-DHWLOC_LIB_DIR=${hwloc.lib}"
    "-DHWLOC_INCLUDE_DIR=${hwloc.dev}"
    "-DNIX_BUILD=1"
    "-DCMAKE_C_COMPILER_WORKS=1"
    "-DCMAKE_CXX_COMPILER_WORKS=1"
  ];

  enableParalellBuilding = true;

  installPhase = ''
    mkdir -p $out/bin
    cp src/FIRESTARTER $out/bin/
  '';

}
