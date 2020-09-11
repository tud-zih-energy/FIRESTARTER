{ stdenv
, cmake
, glibc_multi
, glibc
, git
, pkgconfig
, cudatoolkit
, withCuda ? false
, linuxPackages
}:

let
  hwloc = stdenv.mkDerivation rec {
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

in
with stdenv.lib;
stdenv.mkDerivation rec {
  name = "firestarter";
  version = "0.0";
  src = ./.;

  nativeBuildInputs = [ cmake git ];

  buildInputs = if withCuda then
    [ glibc_multi cudatoolkit linuxPackages.nvidia_x11 ]
    else
    [ glibc.static ];

  cmakeFlags = [
    "-DCMAKE_CXX_FLAGS=\"-DAFFINITY\""
    "-DHWLOC_LIB_DIR=${hwloc.lib}"
    "-DHWLOC_INCLUDE_DIR=${hwloc.dev}"
    "-DNIX_BUILD=1"
    "-DCMAKE_C_COMPILER_WORKS=1"
    "-DCMAKE_CXX_COMPILER_WORKS=1"
  ] ++ optionals withCuda [
    "-DBUILD_CUDA=1"
  ];

  enableParalellBuilding = true;

  installPhase = ''
    mkdir -p $out/bin
    cp src/FIRESTARTER${optionalString withCuda ''_CUDA''} $out/bin/
  '';

}
