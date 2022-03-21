{ stdenv
, lib
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
      url = https://download.open-mpi.org/release/hwloc/v2.7/hwloc-2.7.0.tar.gz;
      sha256 = "0gii1b8m5c5x6zan66m5hbbzqhmxn5sfkl879xxrf3gb2gxf9mi9";
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
stdenv.mkDerivation rec {
  name = "firestarter";
  version = "0.0";
  src = ./.;

  nativeBuildInputs = [ cmake git pkgconfig ];

  buildInputs = if withCuda then
    [ glibc_multi cudatoolkit linuxPackages.nvidia_x11 hwloc ]
    else
    [ glibc.static hwloc ];

  cmakeFlags = [
    "-DFIRESTARTER_BUILD_HWLOC=OFF"
    "-DCMAKE_C_COMPILER_WORKS=1"
    "-DCMAKE_CXX_COMPILER_WORKS=1"
  ] ++ lib.optionals withCuda [
   "-DFIRESTARTER_BUILD_TYPE=FIRESTARTER_CUDA"
  ];

  enableParalellBuilding = true;

  installPhase = ''
    mkdir -p $out/bin
    cp src/FIRESTARTER${lib.optionalString withCuda ''_CUDA''} $out/bin/
  '';

}
