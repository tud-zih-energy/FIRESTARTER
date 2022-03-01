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
      url = https://download.open-mpi.org/release/hwloc/v2.7/hwloc-2.7.0.tar.gz;
      sha256 = "d9b23e9b0d17247e8b50254810427ca8a9857dc868e2e3a049f958d7c66af374";
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

  nativeBuildInputs = [ cmake git pkgconfig ];

  buildInputs = if withCuda then
    [ glibc_multi cudatoolkit linuxPackages.nvidia_x11 hwloc ]
    else
    [ glibc.static hwloc ];

  cmakeFlags = [
    "-DFIRESTARTER_BUILD_HWLOC=OFF"
    "-DCMAKE_C_COMPILER_WORKS=1"
    "-DCMAKE_CXX_COMPILER_WORKS=1"
  ] ++ optionals withCuda [
   "-DFIRESTARTER_BUILD_TYPE=FIRESTARTER_CUDA"
  ];

  enableParalellBuilding = true;

  installPhase = ''
    mkdir -p $out/bin
    cp src/FIRESTARTER${optionalString withCuda ''_CUDA''} $out/bin/
  '';

}
