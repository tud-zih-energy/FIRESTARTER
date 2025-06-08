# ScoreP Tracing example

This folder show how you can include ScoreP for tracing FIRESTARTER.
This requires an installation of [ScoreP](https://www.vi-hps.org/projects/score-p).

Every implementation of tracing needs to implement the three functions `firestarterTracingInitialize`, `firestarterTracingRegionBegin` and `firestarterTracingRegionEnd`.
For an example look at `ScoreP.c`.

Required variables that need to be set to include shared or static tracing libraries into the FIRESTARTER build:
| CMake Variable | Description |
| --- | --- |
| `FIRESTARTER_TRACING` | Set to `External` to enable the tracing interface of FIRESTARTER. |
| `FIRESTARTER_LINK_STATIC` | Set to either `ON` or `OFF` depending if you want to add a static or shared tracing library. |
| `CMAKE_CXX_STANDARD_LIBRARIES` | Add the path to the libraries you want to include. Linker options can be set here, e.g. `-l` and `-L`. |

## Manual build

1. Create the adapter configuration: `scorep-config --adapter-init --user --nokokkos --nocompiler --thread=pthread > .scorep_init.c`
2. Compile the adapter and the tracing library: `scorep --user --nokokkos --nocompiler --thread=pthread gcc -fPIC -c -DSCOREP_USER_ENABLE scorep.c .scorep_init.c`
3. Link the tracing library: `scorep --user --nokokkos --nocompiler --thread=pthread gcc -shared -o libfirestarter_scorep.so scorep.o .scorep_init.o`
4. Configure FIRESTARTER: `cmake -DFIRESTARTER_TRACING=External -DCMAKE_CXX_STANDARD_LIBRARIES="-L<Folder of libfirestarter_scorep.so> -lfirestarter_scorep" -DFIRESTARTER_LINK_STATIC=OFF <Path to the project root>`
5. Build FIRESTARTER: `make -j`

## Automatic build

This folder contains a CMake file that builds FIRESTARTER with ScoreP instrumentation.
Running the normal CMake commands will create an instrumented FIRESTARTER build. `./FIRESTARTER/src/FIRESTARTER-build/src/FIRESTARTER`.

## Running FIRESTARTER with ScoreP instrumentation

- Make sure that the ScoreP library (`libfirestarter_scorep.so`) can be found in the `LD_LIBRARY_PATH`.
- Run `FIRESTARTER` as usual
- Running `FIRESTARTER` will create a profile in your current directory starting with `scorep-*`
- To tune ScoreP for your purposes, change enviroment variables according to the documentation: https://perftools.pages.jscfz-juelich.de/cicd/scorep/tags/latest/pdf/scorep.pdf
