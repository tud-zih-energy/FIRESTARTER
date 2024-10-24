# In this folder, there are examples on how one could include tracing with an external library

Since these will be mostly shared libraries, we probably will only be able to use them with a dynamic build of FIRESTARTER

You should be able to add the external libraries also to a static build by using the flags `FIRESTARTER_TRACING_*` for `cmake`. There you probably would include an archive. But this was not tested yet.

## Example 1: Score-P

- Needs Score-P: https://www.vi-hps.org/projects/score-p
- File: `scorep.c`
- Compilation:
  - 1. create the adapter configuration: `scorep-config --adapter-init --dynamic --user --nokokkos --nocompiler --thread=pthread > .scorep_init.c`
  - 2. compile the adapter and the tracing library: `scorep --user --nocompiler --dynamic --nokokkos --noopenmp --thread=pthread gcc -fPIC -c -DSCOREP_USER_ENABLE scorep.c .scorep_init.c`
  - 3. link the tracing library: `scorep --no-as-needed --dynamic --user --nokokkos --nocompiler --noopenmp --thread=pthread gcc -shared -o libfirestarter_scorep.so scorep.o .scorep_init.o`
  - 4. cmake FIRESTARTER: `cmake -DFIRESTARTER_TRACING=External -DFIRESTARTER_TRACING_LD_FLAGS="-L/home/rschoene/git/FIRESTARTER/examples/tracing -lfirestarter_scorep" -DFIRESTARTER_LINK_STATIC=OFF ..`
  - 5. make FIRESTARTER: `make -j`

- Running:
  - Make sure that the library `libfirestarter_scorep.so` can be found in the `LD_LIBRARY_PATH`
  - Run `FIRESTARTER` as usual
  - Running `FIRESTARTER` should create a profile in your current directory starting with `scorep...`
  - You can change environment variables to tune Score-P for your purposes. Have a look at: https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/latest/pdf/scorep.pdf
