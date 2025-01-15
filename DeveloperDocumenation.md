# Developer Documentation

This document contains knowledge required to work on the FIRESTARTER project.

## Coding Guidelines

Code style follows the LLVM project.
It is enforced by an automatic CI check.
The currently used version for `clang-format` and `clang-tidy` is `1.14` (the default for Ubuntu 22.04).

When using VSCode as your editor it is recomended that you install the [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format) and [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) extension.

## Building

The project can be build using CMake.
See `Building FIRESTARTER` Section in the Readme for more details.

## Running tests

Tests will be build when passing the variable `-DFIRESTARTER_BUILD_TESTS=ON` to CMake.

To run the standard set of tests run `ctest` in the build directory.

To run `clang-tidy` with your changes execute `./tooling/clang-tidy.py clang-tidy-report --build-root <YOUR_BUILD_DIRECTORY>`.
This will create a file called `clang-tidy-report.txt` in the build root containing the failed clang-tidy checks.