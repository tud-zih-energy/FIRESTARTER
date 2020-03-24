[![Build Status](https://travis-ci.org/tud-zih-energy/FIRESTARTER.svg?branch=master)](https://travis-ci.org/tud-zih-energy/FIRESTARTER)
[![Build status](https://ci.appveyor.com/api/projects/status/oon43fcq6ulee503/branch/master?svg=true)](https://ci.appveyor.com/project/bmario/firestarter/branch/master)


# FIRESTARTER - A Processor Stress Test Utility

The python script code-generator.py generates source code of FIRESTARTER. It
generates load functions for the architectures defined in the config.cfg file.

Usage:
Call ./code-generator.py from to generate the source code of FIRESTARTER.
This will generate the required files to build FIRESTARTER in the directory
from which it is called.

optional arguments:
> -h | --help            print usage information

> -v | --verbose         enable debug output

> -c | --enable-cuda     enable CUDA support

> -m | --enable-mac      enable Mac O/S support

> -w | --enable-win      enable windows support

If one of the --enable-* arguments is used it overrides all the feature
selections in the config file, i.e., if one feature is added on the command
line, features that are enabled by default have to be added to the command as
well.

# Reference

A detailed description can be found in the following paper. Please cite this if you use FIRESTARTER for scientific work.

Daniel Hackenberg, Roland Oldenburg, Daniel Molka, and Robert Schöne
[Introducing FIRESTARTER: A processor stress test utility](http://dx.doi.org/10.1109/IGCC.2013.6604507) (IGCC 2013)

Additional information: https://tu-dresden.de/zih/forschung/projekte/firestarter


# Contact

Daniel Hackenberg < daniel dot hackenberg at tu-dresden.de >
