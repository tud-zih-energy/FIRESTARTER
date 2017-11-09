#!/usr/bin/env python
###############################################################################
# FIRESTARTER - A Processor Stress Test Utility
# Copyright (C) 2017 TU Dresden, Center for Information Services and High
# Performance Computing
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Contact: daniel.hackenberg@tu-dresden.de
###############################################################################

import os, sys, getopt, datetime, importlib
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

# import templates
from templates import firestarter_global_h, Makefile, work_c, work_h, main_c, main_win64_c

def usage():
    print("code-generator.py generates source code of FIRESTARTER")
    print("optional arguments:")
    print("-h|--help            print usage information")
    print("-v|--verbose         enable debug output")
    print("-c|--enable-cuda     enable CUDA support")
    print("-m|--enable-mac      enable Mac O/S support")
    print("-w|--enable-win      enable windows support")
    print("If one of the --enable-* arguments is used it overrides all the feature")
    print("selections in the config file, i.e., if one feature is added on the command line,")
    print("features that are enabled by default have to be added to the command as well.")

try:
    opts,args = getopt.getopt(sys.argv[1:], "hvcmw", ["help","verbose","enable-cuda","enable-mac","enable-win"])
except getopt.GetoptError as err:
    print(str(err)) # will print something like "option -a not recognized"
    usage()
    sys.exit(2)

class architecture:
    def __init__(self, name):
        self.name = name

class template:
    def __init__(self, name):
        self.name = name

class version:
    def __init__(self, name):
        self.name = name

architectures = []
templates = []
families = []
verbose = False
feature_override = False
features = [False,False,False]
date = datetime.datetime.now()

dirname = os.path.dirname(os.path.realpath(__file__))+'/'
outdir = os.getcwd()+'/'

for o, a in opts:
    if o in ("-v", "--verbose"):
        verbose = True
    elif o in ("-h", "--help"):
        usage()
        sys.exit()
    elif o in ("-c", "--enable-cuda"):
        feature_override = True
        features[0] = True
    elif o in ("-w", "--enable-win"):
        feature_override = True
        features[1] = True
    elif o in ("-m", "--enable-mac"):
        feature_override = True
        features[2] = True
    else:
        assert False, "unhandled option"

# read+parse config file
cfg = ConfigParser()
cfg.readfp(open(dirname+'config.cfg'))

for each in cfg.sections():
    if each == "VERSION":
        version = version(each)
        version.major = cfg.get(version.name,'major').strip()
        version.minor = cfg.get(version.name,'minor').strip()
        version.info = cfg.get(version.name,'info').strip()
        version.date = date.strftime("%Y-%m-%d")
        version.year = date.strftime("%Y")
        version.targets = 'linux'
        if feature_override == False:
            version.enable_cuda = int(cfg.get(version.name,'enable_cuda').strip())
            version.enable_win64 = int(cfg.get(version.name,'enable_win64').strip())
            version.enable_mac = int(cfg.get(version.name,'enable_mac').strip())
        else:
            version.enable_cuda = 0
            version.enable_win64 = 0
            version.enable_mac = 0
            if features[0] == True:
                version.enable_cuda = 1
            if features[1] == True:
                version.enable_win64 = 1
            if features[2] == True:
                version.enable_mac = 1
        if version.enable_cuda:
            version.targets = version.targets+' cuda'
        if version.enable_win64:
            version.targets = version.targets+' win64'
        # enabling mac support does not result in additional make target
    elif cfg.has_option(each,'template'):
        templates.append( template(each) )
    else:
        architectures.append( architecture(each) )

# read specifications of supported architectures from config file
for each in architectures:
    each.arch=cfg.get(each.name,'arch')
    each.model=cfg.get(each.name,'model')
    each.threads=[x.strip() for x in cfg.get(each.name,'threads').split(',')]
    each.isa=[x.strip() for x in cfg.get(each.name,'isa').split(',')]
    each.cpu_family=cfg.get(each.name,'cpu_family')
    if each.cpu_family not in families:
        families.append(each.cpu_family)
    each.cpu_model=[x.strip() for x in cfg.get(each.name,'cpu_model').split(',')]

    # parameters for code sequences
    sizes=[x.strip() for x in cfg.get(each.name,'buffer_sizes').split(',')]
    if len(sizes) == 4:
        each.l1_size=int(sizes[0])
        each.l2_size=int(sizes[1])
        each.l3_size=int(sizes[2])
        each.ram_size=int(sizes[3])
    else:
        print("Error: invalid setting for \"buffer_sizes\" in architecture "+each.name)
        sys.exit(2)
    each.instr_groups=[x.strip() for x in cfg.get(each.name,'instr_groups').split(',')]
    each.proportion=[x.strip() for x in cfg.get(each.name,'proportion').split(',')]
    if len(each.instr_groups) != len(each.proportion):
        print("Error: length of \"instr_group\" and \"proportion\" parameters not identical in architecture "+each.name)
        sys.exit(2)
    each.lines=int(cfg.get(each.name,'lines'))

    # settings that are currently identical for all supported architectures
    # - coverage:     defines percentage of capacity utilization of buffers for each level (L1/L2/L3/RAM)
    # - cl_size:      cache line width in byte
    each.l1_cover=0.5
    each.l2_cover=0.8
    each.l3_cover=0.8
    each.ram_cover=1.0
    each.cl_size=64

if verbose == True:
    print("source directory: "+dirname)
    print("output directory: "+outdir)
    print("version: "+str(version.major)+"."+str(version.minor))
    print("\nfeatures selection:")
    print("CUDA support:"+str(version.enable_cuda))
    print("Win64 support:"+str(version.enable_win64))
    print("Mac O/S support:"+str(version.enable_mac))
    print("\ngenerating code:")

# list of files from generator root directory that are copied to build directory
files = ['LICENSE','COPYING','CHANGELOG']

for each in templates:
    each.file=cfg.get(each.name,'template')+'_functions'
    each.feature_req=cfg.get(each.name,'feature_req')
    each.flags=[x.strip() for x in cfg.get(each.name,'flags').split(',')]
    each.win64_incl=int(cfg.get(each.name,'win64_incl'))
    each.fallback=[x.strip() for x in cfg.get(each.name,'fallback').split(',')]

    # add source files to list of files and import the corresponding templates
    files.append('source_files/'+each.file+".c")
    globals()[each.file+"_c"] = importlib.import_module("templates."+each.file+"_c")
        
# list of files to generate besides the special assembler files from above
# new files have to be added here. Furthermore, if the files use templates, the
# respective modules have to be imported from the templates directory (see line 25/26)
files.append('source_files/cpu.h')
files.append('source_files/firestarter_global.h')
files.append('source_files/generic.c')
files.append('source_files/help.c')
files.append('source_files/help.h')
files.append('source_files/INSTALL')
files.append('source_files/init_functions.c')
files.append('source_files/main.c')
files.append('source_files/Makefile')
files.append('source_files/README')
files.append('source_files/watchdog.c')
files.append('source_files/watchdog.h')
files.append('source_files/work.c')
files.append('source_files/work.h')
files.append('source_files/x86.c')

# add GPU files if CUDA support is enabled
if version.enable_cuda == 1:
    files.append('source_files/gpu.h')
    files.append('source_files/gpu.c')

# add windows main file if windows support is enabled
if version.enable_win64 == 1:
    files.append('source_files/main_win64.c')
    files.append('source_files/x86_win64.c')

# generate source code from files in source_files directory, apply patches defined in templates directory
for file in files:
    if verbose == True:
        print(" processing "+file.replace("source_files/",""))
    sys.stdout.flush()
    infile = dirname+file
    outfile = outdir+file.replace("source_files/","")
    source = open (infile, 'r')
    dest = open (outfile, 'w')
    lines = source.readlines()
    for line in lines:
        # copy lines that do not need modification to destination file
        # lines that contain '$' symbols (e.g. immediate values in asm blocks) go through the processing
        # of the conditions without changes.
        if (line.find("$",0,1) == -1):
            dest.write(line)
        # process templates and conditional lines
        # lines with conditions are only included if all required options are enabled
        # - conditions can be in arbitrary order
        # - for templates (and tabs) the syntax is: [conditions] $TEMPLATE (or $TAB) command
        else:
            while (line.find("$",0,1) == 0):
                # remove comments
                if (line.find("$$") == 0):
                    line = ""
                # conditions for optional features
                elif (line.find("$CUDA") == 0):
                    if version.enable_cuda == 1:
                        line = line.replace("$CUDA ","").replace("$CUDA","")
                    else:
                        line = ""
                elif (line.find("$WIN64") == 0):
                    if version.enable_win64 == 1:
                        line = line.replace("$WIN64 ","").replace("$WIN64","")
                    else:
                        line = ""
                elif (line.find("$MAC") == 0):
                    if version.enable_mac == 1:
                        line = line.replace("$MAC ","").replace("$MAC","")
                    else:
                        line = ""
                # special condition for Makefiles with multiple targets (at least one option with an extra make target is enabled)
                elif (line.find("$ALL") == 0):
                    if version.targets is not 'linux':
                        line = line.replace("$ALL ","").replace("$ALL","")
                    else:
                        line = ""
                # expand tabs in Makefile
                elif (line.find("$TAB") == 0):
                    line = line.replace("$TAB ","\t").replace("$TAB","\t")
                # process templates in source files
                elif (line.find("$TEMPLATE") == 0):
                    command = line.replace("$TEMPLATE ","")
                    exec(command)
                    line = ""
                else:
                    raise ValueError("The following line can't be parsed: {}".format(line))

            dest.write(line)
    source.close()
    dest.close()
