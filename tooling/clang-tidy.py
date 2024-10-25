#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import json
from pathlib import Path
import subprocess
import click
import multiprocessing
import sys
import typing
import random
from multiprocessing import Pool
from functools import partial

# Find all source files from the compile commands database that are in a specific directory
def find_source_files_from_compile_commands(compile_commands_path: Path, sources_dir: Path) -> typing.List[Path]:
    with open(compile_commands_path, 'r') as fp:
        compile_commands = json.loads(fp.read())
        sources = [ entry['file'] for entry in compile_commands ]
        sources = list(filter(lambda file: str(file).startswith(str(sources_dir)), sources))
        return sources

# Find all source and header files in the project root that belong to FIRESTARTER
def find_source_and_header_files(project_root: Path, build_root: Path) -> typing.List[Path]:
    src_path = project_root / Path('src')
    include_path = project_root / Path('include')

    # find all cpp file from the compile commands database
    compile_commands_path = build_root / Path('compile_commands.json')
    files = find_source_files_from_compile_commands(compile_commands_path, src_path)

    # find all headers based on glob
    files += glob.glob(f'{include_path}/**/*.hpp', recursive=True)
    files += glob.glob(f'{include_path}/**/*.h', recursive=True)

    return files

# Split a list of paths into multiple list of paths
def split_in_chunks(chunk_size: int, input: typing.List[Path]) -> typing.List[typing.List[Path]]:
    length = len(input) // chunk_size
    if length * chunk_size < len(input):
        length += 1
    
    return [ input[i:i+length] for i in range(0, len(input), length)]

# Run clang-tidy on a set of input files and return the stdout
def run_clang_tidy(files: typing.List[Path], project_root_path: Path, build_root_path: Path, clang_tidy_file_path: Path) -> bytes:
    command_args = ['clang-tidy', '-extra-arg=-std=c++17', f'-p={build_root_path}', f'--config-file={clang_tidy_file_path}', '--header-filter=include/firestarter/*', '--format-style=file']
    command_args += files
    print(f'Starting {command_args}')
    p = subprocess.Popen(command_args, stdout=subprocess.PIPE, cwd=project_root_path)

    # Wait for clang-tidy instances to terminate
    if p.poll() is None:
        p.wait()
        stdout, _ = p.communicate()
        return stdout + b'\n'
    
    return b''

@click.command()
@click.option('--project-root', default=Path(__file__).parent.parent.absolute(), help='The folder where the git repository is located.')
@click.option('--build-root', help='The folder where the compile_commands.json is located.', required=True)
@click.option('--cores', default=multiprocessing.cpu_count(), help='The number of clang-tidy processes to spawn.')
def clang_tidy_report(project_root, build_root, cores):
    project_root_path = Path(project_root).absolute()
    build_root_path = Path(build_root).absolute()

    print(f'Looking for compile_commands.json in {build_root_path}')
    compile_commands_path = build_root_path / Path('compile_commands.json')
    if compile_commands_path.exists():
        print(f'Found {compile_commands_path}')
    else:
        sys.exit("Dind't find compile_commands.json. Aborting.")
        
    print(f'Looking for .clang-tidy in {project_root_path}')
    clang_tidy_file_path = project_root_path / Path('.clang-tidy')
    if clang_tidy_file_path.exists():
        print(f'Found {clang_tidy_file_path}')
    else:
        sys.exit("Dind't find .clang-tidy. Aborting.")

    files = find_source_and_header_files(project_root_path, build_root_path)
    print(f'Found {len(files)} source and header files.')
    
    print(f'Lanching {cores} instances of clang-tidy in project root: {project_root_path}')

    # Scramble files to improve runtime performance
    files_scrambled = files.copy()
    random.shuffle(files_scrambled)

    with Pool(cores) as p:
        stdout = p.map(partial(run_clang_tidy, project_root_path=project_root_path, build_root_path=build_root_path, clang_tidy_file_path=clang_tidy_file_path), split_in_chunks(cores, files_scrambled))

    clang_tidy_report_file = build_root_path / Path('clang-tidy-report.txt')
    print(f'Writing report to {clang_tidy_report_file}')
    with open(clang_tidy_report_file, 'wb') as fp:
        fp.write(b''.join(stdout))

if __name__ == '__main__':
    clang_tidy_report()