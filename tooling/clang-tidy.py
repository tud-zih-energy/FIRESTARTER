#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from pathlib import Path
import subprocess
import click
import multiprocessing
import sys
import typing

# Find all source and header files in the project root that belong to FIRESTARTER
def find_source_and_header_files(project_root: Path) -> typing.List[ Path ]:
    src_path = project_root / Path('src')
    include_path = project_root / Path('include')
    files = glob.glob(f'{src_path}/**/*.cpp', recursive=True)
    files += glob.glob(f'{include_path}/**/*.hpp', recursive=True)
    files += glob.glob(f'{include_path}/**/*.h', recursive=True)
    return files

# Split a list of paths into multiple list of paths
def split_in_chunks(chunk_size: int, input: typing.List[Path]) -> typing.List[typing.List[Path]]:
    length = len(input) // chunk_size
    if length * chunk_size < len(input):
        length += 1
    
    return [ input[i:i+length] for i in range(0, len(input), length)]

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

    files = find_source_and_header_files(project_root_path)
    print(f'Found {len(files)} source and header files.')
    
    print(f'Lanching {cores} instances of clang-tidy in project root: {project_root_path}')

    processes = set()
    for chunck in split_in_chunks(cores, files):
        command_args = ['clang-tidy', '-extra-arg=-std=c++17', f'-p={build_root_path}', f'--config-file={clang_tidy_file_path}', '--header-filter=include/firestarter/*', '--format-style=file']
        command_args += chunck
        print(f'Starting {command_args}')
        processes.add(subprocess.Popen(command_args, stdout=subprocess.PIPE, cwd=project_root_path))

    # Wait for clang-tidy instances to terminate
    complete_stdout = b''
    for p in processes:
        if p.poll() is None:
            p.wait()
            stdout, _ = p.communicate()
            complete_stdout += stdout + b'\n'

    clang_tidy_report_file = build_root_path / Path('clang-tidy-report.txt')
    print(f'Writing report to {clang_tidy_report_file}')
    with open(clang_tidy_report_file, 'wb') as fp:
        fp.write(complete_stdout)

if __name__ == '__main__':
    clang_tidy_report()