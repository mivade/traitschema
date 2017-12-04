#!/usr/bin/env python

from __future__ import print_function

from argparse import ArgumentParser
import glob
import platform
import os
import shlex
import shutil
from subprocess import check_call
import sys

parser = ArgumentParser()
parser.add_argument('--no-clean', action='store_true', default=False,
                    help='remove build directory')
parser.add_argument('--no-build', action='store_true', default=False,
                    help='do not build packages')
parser.add_argument('--no-convert', action='store_true', default=False,
                    help='do not convert packages for other platforms')


def clean():
    """Clean the build directory."""
    print("rm -rf build/")
    try:
        shutil.rmtree('build')
        os.mkdir('build')
    except OSError:
        pass


def build():
    """Build conda packages."""
    build_cmd = "conda build conda.recipe --output-folder build/"
    print(build_cmd)
    check_call(shlex.split(build_cmd))


def convert():
    """Convert conda packages to other platforms."""
    os_name = {
        'darwin': 'osx',
        'win32': 'win',
        'linux': 'linux'
    }[sys.platform]
    dirname = '{}-{}'.format(os_name, platform.architecture()[0][:2])
    files = glob.glob('build/{}/*.tar.bz2'.format(dirname))

    for filename in files:
        convert_cmd = "conda convert {} -p all -o build/".format(filename)
        print(convert_cmd)
        check_call(shlex.split(convert_cmd))


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.no_clean:
        clean()

    if not args.no_build:
        build()

    if not args.no_convert:
        convert()
