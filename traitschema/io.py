import contextlib
import os.path as osp
import shutil
from tempfile import mkdtemp

from .schema import Schema


class UnknownArchiveFormat(Exception):
    """Raised when a file extension doesn't match up with a supported archive
    format.

    """


@contextlib.contextmanager
def tempdir():
    dirname = mkdtemp()
    yield dirname
    shutil.rmtree(dirname)


def _get_archive_format(filename):
    formats = {
        '.zip': 'zip',
        '.tar.gz': 'gztar',
        '.tar.bz2': 'bztar',
        '.tar.xz': 'xztar',
        '.tar': 'tar',
    }
    for extension, format in formats.items():
        if filename.endswith(extension):
            return filename[:-len(extension)], format
    raise UnknownArchiveFormat


def bundle_schema(outfile, schema, format='npz'):
    """Bundle several :class:`Schema` objects into a single archive.

    Parameters
    ----------
    outfile : str
        Output bundle filename. Archive format is determined by the extension
        and can be any supported by :func:`shutil.make_archive`.
    schema : Dict[str, Schema]
        Dictionary of :class:`Schema` objects to bundle together. Keys are names
        to give each schema and are used when loading a bundle.
    format : str
        Format to save individual schema as (default: ``'npz'``).

    Notes
    -----
    Default options are used with all saving functions (e.g., no compression
    is used for individual serialized schema).

    """
    basename, archive_format = _get_archive_format(outfile)

    with tempdir() as staging_dir:
        for key, obj in schema.items():
            filename = osp.join(staging_dir, key + '.' + format)
            obj.save(filename)

        archve_filename = shutil.make_archive(basename, archive_format,
                                              staging_dir)
        shutil.move(archve_filename, outfile)
