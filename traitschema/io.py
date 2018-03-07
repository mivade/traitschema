import contextlib
import json
import os.path as osp
import shutil
from tempfile import mkdtemp

BUNDLE_VERSION = 1


class UnsupportedArchiveFormat(Exception):
    """Raised when a file extension doesn't match up with a supported archive
    format.

    """


@contextlib.contextmanager
def tempdir():
    dirname = mkdtemp()
    yield dirname
    shutil.rmtree(dirname)


def bundle_schema(outfile, schema, format='npz'):
    """Bundle several :class:`Schema` objects into a single archive.

    Parameters
    ----------
    outfile : str
        Output bundle filename. Only zip archives are supported.
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
    if not outfile.endswith('.zip'):
        raise UnsupportedArchiveFormat
    basename, ext = osp.splitext(outfile)
    archive_format = ext.lstrip('.')

    with tempdir() as staging_dir:
        index = {'schema': {}, 'bundle_version': BUNDLE_VERSION}
        for key, obj in schema.items():
            filename = osp.join(staging_dir, key + '.' + format)
            obj.save(filename)
            index['schema'][key] = {
                'filename': filename,
                'classname': obj.__class__.__name__
            }

        with open(osp.join(staging_dir, '.index.json'), 'w') as f:
            f.write(json.dumps(index))

        archve_filename = shutil.make_archive(basename, archive_format,
                                              staging_dir)
        shutil.move(archve_filename, outfile)


def load_bundle(filename):
    """Loads a bundle of schema saved with :func:`bundle_schema`.

    Parameters
    ----------
    filename : str
        Path to bundled schema archive.

    """
    raise NotImplementedError
