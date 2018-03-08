from __future__ import division

import json
import os.path as osp

import numpy as np
from traits.api import HasTraits


try:
    import h5py
except ImportError:  # pragma: nocover
    h5py = None


class OptionalDependencyMissingError(Exception):
    """Raised when an optional dependency such as h5py is required but not
    installed.

    """


class _NumpyJsonEncoder(json.JSONEncoder):
    def default(self, o):
        # TODO: Figure out the right way to do this that maintains dtypes
        if isinstance(o, np.recarray):
            raise RuntimeError("Recarrays are not currently supported when "
                               "saving to json")
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return json.JSONEncoder.default(self, o)


class Schema(HasTraits):
    """Extension to :class:`HasTraits` to add methods for automatically saving
    and loading typed data.

    Examples
    --------
    Create a new data class::

        import numpy as np
        from traits.api import Array
        from traitschema import Schema

        class Matrix(Schema):
            data = Array(dtype=np.float64)

        matrix = Matrix(data=np.random.random((8, 8)))

    Serialize to HDF5 using :mod:`h5py`::

        matrix.to_hdf("out.h5")

    Load from HDF5::

        matrix_copy = Matrix.from_hdf("out.h5")

    """
    def __init__(self, **kwargs):
        super(Schema, self).__init__(**kwargs)

        traits = self.class_visible_traits()
        for key, value in kwargs.items():
            if key not in traits:
                raise RuntimeError("trait {} is not in {}".format(
                    key, self.__class__.__name__
                ))
            setattr(self, key, value)

    def __str__(self):  # pragma: nocover
        attr_strs = ["{}={}".format(attr, getattr(self, attr))
                     for attr in self.visible_traits()]
        return "<{}({})>".format(self.__class__.__name__, '\n    '.join(attr_strs))

    def __repr__(self):  # pragma: nocover
        return self.__str__()

    def __eq__(self, other):
        for attr in self.visible_traits():
            this = getattr(self, attr)
            that = getattr(other, attr)
            try:
                if this != that:
                    return False
            except AttributeError:
                return False
            except ValueError:
                if not all(this == that):
                    return False
        return True

    def to_dict(self):
        """Return all visible traits as a dictionary."""
        return {name: getattr(self, name) for name in self.visible_traits()}

    def save(self, filename):
        """Serialize using the type determined by the file extension.

        Parameters
        ----------
        filename : str
            Full output path.

        Notes
        -----
        Only default saving options are used, so this method is less flexible
        than using the ``to_xyz`` methods instead.

        """
        func = {
            '.npz': 'to_npz',
            '.h5': 'to_hdf',
            '.json': 'to_json',
        }[osp.splitext(filename)[1]]
        if func != 'to_json':
            getattr(self, func)(filename)
        else:
            with open(filename, 'w') as jf:
                jf.write(getattr(self, func)())

    @classmethod
    def load(cls, filename):
        """Counterpart to :meth:`save`."""
        func = {
            '.npz': 'from_npz',
            '.h5': 'from_hdf',
            '.json': 'from_json',
        }[osp.splitext(filename)[1]]
        if func != 'from_json':
            return getattr(cls, func)(filename)
        else:
            with open(filename, 'r') as jf:
                return getattr(cls, func)(jf)

    def to_npz(self, filename, compress=False):
        """Save in numpy's npz archive format.

        Parameters
        ----------
        filename : str
        compress : bool
            Save as a compressed archive (default: False)

        Notes
        -----
        To ensure loading of scalar values works as expected, casting traits
        should be used (e.g., ``CStr`` instead of ``String`` or ``Str``). See
        the :mod:`traits` documentation for details.

        """
        save = np.savez_compressed if compress else np.savez
        attrs = self.to_dict()
        save(filename, **attrs)

    @classmethod
    def from_npz(cls, filename):
        """Load data from numpy's npz format.

        Parameters
        ----------
        filename : str

        """
        npz = np.load(filename)
        attrs = {key: value for key, value in npz.items()}
        self = cls(**attrs)
        return self

    def to_hdf(self, filename, mode='w', compression=None,
               compression_opts=None, encode_string_arrays=True,
               encoding='utf8'):
        """Serialize to HDF5 using :mod:`h5py`.

        Parameters
        ----------
        filename : str
            Path to save HDF5 file to.
        mode : str
            Default: ``'w'``
        compression : str or None
            Compression to use with arrays (see :mod:`h5py` documentation for
            valid choices).
        compression_opts : int or None
            Compression options, generally a number specifying compression level
            (see :mod:`h5py` documentation for details).
        encode_string_arrays : bool
            When True, force encoding of arrays of unicode strings using the
            ``encoding`` keyword argument. Not setting this will result in
            errors if using arrays of unicode strings. Default: True.
        encoding : str
            Encoding to use when forcing encoding of unicode string arrays.
            Default: ``'utf8'``.

        Notes
        -----
        Each stored dataset will also have a ``desc`` attribute which uses the
        ``desc`` attribute of each trait.

        The root node also has attributes:

        * ``classname`` - the class name of the instance being serialized
        * ``python_module`` - the Python module in which the class is defined

        """
        if h5py is None:  # pragma: nocover
            raise OptionalDependencyMissingError("h5py not found")

        with h5py.File(filename, mode) as hfile:
            for name in self.class_visible_traits():
                trait = self.trait(name)

                # Workaround for saving arrays containing unicode. When the
                # data type is unicode, each element is encoded as utf-8
                # before being saved to hdf5
                data = getattr(self, name)

                if data is None:
                    # If a trait has not been populated, don't try to store it
                    continue

                data_is_recarray = isinstance(data, np.recarray)
                if trait.array is True and encode_string_arrays:
                    # Encode each element of an array containing unicode
                    # elements
                    if ~data_is_recarray and data.dtype.char == 'U':
                        data = [s.encode(encoding) for s in data]

                    elif data_is_recarray:
                        # Determine what the final dtypes will be
                        final_dtypes = []
                        unicode_fields = []
                        for i, field in enumerate(data.dtype.names):
                            if data[field].dtype.kind != 'U':
                                final_dtypes.append((field,
                                                     data[field].dtype.str))
                            else:
                                final_dtypes.append((field, '<S256'))
                                unicode_fields.append(field)

                        # Update dtypes of the data. This will coerce the
                        # unicode fields to bytes automatically
                        data = data.astype(final_dtypes)

                chunks = True if trait.array else False

                compression_kwargs = {}
                if chunks:
                    if compression is not None:
                        compression_kwargs['compression'] = compression
                        if compression_opts is not None:
                            compression_kwargs['compression_opts'] = compression_opts

                dset = hfile.create_dataset('/{}'.format(name),
                                            data=data,
                                            chunks=chunks,
                                            **compression_kwargs)

                # Store the data type as an attribute to make it easier to
                # reconstruct with correct data types
                dset.attrs['type'] = str(type(data))

                if trait.desc is not None:
                    dset.attrs['desc'] = trait.desc

            hfile.attrs['classname'] = self.__class__.__name__
            hfile.attrs['python_module'] = self.__class__.__module__

    @classmethod
    def from_hdf(cls, filename, decode_string_arrays=True, encoding='utf-8'):
        """Deserialize from HDF5 using :mod:`h5py`.

        Parameters
        ----------
        filename : str
        decode_string_arrays: bool
            Arrays of bytes should be decoded into strings
        encoding: str
            Encoding scheme to use for decoding

        Returns
        -------
        Deserialized instance

        """
        if h5py is None:  # pragma: nocover
            raise OptionalDependencyMissingError("h5py not found")

        self = cls()
        with h5py.File(filename, 'r') as hfile:
            for name in self.visible_traits():
                trait = self.trait(name)
                if name not in hfile:
                    continue
                dset = hfile['/{}'.format(name)]
                data = dset.value

                # Use type attribute to determine how to proceed
                data_is_recarray = dset.attrs['type'] == str(np.recarray)

                if trait.array is True and decode_string_arrays:
                    # Encode each element of an array containing bytes
                    if ~data_is_recarray and data.dtype.char == 'S':
                        data = [s.decode(encoding) for s in data]

                    elif data_is_recarray:
                        # Determine what the final dtypes will be
                        final_dtypes = []
                        bytes_fields = []
                        for i, field in enumerate(data.dtype.names):
                            if data[field].dtype.kind != 'S':
                                final_dtypes.append((field,
                                                     data[field].dtype.str))
                            else:
                                final_dtypes.append((field, '<U256'))
                                bytes_fields.append(field)

                        # Update dtypes of the data. This will coerce the
                        # bytes fields to unicode automatically
                        data = data.astype(final_dtypes)

                setattr(self, name, data)

        return self

    # FIXME: this should optionally write to a file
    def to_json(self, json_kwargs={}):
        """Serialize to JSON.

        Parameters
        ----------
        json_kwargs : dict
            Keyword arguments to pass to :func:`json.dumps`.

        Returns
        -------
        JSON string.

        Notes
        -----
        This uses a custom JSON encoder to handle numpy arrays but could
        conceivably lose precision. If this is important, please consider
        serializing in HDF5 format instead. As a consequence of using a custom
        encoder, the ``cls`` keyword arugment, if passed, will be ignored.

        """
        data = {name: getattr(self, name) for name in self.visible_traits()}
        json_kwargs['cls'] = _NumpyJsonEncoder
        return json.dumps(data, **json_kwargs)

    # FIXME allow filenames
    @classmethod
    def from_json(cls, data):
        """Deserialize from a JSON string or file.

        Parameters
        ----------
        data : str or file-like

        Returns
        -------
        Deserialized instance

        """
        if not isinstance(data, str):
            loaded = json.load(data)
        else:
            loaded = json.loads(data)

        return cls(**{key: value for key, value in loaded.items()})
