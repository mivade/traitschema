from __future__ import division

import json

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
        if isinstance(o, np.ndarray):
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

    def to_dict(self):
        """Return all visible traits as a dictionary."""
        return {name: getattr(self, name) for name in self.visible_traits()}

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
                # tt = trait.trait_type

                # Workaround for saving arrays containing unicode. When the
                # data type is unicode, each element is encoded as utf-8
                # before being saved to hdf5
                # FIXME: is there a better way of determining stringyness?
                data = getattr(self, name)
                if encode_string_arrays:
                    if trait.array is True and str(data.dtype).find("<U") != -1:
                        data = [s.encode(encoding) for s in data]

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
                if trait.desc is not None:
                    dset.attrs['desc'] = trait.desc

            hfile.attrs['classname'] = self.__class__.__name__
            hfile.attrs['python_module'] = self.__class__.__module__

    @classmethod
    def from_hdf(cls, filename):
        """Deserialize from HDF5 using :mod:`h5py`.

        Parameters
        ----------
        filename : str

        Returns
        -------
        Deserialized instance

        """
        if h5py is None:  # pragma: nocover
            raise OptionalDependencyMissingError("h5py not found")

        self = cls()
        with h5py.File(filename, 'r') as hfile:
            for name in self.visible_traits():
                setattr(self, name, hfile['/{}'.format(name)].value)
        return self

    def to_json(self, **kwargs):
        """Serialize to JSON. Keyword arguments are passed to :func:`json.dumps`.

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
        d = {name: getattr(self, name) for name in self.visible_traits()}
        kwargs['cls'] = _NumpyJsonEncoder
        return json.dumps(d, **kwargs)

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
