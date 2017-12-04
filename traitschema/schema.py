from __future__ import division

import json

import numpy as np
from traits.api import HasTraits


try:
    import h5py
except ImportError:  # pragma: nocover
    h5py = None


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

    def to_hdf(self, filename, mode='w'):
        """Serialize to HDF5 using :mod:`h5py`.

        Parameters
        ----------
        filename : str
            Path to save HDF5 file to.
        mode : str
            Default: ``'w'``

        """
        if h5py is None:
            raise RuntimeError("h5py not found")

        with h5py.File(filename, mode) as hfile:
            for name in self.class_visible_traits():
                trait = self.trait(name)
                # tt = trait.trait_type

                chunks = True if trait.array else False
                dset = hfile.create_dataset('/{}'.format(name),
                                            data=getattr(self, name),
                                            chunks=chunks)
                if trait.desc is not None:
                    dset.attrs['desc'] = trait.desc

            hfile.attrs['classname'] = self.__class__.__name__

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
        if h5py is None:
            raise RuntimeError("h5py not found")

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
