traitschema
===========

.. image:: https://travis-ci.org/mivade/traitschema.svg?branch=master
    :target: https://travis-ci.org/mivade/traitschema

.. image:: https://readthedocs.org/projects/traitschema/badge/?version=latest
    :target: http://traitschema.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/mivade/traitschema/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mivade/traitschema

.. image:: https://img.shields.io/github/release/mivade/traitschema.svg
    :target: https://github.com/mivade/traitschema

Create serializable, type-checked schema using traits_ and Numpy. A typical use
case involves saving several Numpy arrays of varying shape and type.

.. _traits: http://docs.enthought.com/traits/


Defining schema
---------------

.. note::

    The following assumes a basic familiarity with the ``traits`` package. See
    its `documentation <http://docs.enthought.com/traits/>`_ for details.

In order to be able to properly serialize data, non-scalar traits should be
declared as a ``traits.api.Array`` type. Example:

.. code-block:: python

    import numpy as np
    from traits.api import Array, String
    from traitschema import Schema

    class NamedMatrix(Schema):
        name = String()
        data = Array(dtype=np.float64)

    matrix = NamedMatrix(name="name", data=np.random.random((8, 8)))

For other demos, see the ``demos`` directory.


Saving and loading
------------------

Data can be stored in the following formats:

* HDF5 via ``h5py``
* JSON via the standard library ``json`` module
* Numpy ``npz`` format
