traitschema
===========

.. image:: https://travis-ci.org/mivade/traitschema.svg?branch=master
    :target: https://travis-ci.org/mivade/traitschema

.. image:: https://codecov.io/gh/mivade/traitschema/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/mivade/traitschema

Create serializable, type-checked schema using traits_ and Numpy. A typical use
case involves saving several Numpy arrays of varying shape.

.. _traits: http://docs.enthought.com/traits/


Defining schema
---------------

.. note::

    The following assumes a basic familiarity with the ``traits`` package. See
    its `documentation <http://docs.enthought.com/traits/>`_ for details.

In order to be able to properly serialize data, non-scalar traits should be
declared as a ``traits.api.Array`` type. Example::

    import numpy as np
    from traits.api import Array, String
    from traitschema import Schema

    class NamedMatrix(Schema):
        name = String()
        data = Array(dtype=np.float64)

    matrix = NamedMatrix(name="name", data=np.random.random((8, 8)))


Saving and loading
------------------

Data can be stored in the following formats:

* HDF5 via ``h5py``
* JSON via the standard library ``json`` module
* Numpy ``npz`` format
