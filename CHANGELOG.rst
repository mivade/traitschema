Changes
=======

Version 1.1.3
-------------

**2018-01-25**

* Allow ``None`` values and improve recarray handling when using the HDF5
  format (#8)


Version 1.1.2
-------------

**2018-01-23**

* Added options for encoding string arrays when saving as HDF5 (#3 and #4)
* Added support for saving Numpy recarrays in HDF5 and npy formats (#5)

Version 1.1.1
-------------

**2018-01-20**

* Added compression options for saving in HDF5 format
* Included ``Pipfile`` for use with ``pipenv``
* Automatically encode unicode when saving to HDF5

Version 1.1.0
-------------

**2017-12-12**

* Added ``python_module`` attribute to HDF5 files

Version 1.0.0
-------------

**2017-12-04**

First official release featuring serialization in the following formats:

* HDF5
* JSON
* Numpy's native ``.npz`` archive format
