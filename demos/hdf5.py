import numpy as np
from traits.api import Array, String
from traitschema import Schema


class NamedMatrix(Schema):
    name = String()
    data = Array(dtype=np.float64)


matrix = NamedMatrix(name="riker", data=np.random.random((8, 8)))
matrix.to_hdf("out.h5")

new = NamedMatrix.from_hdf("out.h5")
print(new.name)
print(new.data)
