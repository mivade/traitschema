"""Demo showing saving as and loading from JSON."""

import numpy as np
from traits.api import Array, String
from traitschema import Schema


class MatrixSchema(Schema):
    meta = String("default")
    data = Array(dtype=np.float64, shape=(8, 8))


matrix = MatrixSchema()
matrix.data = np.random.random((8, 8))

with open('out.json', 'w') as jf:
    jf.write(matrix.to_json(indent=2))

with open('out.json', 'r') as jf:
    print(MatrixSchema.from_json(jf).data)
