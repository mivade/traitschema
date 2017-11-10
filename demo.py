import numpy as np
from traits.api import Array
from traitschema import Schema


class MatrixSchema(Schema):
    data = Array(dtype=np.float64, shape=(8, 8))


matrix = MatrixSchema()
matrix.data = np.random.random((8, 8))

with open('out.json', 'w') as jf:
    jf.write(matrix.to_json())

with open('out.json', 'r') as jf:
    print(MatrixSchema.from_json(jf).data)
