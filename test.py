import json
import pytest
import numpy as np
from numpy.testing import assert_equal
import h5py
import string
import random

from traits.api import Array, CStr, Float
from traitschema import Schema


def generate_random_string(size=10):
    mystring = ''.join(random.choice(string.ascii_uppercase + string.digits)
                       for _ in range(size))
    return mystring


class SomeSchema(Schema):
    x = Array(dtype=np.float)
    name = CStr()


@pytest.mark.parametrize('mode', ['w', 'a'])
@pytest.mark.parametrize('desc', ['a float', 'a number', None, 'a string'])
@pytest.mark.parametrize('compression', [None, 'gzip', 'lzf'])
@pytest.mark.parametrize('compression_opts', [None, 6])
def test_to_hdf(mode, desc, compression, compression_opts, tmpdir):
    class MySchema(Schema):
        w = Float(desc=desc)
        x = Array(dtype=np.float64, desc=desc)
        y = Array(dtype=np.int32, desc=desc)
        z = Array(dtype=np.unicode, desc=desc)
    obj = MySchema(w=0.01, x=np.random.random(100), y=np.random.random(100),
                   z=np.array([generate_random_string() for _ in range(100)],
                              dtype=np.unicode))
    filename = str(tmpdir.join('test.h5'))

    call = lambda: obj.to_hdf(
        filename, mode, compression=compression, compression_opts=compression_opts)

    if compression == 'lzf' and compression_opts is not None:
        with pytest.raises(ValueError):
            call()
        return
    else:
        call()

    with h5py.File(filename, 'r') as hfile:
        assert_equal(hfile['/w'].value, obj.w)
        assert_equal(hfile['/x'][:], obj.x)
        assert_equal(hfile['/y'][:], obj.y)
        assert_equal([s.decode('utf8') for s in hfile['/z'][:]], obj.z)
        assert hfile.attrs['classname'] == 'MySchema'
        assert hfile.attrs['python_module'] == 'test'
        if desc is not None:
            assert hfile['/w'].attrs['desc'] == desc
            assert hfile['/x'].attrs['desc'] == desc
            assert hfile['/y'].attrs['desc'] == desc
            assert hfile['/z'].attrs['desc'] == desc
        else:
            assert len(hfile['/x'].attrs.keys()) == 0
            assert len(hfile['/x'].attrs.keys()) == 0
            assert len(hfile['/y'].attrs.keys()) == 0
            assert len(hfile['/z'].attrs.keys()) == 0


def test_from_hdf(tmpdir):
    x = np.arange(10)
    y = np.arange(10, dtype=np.int32)
    z = np.array([generate_random_string().encode('utf8') for _ in range(5)])

    path = str(tmpdir.join('test.h5'))

    with h5py.File(path, 'w') as hfile:
        hfile.create_dataset('/x', data=x, chunks=True)
        hfile.create_dataset('/y', data=y, chunks=True)
        hfile.create_dataset('/z', data=z, chunks=True)

    class MySchema(Schema):
        x = Array(dtype=np.float64)
        y = Array(dtype=np.int32)
        z = Array(dtype=np.unicode)

    instance = MySchema.from_hdf(path)

    assert_equal(instance.x, x)
    assert_equal(instance.y, y)
    assert_equal(instance.z, [s.decode('utf8') for s in z])


def test_to_dict():
    obj = SomeSchema()
    obj.name = 'test'
    obj.x = [1, 2, 3]

    d = obj.to_dict()
    assert d['name'] == obj.name
    assert_equal(obj.x, d['x'])


@pytest.mark.parametrize('compress', [True, False])
def test_to_npz(compress, tmpdir):
    obj = SomeSchema(name='test', x=[1, 2, 3])
    path = str(tmpdir.join('test.npz'))
    obj.to_npz(path, compress=compress)

    npz = np.load(path)
    assert str(npz['name']) == 'test'
    assert_equal([1, 2, 3], npz['x'])


def test_from_npz(tmpdir):
    path = str(tmpdir.join('output.npz'))
    np.savez(path, x=[1, 2, 3], name='test')

    obj = SomeSchema.from_npz(path)
    assert obj.name == 'test'
    assert_equal([1, 2, 3], obj.x)


def test_to_json():
    obj = SomeSchema(x=list(range(10)), name="whatever")
    jobj = obj.to_json()

    loaded = json.loads(jobj)
    assert_equal(loaded['x'], obj.x)
    assert loaded['name'] == obj.name


@pytest.mark.parametrize('fromfile', [True, False])
def test_from_json(fromfile, tmpdir):
    data = {
        "x": list(range(10)),
        "name": "whatever"
    }

    if not fromfile:
        obj = SomeSchema.from_json(json.dumps(data))
    else:
        filename = str(tmpdir.join('test.json'))
        with open(filename, 'w') as f:
            json.dump(data, f)

        with open(filename, 'r') as f:
            obj = SomeSchema.from_json(f)

    assert_equal(obj.x, data['x'])
    assert obj.name == data['name']
