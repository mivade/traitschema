import json
import os.path as osp
import string
import random

import h5py
import numpy as np
from numpy.testing import assert_equal
import pytest
from traits.api import Array, CStr, Float, ArrayOrNone
from traitschema import Schema


def generate_random_string(size=10):
    mystring = ''.join(random.choice(string.printable)
                       for _ in range(size))
    return mystring


@pytest.fixture(scope='session')
def sample_recarray():
    sample = np.rec.array([('sample_unicode_field', 0),
                           ('another_sample', 1)],
                          dtype=[('field_1', '<U256'),
                                 ('field_2', '<i8')])
    return sample


class SomeSchema(Schema):
    x = Array(dtype=np.float)
    y = Array()
    z = ArrayOrNone()
    name = CStr()


@pytest.mark.parametrize('format', ['.npz', '.h5', '.json'])
def test_save_load(format, tmpdir):
    x = np.random.random(100)
    y = np.linspace(0, 100, 100, dtype=np.int)
    z = None
    name = 'a name'

    schema = SomeSchema(x=x, y=y, z=z, name=name)
    outfile = str(tmpdir.join('filename' + format))
    schema.save(outfile)
    assert osp.exists(outfile)

    loaded = SomeSchema.load(outfile)


@pytest.mark.parametrize('mode', ['w', 'a'])
@pytest.mark.parametrize('desc', ['a number', None])
@pytest.mark.parametrize('compression', [None, 'gzip', 'lzf'])
@pytest.mark.parametrize('compression_opts', [None, 6])
@pytest.mark.parametrize('encoding', ['utf8', 'ascii', 'latin1'])
def test_to_hdf(mode, desc, compression, compression_opts, encoding, tmpdir,
                sample_recarray):
    class MySchema(Schema):
        u = ArrayOrNone()
        v = Array(desc=desc)
        w = Float(desc=desc)
        x = Array(dtype=np.float64, desc=desc)
        y = Array(dtype=np.int32, desc=desc)
        z = Array(dtype=np.unicode, desc=desc)
    obj = MySchema(u=None,
                   v=sample_recarray,
                   w=0.01,
                   x=np.random.random(100),
                   y=np.random.random(100),
                   z=np.array([generate_random_string() for _ in range(100)],
                              dtype=np.unicode))

    filename = str(tmpdir.join('test.h5'))

    call = lambda: obj.to_hdf(
        filename, mode, compression=compression, compression_opts=compression_opts,
        encode_string_arrays=True, encoding=encoding
    )

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
            assert hfile['/v'].attrs['desc'] == desc
            assert hfile['/w'].attrs['desc'] == desc
            assert hfile['/x'].attrs['desc'] == desc
            assert hfile['/y'].attrs['desc'] == desc
            assert hfile['/z'].attrs['desc'] == desc
        else:
            # 'type' attribute should be populated
            assert 'type' in hfile['/v'].attrs.keys()
            assert 'type' in hfile['/w'].attrs.keys()
            assert 'type' in hfile['/x'].attrs.keys()
            assert 'type' in hfile['/y'].attrs.keys()
            assert 'type' in hfile['/z'].attrs.keys()


@pytest.mark.parametrize("encoding", ['utf-8'])
@pytest.mark.parametrize("decode_string_arrays", [True, False])
def test_from_hdf(tmpdir, encoding, decode_string_arrays, sample_recarray):
    w = sample_recarray
    w = w.astype(dtype=[('field_1', '<S256'), ('field_2', '<i8')])
    x = np.arange(10)
    y = np.arange(10, dtype=np.int32)
    z = np.array([generate_random_string().encode('utf-8') for _ in range(5)])

    path = str(tmpdir.join('test.h5'))

    with h5py.File(path, 'w') as hfile:
        dset_w = hfile.create_dataset('/w', data=w)
        dset_w.attrs['type'] = str(np.recarray)
        dset_x = hfile.create_dataset('/x', data=x, chunks=True)
        dset_x.attrs['type'] = str(type(x))
        dset_y = hfile.create_dataset('/y', data=y, chunks=True)
        dset_y.attrs['type'] = str(type(y))
        dset_z = hfile.create_dataset('/z', data=z, chunks=True)
        dset_z.attrs['type'] = str(type(z))

    class MySchema(Schema):
        w = Array()
        x = Array(dtype=np.float64)
        y = Array(dtype=np.int32)
        z = Array()

    instance = MySchema.from_hdf(path,
                                 decode_string_arrays=decode_string_arrays,
                                 encoding=encoding)

    assert_equal(instance.x, x)
    assert_equal(instance.y, y)

    if decode_string_arrays:
        assert_equal(instance.z, [s.decode(encoding) for s in z])
    else:
        assert_equal(instance.z, z)


def test_to_dict(sample_recarray):
    obj = SomeSchema()
    obj.name = 'test'
    obj.x = [1, 2, 3]
    obj.y = sample_recarray

    d = obj.to_dict()
    assert d['name'] == obj.name
    assert_equal(obj.x, d['x'])
    assert_equal(obj.y, d['y'])


@pytest.mark.parametrize('compress', [True, False])
def test_to_npz(compress, tmpdir, sample_recarray):
    obj = SomeSchema(name='test',
                     x=[1, 2, 3],
                     y=sample_recarray,
                     z=None)
    path = str(tmpdir.join('test.npz'))
    obj.to_npz(path, compress=compress)

    npz = np.load(path)
    assert str(npz['name']) == 'test'
    assert_equal([1, 2, 3], npz['x'])
    assert_equal(sample_recarray, npz['y'])


def test_from_npz(tmpdir, sample_recarray):
    path = str(tmpdir.join('output.npz'))
    np.savez(path, x=[1, 2, 3], name='test', y=sample_recarray)

    obj = SomeSchema.from_npz(path)
    assert obj.name == 'test'
    assert_equal([1, 2, 3], obj.x)
    assert_equal(sample_recarray, obj.y)


def test_to_json(sample_recarray):
    obj_with_recarray = SomeSchema(x=list(range(10)), name="whatever",
                                   y=sample_recarray,
                                   z=None)
    with pytest.raises(RuntimeError):
        jobj = obj_with_recarray.to_json()

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
