from setuptools import setup
from traitschema import __version__

with open("README.rst", 'r') as f:
    README = f.read()

setup(
    name="traitschema",
    version=__version__,
    description="Serializable schema using traits",
    long_description=README,
    url="https://github.com/mivade/traitschema",
    author="Michael V. DePalatis",
    author_email="mike@depalatis.net",
    license="BSD",
    packages=['traitschema'],
    package_data={
        "": ["*.txt", "*.json"]
    },
    # install_requires=[
    #     "numpy",
    #     "traits",
    # ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python"
    ]
)
