import glob
import os
import re

from setuptools import Extension as _Extension
from setuptools import find_packages, setup

with open(os.path.join("./mgeconvert", "version.py")) as f:
    __version_py__ = f.read()

MEGENGINE_LOWER = re.search(r"MEGENGINE_LOWER = \"(.*)\"", __version_py__).group(1)

__version__ = re.search(r"__version__ = \"(.*)\"", __version_py__).group(1)


class Extension(_Extension):
    def __init__(self, *a, include_dirs=None, **kw):
        super(Extension, self).__init__(*a, **kw)
        self._include_dirs = include_dirs or []

    """
    Here we make `include_dirs` an property to make it an deferred attibute,
    so the `numpy` package can be installed via `setup_requires`, and imported
    in this property.
    """

    @property
    def include_dirs(self):
        import numpy as np

        return self._include_dirs + [np.get_include()]

    @include_dirs.setter
    def include_dirs(self, value):
        self._include_dirs = value


setup(
    name="mgeconvert",
    version=__version__,
    description="MegEngine Converter",
    author="Megvii Engine Team",
    author_email="brain-engine@megvii.com",
    url="https://github.com/MegEngine/mgeconvert",
    packages=find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    setup_requires=["Cython >= 0.20", "numpy"],
    install_requires=["megengine >={}".format(MEGENGINE_LOWER), "numpy"],
    scripts=glob.glob("util/*"),
)
