import glob
import os
import re
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

with open(os.path.join("./mgeconvert", "version.py")) as f:
    __version_py__ = f.read()

MEGENGINE_LOWER = re.search(r"MEGENGINE_LOWER = \"(.*)\"", __version_py__).group(1)

__version__ = re.search(r"__version__ = \"(.*)\"", __version_py__).group(1)


class CMakeExtension(Extension):
    def __init__(self, name, source_dir, lib_dir, products):
        super().__init__(name, sources=[])
        self.source_dir = source_dir
        self.lib_dir = os.path.join(source_dir, lib_dir)
        self.products = products


class cmake_build_ext(build_ext):
    def run(self):
        for ext in self.extensions:
            try:
                self.build_cmake(ext)
            except subprocess.CalledProcessError:
                continue

    def build_cmake(self, ext):
        source_dir = os.path.abspath(ext.source_dir)

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        path = os.path.join(source_dir, "swig", "numpy.i")
        url = "https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i"
        if not os.path.exists(path):
            subprocess.check_call(["wget", "-P", os.path.dirname(path), url])

        subprocess.check_call(["cmake", source_dir], cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)

        for product in ext.products:
            source_file = os.path.join(self.build_temp, product)
            dest_file = os.path.join(self.build_lib, ext.lib_dir, product)
            self.copy_file(source_file, dest_file)


if os.getenv("USE_CAMBRICON_CONVERTER"):
    ext_modules = [
        CMakeExtension(
            name="_cambriconLib",
            source_dir="mgeconvert/cambricon_converter",
            lib_dir="lib/cnlib",
            products=["_cambriconLib.so", "cambriconLib.py"],
        )
    ]
    cmdclass = {"build_ext": cmake_build_ext}
else:
    ext_modules = []
    cmdclass = {}


setup(
    name="mgeconvert",
    version=__version__,
    description="MegEngine Converter",
    author="Megvii Engine Team",
    author_email="brain-engine@megvii.com",
    url="https://github.com/MegEngine/mgeconvert",
    packages=find_packages(exclude=["test", "test.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
    setup_requires=["Cython >= 0.20", "numpy"],
    install_requires=["numpy"],
    scripts=glob.glob("util/*"),
)
