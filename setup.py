import os
import re
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as _install

with open(os.path.join("./mgeconvert", "version.py")) as f:
    __version_py__ = f.read()

__version__ = re.search(r"__version__ = \"(.*)\"", __version_py__).group(1)


targets = []
tfversion = None


class install(_install):
    user_options = _install.user_options + [
        ("targets=", None, "<description for this custom option>"),
        ("tfversion=", None, "the version of tflite schema"),
    ]

    def initialize_options(self):
        _install.initialize_options(self)
        self.targets = None
        self.tfversion = None

    def finalize_options(self):
        _install.finalize_options(self)

    def run(self):
        options = ["caffe", "onnx", "tflite"]
        if self.targets == "all":
            targets.extend(options)
        else:
            targets.extend(i for i in options if self.targets.find(i) >= 0)

        with open("mgeconvert/__init__.py", "a+") as init_file:
            [
                init_file.write(
                    "from .converters.mge_to_%s import mge_to_%s\n" % (i, i)
                )
                for i in targets
            ]
            [
                init_file.write(
                    "from .converters.tm_to_%s import tracedmodule_to_%s\n" % (i, i)
                )
                for i in targets
            ]

        global tfversion
        tfversion = self.tfversion

        _install.run(self)


class build_ext(_build_ext):
    def run(self):
        for target in targets:
            self.build_all(self.find_extension(target))

    def find_extension(self, name):
        for ext in self.extensions:
            if ext.name == name:
                return ext
        raise TypeError("can not build %s" % name)

    def build_all(self, ext):
        if ext.name == "tflite" and tfversion is not None:
            subprocess.check_call([ext.script, tfversion])
        else:
            subprocess.check_call(ext.script)
        if ext.artifacts is not None:
            self.copy_tree(ext.artifacts, os.path.join(self.build_lib, ext.artifacts))


class BuildExtension(Extension):
    def __init__(self, name, script, artifacts=None):
        super().__init__(name, sources=[])
        self.script = script
        self.artifacts = artifacts


ext_modules = [
    BuildExtension(
        name="caffe",
        script="mgeconvert/backend/ir_to_caffe/init.sh",
        artifacts="mgeconvert/backend/ir_to_caffe/caffe_pb",
    ),
    BuildExtension(name="onnx", script="mgeconvert/backend/ir_to_onnx/init.sh"),
    BuildExtension(
        name="tflite",
        script="mgeconvert/backend/ir_to_tflite/init.sh",
        artifacts="mgeconvert/backend/ir_to_tflite/",
    ),
]

setup(
    name="mgeconvert",
    version=__version__,
    description="MegEngine Converter",
    author="Megvii Engine Team",
    author_email="brain-engine@megvii.com",
    url="https://github.com/MegEngine/mgeconvert",
    packages=find_packages(exclude=["test", "test.*"]),
    ext_modules=ext_modules,
    cmdclass={"install": install, "build_ext": build_ext},
    include_package_data=True,
    install_requires=["numpy"],
    scripts=["bin/convert"],
)
