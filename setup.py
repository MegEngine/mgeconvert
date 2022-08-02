import os
import re
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as _install

with open(os.path.join("./mgeconvert", "version.py")) as f:
    __version_py__ = f.read()

__version__ = re.search(r"__version__ = \"(.*)\"", __version_py__).group(1)


targets = set()
tfversion = None
IS_VENV = (
    hasattr(sys, "real_prefix")
    or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    or os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    or (("Continuum Analytics" in sys.version) or ("Anaconda" in sys.version))
)


def write_init(targets, tflite_schema_version=None):
    with open("mgeconvert/__init__.py", "w") as init_file:
        [
            init_file.write("from .converters.mge_to_%s import mge_to_%s\n" % (i, i))
            for i in targets
        ]
        [
            init_file.write(
                "from .converters.tm_to_%s import tracedmodule_to_%s\n" % (i, i)
            )
            for i in targets
        ]
        if "onnx" in targets:
            init_file.write("from .converters.onnx_to_mge import onnx_to_mge\n")
            init_file.write("from .converters.onnx_to_tm import onnx_to_tracedmodule\n")
        if not tflite_schema_version:
            tflite_schema_version = "r2.3"
        init_file.write(f'tflite_schema_version="{tflite_schema_version}"\n')


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
        if self.targets.find("all") >= 0:
            targets.update(set(options))
        elif self.targets is None:
            pass
        else:
            targets.update(set(i for i in options if self.targets.find(i) >= 0))

        global tfversion
        if self.tfversion:
            tfversion = self.tfversion
        write_init(targets, tfversion)

        if "tflite" in targets:
            tflite_path = os.path.join(
                os.path.dirname(__file__), "mgeconvert", "backend", "ir_to_tflite"
            )
            if (
                not os.path.exists(
                    os.path.join(tflite_path, "pyflexbuffers", "bin", "flatc")
                )
                or not os.path.exists(
                    os.path.join(tflite_path, "pyflexbuffers", "include", "flatbuffers")
                )
                or not os.path.exists(os.path.join(tflite_path, "pyflexbuffers", "lib"))
            ):
                ret = os.system(f"{tflite_path}/build_flatbuffer.sh")
                if ret:
                    raise RuntimeError("build flatbuffer failed!")

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
        if ext.script:
            if ext.name == "tflite" and tfversion is not None:
                subprocess.check_call(
                    [ext.script, str(IS_VENV), sys.executable, tfversion]
                )
            else:
                subprocess.check_call([ext.script, str(IS_VENV), sys.executable])
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

if __name__ == "__main__":
    install_requires = ["numpy", "tqdm"]
    requires_mapping = {
        "onnx": ["onnx>=1.8.0", "onnx-simplifier>=0.3.6", "protobuf",],
        "caffe": ["protobuf>=3.11.1"],
        "tflite": ["flatbuffers==1.12.0", "pybind11==2.6.2"],
        "all": [
            "onnx>=1.8.0",
            "onnx-simplifier>=0.3.6",
            "protobuf>=3.11.1",
            "flatbuffers==1.12.0",
        ],
    }
    pkg_name = "mgeconvert"
    if len(sys.argv) >= 2:
        if sys.argv[1] == "install":
            assert (
                len(sys.argv) > 2
            ), 'use "--targets=[eg, tflite,caffe,onnx,all]" to indicate converters to install'
            targets_cache = set()
            for v in sys.argv[2:]:
                targets_args = re.findall(
                    "--targets\s*=\s*['\"]?([\w,\s]+)['\"]?\s*", str(v)
                )
                tfversion_args = re.findall("--tfversion\s*=\s*([\w\.\d]+)\s*", str(v))

                if targets_args:
                    targets_args = (",".join(targets_args)).replace(" ", "").split(",")
                    for opt in targets_args:
                        assert opt in requires_mapping, f"opt={opt}, {sys.argv}"
                        if opt not in targets_cache:
                            install_requires.extend(requires_mapping[opt])
                            targets_cache.add(opt)

                if tfversion_args:
                    tfversion = tfversion_args[0]

                if "targets" in v or "tfversion" in v:
                    assert targets_args or tfversion_args, (
                        f"cant parse option={v}."
                        "please use '--targets=[eg, tflite,caffe,onnx,all]' to indicate converters to install "
                        "and `--tfversion=[eg, r2.6]` to indicate the tflite version."
                    )

    setup(
        name=pkg_name,
        version=__version__,
        description="MegEngine Converter",
        author="Megvii Engine Team",
        author_email="brain-engine@megvii.com",
        url="https://github.com/MegEngine/mgeconvert",
        packages=find_packages(exclude=["test", "test.*"]),
        ext_modules=ext_modules,
        cmdclass={"install": install, "build_ext": build_ext},
        include_package_data=True,
        install_requires=install_requires,
        scripts=["bin/convert"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )
