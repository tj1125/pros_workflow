import glob
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.version import hip as _hip_version

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)

requirements = ["torch>=1.4"]

# On a ROCm/HIP PyTorch build the device compiler is hipcc (clang), which does
# not understand NVCC-only flags like -Xfatbin/-compress-all. Emit them only for
# real CUDA builds.
_nvcc_args = ["-O3"] if _hip_version else ["-O3", "-Xfatbin", "-compress-all"]

exec(open(osp.join("pointnet2_ops", "_version.py")).read())

setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": _nvcc_args,
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
