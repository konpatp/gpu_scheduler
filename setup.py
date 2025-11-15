from setuptools import setup

setup(
    name="gpu_scheduler",
    version="0.1.0",
    description="A Python package for managing GPU allocation and scheduling in multi-GPU environments",
    author="konpatp",
    packages=["gpu_scheduler"],
    package_dir={"gpu_scheduler": "."},
    python_requires=">=3.6",
)

