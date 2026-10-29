from setuptools import setup, Extension
import pybind11

# Define the C++ extension module
ext_modules = [
    Extension(
        "localmind_cpp",          # The name of the module we will import in Python
        ["cpp/fast_chunker.cpp"], # The C++ source file
        include_dirs=[pybind11.get_include()],
        language="c++",
        # Force modern C++ standard
        extra_compile_args=["/std:c++20"] if "msvc" in pybind11.get_include() else ["/std:c++20"]
    ),
]

setup(
    name="localmind_cpp",
    ext_modules=ext_modules,
)