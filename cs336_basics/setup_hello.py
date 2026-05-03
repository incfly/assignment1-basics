from setuptools import Extension, setup


setup(
    name="cs336_basics_hello",
    packages=["cs336_basics"],
    ext_modules=[
        Extension(
            "cs336_basics._hello",
            sources=["cs336_basics/hello_native.cpp"],
            language="c++",
        )
    ],
)
