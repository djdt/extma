from setuptools import setup


setup(
    name="extma",
    version="0.1.0",
    description="Extractor for microarray or spotted tissue LA-ICP-MS data.",
    packages=["extma"],
    license="LGPL",
    author="djdt",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "pew@git+https://github.com/djdt/pew#egg=pew-0.4.4",
    ],
    entry_points={"console_scripts": ["extma=extma.__main__:main"]},
)
