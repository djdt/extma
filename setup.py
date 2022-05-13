from setuptools import setup


setup(
    name="extma",
    version="0.1.2",
    description="Extractor for microarray or spotted tissue LA-ICP-MS data.",
    packages=["extma"],
    license="LGPL",
    author="djdt",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "pewlib",
    ],
    entry_points={"console_scripts": ["extma=extma.__main__:main"]},
)
