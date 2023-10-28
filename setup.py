from setuptools import setup


setup(
    name="extma",
    version="0.1.3",
    description="Extractor for microarray or spotted tissue LA-ICP-MS data.",
    packages=["extma"],
    license="LGPL",
    author="djdt",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "pewlib>=0.8.3",
    ],
    entry_points={"console_scripts": ["extma=extma.__main__:main"]},
)
