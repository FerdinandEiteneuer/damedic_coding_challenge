import os
import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="damedic coding challenge",
    version="1.0",
    author="Ferdinand Eiteneuer",
    author_email="",
    description="Medical recommender system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FerdinandEiteneuer/damedic_coding_challenge",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
)
