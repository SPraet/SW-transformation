# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:13:25 2019

@author: SPraet
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SW-transformation",
    version="0.0.1",
    author="Stiene Praet",
    author_email="stiene.praet@uantwerp.be",
    description="A fast classifier for binary node classification in bipartite graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SPraet/SW-transformation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)