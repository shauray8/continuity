import os
import sys
from setuptools import Command , find_packages , setup
version  = "0.0.1"

def read_requirements():
    with open('requiremnts-cuda.txt') as req :
        content = req.read()
        requirements  = content.split('\n')
    return requirements

requirements = read_requirements()


setup(
    name = "continuity",
    version=version,
    packages=find_packages(),
    install_requires=requirements,
    author = "Shauray Singh and Tanmay Patil",
    author_email= "",
    description="contains all the evals not available at lm-eval-harness and layer wise inference for bigger models with GGUF support.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url = "https://github.com/shauray8/continuity",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={"console_scripts": ["continuity-cli=continuity.cli.entrypoint.py:main"]},
)