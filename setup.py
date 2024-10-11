import os
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mol2DreaMS",  # Updated package name to reflect your repository
    packages=find_packages(),
    version="1.0.0",
    description="Mol2DreaMS (Deep Representations Empowering the Annotation of Mass Spectra)",
    author="Your Name or Team",  # Update as appropriate
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jozefov/mol2DreaMS",  # Updated repository URL
    install_requires=[
        "torch==2.2.1",
        "pytorch-lightning==2.0.8",
        "torchmetrics==1.3.2",
        "pandas==2.2.3",
        "pyarrow==15.0.2",
        "h5py==3.11.0",
        "rdkit==2023.9.6",
        "umap-learn==0.5.6",
        "seaborn==0.13.2",
        "plotly==5.20.0",
        "ase==3.22.1",
        "wandb==0.16.4",
        "pandarallel==1.6.5",
        "matchms==0.27.0",
        "pyopenms==3.0.0",
        "python-igraph==0.11.6",
        "molplotly==1.1.7",
        "fire==0.6.0",
        "huggingface_hub==0.24.5",
        "msml @ git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main",
        "annoy==1.17.3",
        "click==8.1.7",
        "matplotlib==3.9.2",
        "networkx==3.3",
        "numba==0.57.1",
        "numpy==1.24.4",
        "pynndescent==0.5.13",
        "pyteomics==4.7.4",
        "PyYAML==6.0.2",
        "scikit_learn==1.5.2",
        "scipy==1.14.1",
        "torch_geometric==2.6.1",
        "tqdm==4.66.5",
    ],
    extras_require={
        "dev": [
            "black==24.4.2",
            "pytest==8.3.3",  # Updated from 8.2.1 to match requirements.txt
            "pytest-cov==5.0.0",
            "Cython==3.0.9",
            "SpectralEntropy @ git+https://github.com/YuanyueLi/SpectralEntropy@a1151cfcd9adc66e46f95fb3b06a660e1b0c9b56#egg=SpectralEntropy",
        ],
        "notebooks": [
            "jupyter==1.0.0",
            "ipywidgets==8.1.3",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)