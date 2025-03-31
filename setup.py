import setuptools

setuptools.setup(
    name="final_nn",
    version="0.1.0",
    author="Justin Sim",
    author_email="justin.sim@ucsf.edu",
    description="Final nn project, autoencoder and classifier",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/justinsim12/final-nn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pytest",
        "logomaker"
    ],
)