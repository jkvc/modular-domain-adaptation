from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mda",
    version="0.1",
    description="Modular Domain Adaptation is a framework to produce and consume models while addressing the domain misalignment issue.",
    py_modules=["mda"],
    package_dir={"mda": "mda"},
    install_requires=[
        "pandas",
        "torchvision",
        "nltk",
        "matplotlib",
        "transformers",
        "sklearn",
        "tensorboard",
        "hydra-core",
        "pydantic",
    ],
    extras_require={
        "dev": [
            "pytest >= 3.8",
            "check-manifest",
            "twine",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Junshen K Chen",
    author_email="kevinehc@gmail.com",
    url="https://github.com/jkvc/modular-domain-adaptation",
)
