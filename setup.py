from setuptools import setup, find_packages

setup(
    name="osrs-flipper-ai",
    version="0.1.0",
    author="Matthew Calder",
    description="Automated OSRS Grand Exchange flipping optimizer powered by AI.",
    packages=find_packages(include=["osrs_flipper_ai", "osrs_flipper_ai.*"]),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.25",
        "requests>=2.31",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "matplotlib>=3.7",
        "pyarrow>=15.0",
    ],
    extras_require={
        "dev": ["black", "flake8", "pytest"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
