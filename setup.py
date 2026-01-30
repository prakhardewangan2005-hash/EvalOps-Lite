"""
Setup script for EvalOps Lite
"""

from setuptools import setup, find_packages
import os

# Read version from package
version = {}
with open("src/__init__.py") as fp:
    exec(fp.read(), version)

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="evalops-lite",
    version=version.get("__version__", "1.0.0"),
    description="Production-grade ML + GenAI evaluation framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Microsoft MLE Intern Candidate",
    author_email="candidate@example.com",
    url="https://github.com/microsoft-mle-intern/evalops-lite",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evalops-train=scripts.train:main",
            "evalops-evaluate=scripts.evaluate:main",
            "evalops-serve=scripts.serve:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine-learning genai evaluation mlops deployment",
    project_urls={
        "Documentation": "https://github.com/microsoft-mle-intern/evalops-lite",
        "Source": "https://github.com/microsoft-mle-intern/evalops-lite",
        "Tracker": "https://github.com/microsoft-mle-intern/evalops-lite/issues",
    },
)
