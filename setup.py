"""Setup script for Alpha Research Trading System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="alpha-research",
    version="1.0.0",
    author="Alpha Research Team",
    description="Systematic quantitative trading system with Core factors and LLM satellites",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MauveAndromeda/Alpha-Research",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "mypy>=1.8.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alpha-run-daily=scripts.run_daily:main",
            "alpha-backtest=scripts.backtest:main",
        ],
    },
)
