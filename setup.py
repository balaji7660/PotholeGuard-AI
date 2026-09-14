"""
setup.py — Pothole Avoidance Project
Editable install: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="pothole_avoidance",
    version="1.0.0",
    description="Robust Pothole Avoidance in Autonomous Driving using "
                "Multi-Task Transformer Perception and Ensemble RL",
    author="Research Team",
    python_requires=">=3.12",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.12",
        "einops>=0.7.0",
        "stable-baselines3>=2.2.1",
        "sb3-contrib>=2.2.1",
        "gymnasium>=0.29.1",
        "opencv-python>=4.9.0.80",
        "Pillow>=10.2.0",
        "albumentations>=1.3.1",
        "numpy>=1.26.4",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "PyYAML>=6.0.1",
        "omegaconf>=2.3.0",
        "tensorboard>=2.16.2",
        "tqdm>=4.66.2",
        "rich>=13.7.1",
        "matplotlib>=3.8.3",
    ],
)
