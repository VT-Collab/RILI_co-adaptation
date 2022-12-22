from setuptools import setup

setup(name="gym_rili",
      version="0.1",
      author="Collab",
      packages=["gym_rili", "gym_rili.envs"],
      install_requires = ["gym", "numpy"]
)
