from setuptools import setup, find_packages

# Read the contents of requirements.txt file
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='bongovaad',
    version='0.3',
    packages=find_packages(),
    install_requires=requirements,
)
