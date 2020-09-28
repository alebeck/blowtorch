from setuptools import setup, find_packages

setup(
    name='aurora',
    version='0.0.2',
    description='Pytorch trainer',
    author='Alexander Becker',
    python_requires='>= 3.6',
    packages=find_packages(),
    install_requires=['colorama', 'PyYAML>=5.1.2', 'halo', 'coolname', 'tqdm']
)
