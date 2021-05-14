from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='vrp-rl',
    version='0.1.0',
    description='Reinforcment Learning by using a Markov Decision process to solve VRP',
    long_description=readme,
    author='Pablo Stockhausen',
    author_email='stockhap@th-brandenbug.de',
    url='https://github.com/Stockhausen-Pablo/vrp-rl',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)