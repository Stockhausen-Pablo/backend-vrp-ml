from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    usage_license = f.read()

setup(
    name='backend-vrp-ml',
    version='1.0.0',
    description='Different machine learning approaches to solve CVRP',
    long_description=readme,
    author='Pablo Stockhausen',
    author_email='stockhausen017@gmail.com',
    url='https://gitlab.com/P-Stockhausen/backend-vrp-ml',
    license=usage_license,
    packages=find_packages(exclude=('tests', 'docs'))
    # packages=find_namespace_packages()
)
