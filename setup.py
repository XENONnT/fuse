import setuptools 

with open("README.md", "r") as file:
    readme = file.read()

setuptools.setup(
    name='XeSim',
    version='0.0.0',
    description='XENON Simulation Refactor',
    author='For now: Just Me',
    url='https://github.com/XENONnT/XeSim',
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=['pytest-runner'],
    python_requires=">=3.6",

    packages=setuptools.find_packages(),
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics'],
    zip_safe=False)