from setuptools import setup


setup(
    name='matplotlib_logo',
    version='0.1',
    description=(
        'Draws DNA/RNA/Protein sequence logos'
    ),
    author='Matthew Parker',
    packages=[
        'matplotlib_logo',
    ],
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
    ],
)