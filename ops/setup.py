import setuptools

with open("README.rst", "r") as readme:
    long_description = readme.read()

setuptools.setup(
    name='mantrap',
    version='1.0.0',
    url='https://github.com/simon-schaefer/mantrap.git',
    author='Simon Schaefer',
    author_email='sischaef@ethz.ch',
    description='Minimal interfering Interactive Risk-aware Planning',
    long_description=long_description,
    long_description_content_type="text/rst",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
