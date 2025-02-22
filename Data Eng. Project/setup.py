from setuptools import find_packages, setup

with open("pypi/README.txt", "r") as f:
    long_description = f.read()

setup(
    name="astrophil",
    version="0.8",
    description="data cleaner",
    package_dir={"": "pypi"},
    packages=find_packages(where="pypi"),
    long_description="README.txt",
    long_description_content_type="text",
    url="",
    author="Burak Nafi Girgin",
    author_email="bnafigrgn@gmail.com",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.17",
        "Operating System :: OS Independent"
    ],
    python_requires = " < 3.11"
)