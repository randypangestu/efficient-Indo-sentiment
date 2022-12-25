import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="efficient-indo-sentiment",  # Replace with your own username
    version="0.0.1",
    author="Randy Pang",
    author_email="",
    description="Indonesia sentiment analysis using SetFit, and distillation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/randypangestu/efficient-Indo-sentiment.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache v2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)