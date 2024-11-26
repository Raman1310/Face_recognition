from setuptools import setup, find_packages

setup(
    name="my_project",  # Replace with your project name
    version="1.0.0",
    author="RP",
    author_email="rpdeveloper29@gmail.com",
    description="Python module",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_project",  # Replace with your repository URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)