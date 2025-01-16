from setuptools import setup, find_packages

setup(
    name="personabench",                   
    version="0.1.0",                     
    description="Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data",
    author="Juntao Tan",
    author_email="juntao.tan@salesforce.com",
    packages=find_packages(),            
    python_requires=">=3.11",             
)