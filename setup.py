from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''This function returns the list of requirements'''
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
    requirements = [req.strip() for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="my_package",
    version="0.0.1",
    author="Chandan Kumar Yaduvanshi",
    author_email="gchandan16@gmail.com",
    description="A sample Python package",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
