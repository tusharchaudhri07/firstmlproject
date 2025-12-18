from setuptools import find_packages, setup
setup (
    name = 'my first ML project',
    version= '0.0.1',
    author= ' tushar',
    author_email= ' tc3158494@gmail.com',
    packages = find_packages(),
    install_requires= get_requirment('requirement.txt')
)
