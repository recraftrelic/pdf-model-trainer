from setuptools import setup

setup(
    name='pdfmodeltrainer',
    version='0.0.1',
    packages=[''],
    url='',
    license='MIT',
    author='Manoj Singh Negi',
    author_email='justanothermanoj@gmail.com',
    description='A package for training name model from pdfs',
    install_requires=[
        'chardet==3.0.4',
        'cryptography==3.2.1',
        'pdfminer.six==20201018',
        'setuptools~=50.3.2',
        'plac~=1.1.3',
        'spacy~=2.3.4',
        'docx2txt~=0.8'
    ]
)
