from setuptools import setup, find_packages

setup(
    name='edterm',
    version='0.1.4',
    packages=find_packages(),
    description='A terminal-based GROMACS EDR data plotting tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Mattia Felice Palermo',
    author_email='mattiafelice.palermo@gmail.com',
    url='https://github.com/mattiafelice-palermo/edterm',
    install_requires=[
        'pandas',
        'plotext',
        'panedr',
    ],
    entry_points={
        'console_scripts': [
            'edterm = edterm.edterm:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
)
