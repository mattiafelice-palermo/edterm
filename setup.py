from setuptools import find_packages, setup

try:
    from setuptools_rust import Binding, RustExtension

    rust_extensions = [
        RustExtension(
            'edterm_rust_ext',
            path='rust_ext/Cargo.toml',
            binding=Binding.PyO3,
            debug=False,
            optional=True,
        )
    ]
except Exception:
    rust_extensions = []

setup(
    name='edterm',
    version='0.1.5',
    packages=find_packages(),
    description='A terminal-based GROMACS EDR data plotting tool',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Mattia Felice Palermo',
    author_email='mattiafelice.palermo@gmail.com',
    url='https://github.com/mattiafelice-palermo/edterm',
    install_requires=[
        'numpy',
        'plotext',
    ],
    rust_extensions=rust_extensions,
    include_package_data=True,
    zip_safe=False,
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Programming Language :: Rust',
    ],
    python_requires='>=3.8',
)
