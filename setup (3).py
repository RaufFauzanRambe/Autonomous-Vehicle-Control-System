"""
Autonomous Vehicle Control System - Python Package Setup

Configuration for installing AVCS as a Python package with
both C++ extensions and pure Python modules.
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys

# ===========================================================================
# CMake Build Extension
# ===========================================================================

class CMakeBuildExt(build_ext):
    """Custom build_ext command that runs CMake to build C++ extensions."""

    def run(self):
        """Execute the CMake build process."""
        try:
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the C++ extensions. "
                "Please install CMake >= 3.20 and try again."
            )

        for ext in self.extensions:
            if ext.name == 'avcs._cpp_core':
                self.build_cmake_extension(ext)

    def build_cmake_extension(self, ext):
        """Build a single CMake-based extension."""
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DBUILD_PYTHON_BINDINGS=ON',
        ]

        build_args = [
            '--config', 'Release',
            '--', '-j4',
        ]

        build_temp = os.path.join(self.build_temp, 'cpp_core')
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(
            ['cmake', os.path.abspath('.')] + cmake_args,
            cwd=build_temp
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=build_temp
        )


# ===========================================================================
# Package Setup
# ===========================================================================

setup(
    name='autonomous-vehicle-control-system',
    version='0.1.0',
    author='AVCS Team',
    author_email='team@avcs.dev',
    description='Modular autonomous vehicle control system with perception, localization, planning, and control',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/avcs/autonomous-vehicle-control',
    license='Apache-2.0',

    python_requires='>=3.10',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    ext_modules=[
        Extension(
            name='avcs._cpp_core',
            sources=[],  # Built by CMake
        )
    ],

    cmdclass={
        'build_ext': CMakeBuildExt,
    },

    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'opencv-python>=4.8.0',
        'pyyaml>=6.0',
        'matplotlib>=3.7.0',
    ],

    extras_require={
        'perception': [
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'onnxruntime>=1.15.0',
            'albumentations>=1.3.0',
        ],
        'localization': [
            'eigenpy>=3.1.0',
            'open3d>=0.17.0',
        ],
        'planning': [
            'cvxpy>=1.3.0',
            'osqp>=0.6.3',
        ],
        'simulation': [
            'carla>=0.9.15',
        ],
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'isort>=5.12.0',
            'flake8>=6.1.0',
            'mypy>=1.4.0',
            'pre-commit>=3.3.0',
        ],
        'docs': [
            'sphinx>=7.1.0',
            'sphinx-rtd-theme>=1.3.0',
        ],
        'all': [
            'autonomous-vehicle-control-system[perception]',
            'autonomous-vehicle-control-system[localization]',
            'autonomous-vehicle-control-system[planning]',
            'autonomous-vehicle-control-system[simulation]',
        ],
    },

    entry_points={
        'console_scripts': [
            'avcs-run=avcs.cli:main',
            'avcs-sim=avcs.simulation.cli:main',
            'avcs-test=avcs.testing.cli:main',
        ],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Embedded Systems',
        'Topic :: System :: Hardware :: Hardware Drivers',
    ],

    keywords=[
        'autonomous-driving', 'self-driving-car', 'adas',
        'perception', 'localization', 'planning', 'control',
        'sensor-fusion', 'slam', 'v2x', 'ros2',
    ],
)
