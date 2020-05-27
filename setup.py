from setuptools import setup

setup(name='gym-satellite-trajectory',
      version='0.0.1-snapshot',
      install_requires=['gym>=0.17.1',
                        'setuptools>=41.4.0',
                        'numpy>=1.16.5',
                        'matplotlib>=3.1.1',
                        'orekit>=10.0',
                        ],
      tests_require=['pytest>=5.2.1',
                     'pytest-cov>=2.9.0',
                     'coverage>=5.1',
                     'pytest-ordering>=0.6',
                     ],
      )
