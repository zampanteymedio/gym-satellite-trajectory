from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='gym-satellite-trajectory',
      version='0.0.1-snapshot',
      python_requires='>=3.6',
      packages=['gym_satellite_trajectory'],
      scripts=[],
      description='Gym environments for spacecraft trajectory optimization',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/zampanteymedio/gym-satellite-trajectory',
      keywords='gym openai env environment ai reinforcement learning spacecraft satellite trajectory optimization',
      license='Apache License 2.0',
      author='Carlos M. Casas Cuadrado',
      author_email='carlos.marce@gmail.com',
      install_requires=['gym>=0.17.1',
                        'setuptools>=41.4.0',
                        'numpy>=1.16.5',
                        'matplotlib>=3.1.1',
                        'orekit>=10.0',
                        ],
      extras_require={'test': ['pytest>=5.2.1',
                               'pytest-cov>=2.9.0',
                               'coverage>=5.1',
                               'pytest-ordering>=0.6',
                               'codecov>=2.1.3',
                               ],
                      }
      )
