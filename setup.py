from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='gym-satellite-trajectory',
      version='0.2+snapshot',
      python_requires='>=3.6',
      packages=find_packages(where='.', exclude=('test',)),
      package_data={'gym_satellite_trajectory': ['data/orekit-data.zip']},
      scripts=[],
      description='Gym environments for spacecraft trajectory optimization',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/zampanteymedio/gym-satellite-trajectory',
      keywords='gym openai env environment ai reinforcement learning spacecraft satellite trajectory optimization',
      license='Apache License 2.0',
      author='Carlos M. Casas Cuadrado',
      author_email='carlos.marce@gmail.com',
      install_requires=['gym>=0.17.3',
                        'matplotlib>=3.3.3',
                        'numpy>=1.19.4',
                        'orekit>=10.3',
                        'setuptools>=51.0.0',
                        ],
      extras_require={'test': ['codecov>=2.1.10',
                               'coverage>=5.3',
                               'pytest>=6.1.2',
                               'pytest-cov>=2.10.0',
                               'pytest-ordering>=0.6',
                               'stable-baselines3>=1.0',
                               ],
                      },
      )
