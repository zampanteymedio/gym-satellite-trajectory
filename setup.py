from setuptools import setup


def license():
    with open('LICENSE.txt') as f:
        return f.read()


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='gym-satellite-trajectory',
      version='0.0.1-snapshot',
      python_requires='>=3.6',
      description='Gym environments for spacecraft trajectory optimization',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='gym openai env environment ai reinforcement learning spacecraft satellite trajectory optimization',
      license=license(),
      author_email='carlos.marce@gmail.com',
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
                     'codecov>=2.1.3',
                     ],
      )
