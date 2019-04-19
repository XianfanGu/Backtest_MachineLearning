from setuptools import setup, find_packages

setup(
    # Whatever arguments you need/want
)

from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Backtest_MachineLearning',
    url='https://github.com/jladan/package_demo',
    author='Xianfan Gu',
    author_email='xianfang@asu.edu',
    scripts=['bin/download'],
    # Needed to actually package something
    packages=['Backtest_MachineLearning'],
    # Needed for dependencies
    install_requires=['talib','pandas','numpy','zipline','sklearn','math'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
