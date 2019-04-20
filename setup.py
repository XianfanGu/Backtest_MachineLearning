from setuptools import setup, find_packages

setup(
    # Whatever arguments you need/want
)

from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='toollib',
    url='https://github.com/XianfanGu/toollib',
    author='Xianfan Gu',
    author_email='xianfang@asu.edu',
    # Needed to actually package something
    packages=['toollib'],
    # Needed for dependencies
    install_requires=['talib','pandas','zipline','sklearn'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
