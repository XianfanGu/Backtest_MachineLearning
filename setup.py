from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import toollib.Download.download_csv as downloader
import toollib.Download.config as config

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        downloader.download()
        config.create_config()
        install.run(self)
setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='toollib',
    # *strongly* suggested for sharing
    version='0.1',
    url='https://github.com/XianfanGu/toollib',
    author='Xianfan Gu',
    author_email='xianfang@asu.edu',
    # Needed to actually package something
    packages=['toollib'],
    cmdclass={
            'develop': PostDevelopCommand,
            'install': PostInstallCommand,
        },

    # The license can be anything you like
    license='MIT',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
