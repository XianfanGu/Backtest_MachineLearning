from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import toollib.Download.download_csv as downloader
import toollib.Download.config as config
import sys
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        if(query_yes_no("Do you try creating a new data bundle?")):
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
    packages=['toollib','toollib.pair_trading','toollib.Data','toollib.algorithm','toollib.algorithm.MachineLearning','toollib.Download','toollib.Data.TA',],
    cmdclass={
            'develop': PostDevelopCommand,
            'install': PostInstallCommand,
        },

    # The license can be anything you like
    license='MIT',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
