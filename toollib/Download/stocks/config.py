import subprocess
import shutil
import os
def create_config():
    cwd = os.getcwd()
    with open(cwd+"/src/extension.py","r") as f:
        lines = f.readlines()

    with open(cwd+"/src/extension.py","r+") as f1:
        f1.seek(0)
        lines[11] = '\t\''+cwd+"/csv"+'\',\n'
        for line in lines:
            f1.write(line)
        f1.truncate()
    path = os.path.expanduser('~/.zipline/extension.py')
    shutil.copyfile('src/extension.py', path)
    subprocess.call(["zipline", "ingest", "-b", "custom-na-csvdir-bundle"])
