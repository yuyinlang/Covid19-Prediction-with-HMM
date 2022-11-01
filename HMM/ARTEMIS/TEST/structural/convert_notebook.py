"""
@author: Lukas Schöbel 2019.
"""

import os, sys

def import_notebook(name):
    """Converts a provided Jupyter notebook to a Python script

    :param name: Name of the notebook that should be converted 

    """
    
    fname = name + '.ipynb'

    # Check for renaming of the notebook
    if not os.path.exists(fname):
        raise ImportError('You seem to have renamed the notebook. Please use the original name') 
    
    # Check if notebook has already been converted
    if not os.path.exists(name + '.py'):
        command = 'jupyter nbconvert --to script ' + fname
        os.system(command)
        print(f'>> Sucessfully converted {fname} to {name}.py')


if __name__ == "__main__":
    
    if len(sys.argv) - 1: import_notebook(sys.argv[1]) 