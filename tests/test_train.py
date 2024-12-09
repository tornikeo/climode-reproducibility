# Run pytest -s in order to see outputs
import os
from pathlib import Path
import pytest
import subprocess
import sys


def test_training_works():
    sys.path.append('scripts') # Append scripts, since it's not a proper python module
    from train import main
    train(dryrun=True)
    

def test_training_and_eval(tmp_path: Path):
    sys.path.append('scripts') # Append scripts, since it's not a proper python module
    
    from train import main as train
    from evaluate import main as evaluate
    
    model_path = train(dryrun=True)
    evaluate(checkpoint_path=model_path)