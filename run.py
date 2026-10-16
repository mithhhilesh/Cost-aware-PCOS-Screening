"""Launch the local PCOS frontend using this project's single artifacts folder."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
backend = root / 'pcos-insight-frontend' / 'backend'
sys.path.insert(0, str(backend))
sys.argv = [str(backend / 'server.py'), '--artifacts', str(root / 'artifacts'), *sys.argv[1:]]
runpy.run_path(backend / 'server.py', run_name='__main__')
