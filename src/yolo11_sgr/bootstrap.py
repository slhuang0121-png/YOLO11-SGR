"""Select the retained implementation without changing the user's installed packages."""
from pathlib import Path
import sys
def activate():
    root=Path(__file__).resolve().parent
    for rel in ["_vendor","_frozen/algorithm","_frozen/person","_frozen/analysis","_frozen/lab_modern"]:
        path=str(root/rel)
        if path not in sys.path:sys.path.insert(0,path)
