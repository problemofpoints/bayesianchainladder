"""Add this directory to sys.path so test files can import the flat-layout modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
