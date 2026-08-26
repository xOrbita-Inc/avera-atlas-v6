"""
services/validity/conftest.py

Adds this service's own directory to sys.path so its tests can bare-import
routing.py / wrapper.py (e.g. `from routing import ...`), matching the
pattern used by the other services in this repo (planner, tracker, etc.).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
