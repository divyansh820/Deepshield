"""
DeepShield — Vercel Serverless Function Entrypoint
Exposes Flask WSGI application for Vercel Python runtime.
"""

import sys
import os

# Ensure the root directory is on the path so app and other modules can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app
