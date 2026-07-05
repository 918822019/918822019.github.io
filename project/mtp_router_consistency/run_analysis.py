#!/usr/bin/env python3
"""Entry point for MTP Router Consistency Analysis.

Usage: python run_analysis.py

Adds the core/ directory to sys.path and runs the main pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "core"))
from main import main

main()
