"""
Run Vaani Hindi Whisper Medium model evaluation
Wrapper around run_comparison.py for single-model execution
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from run_comparison import main

if __name__ == "__main__":
    # Inject model argument to run only Vaani Medium
    if '--models' not in sys.argv:
        sys.argv.extend(['--models', 'vaani_medium'])
    main()
