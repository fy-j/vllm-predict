# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make the harness importable when pytest runs from the repository root.

Running from the root matters: it is what puts the repository's own `vllm` on
the path instead of any separately installed build.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
