# file: tests/unit/conftest.py

import sys
from unittest.mock import MagicMock

sys.modules["torch_scatter"] = MagicMock()
sys.modules["torch_cluster"] = MagicMock()
sys.modules["pointops"] = MagicMock()
sys.modules["spconv"] = MagicMock()
sys.modules["spconv.pytorch"] = MagicMock()