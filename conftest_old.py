import mock_dependencies
import sys
from unittest.mock import MagicMock

# این کد قبل از هرگونه جستجوی تست توسط pytest اجرا می‌شود
# و جلوی لود شدن کتابخانه‌های خراب را می‌گیرد
print("--- Global Mocking for Windows DLL Issues ---")
sys.modules["torch_scatter"] = MagicMock()
sys.modules["torch_cluster"] = MagicMock()
sys.modules["pointops"] = MagicMock()
