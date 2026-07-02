import sys
import os

# 1. اضافه کردن مسیر ماک‌ها به sys.path
current_dir = os.path.abspath(os.path.dirname(__file__))
mock_path = os.path.join(current_dir, "tests", "integration")
if mock_path not in sys.path:
    sys.path.insert(0, mock_path)

# 2. لود کردن ماک‌ها قبل از pytest
try:
    import mock_dependencies
    print("[INIT] Dependencies mocked successfully before Pytest starts.")
except ImportError:
    print("[ERROR] mock_dependencies.py not found in:", mock_path)
    sys.exit(1)

# 3. اجرای Pytest
import pytest

if __name__ == "__main__":
    # آرگومان‌های پایه برای لاگ‌های خواناتر
    pytest_args = ["-v", "-s"]
    
    # بررسی می‌کنیم آیا کاربر نام فایلی را در ترمینال وارد کرده است یا خیر
    user_args = sys.argv[1:]
    
    if len(user_args) > 0:
        # اگر کاربر نام فایل خاصی داد، مسیر کامل آن را می‌سازیم
        resolved_args = []
        for arg in user_args:
            if arg.endswith('.py') and not os.path.isabs(arg):
                resolved_args.append(os.path.join(mock_path, arg))
            else:
                resolved_args.append(arg)
        pytest_args.extend(resolved_args)
    else:
        # اگر هیچ آرگومانی داده نشد، کل پوشه integration را اجرا کن
        pytest_args.append(mock_path)

    print(f"[RUN] Executing pytest with arguments: {pytest_args}")
    sys.exit(pytest.main(pytest_args))
