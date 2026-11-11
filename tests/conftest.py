"""Keep the core suite hermetic under pytest.

`unittest discover` never reads this file, so every test module imports the
shared guard in hermetic.py itself; this covers the pytest path.
"""
import hermetic  # noqa: F401
