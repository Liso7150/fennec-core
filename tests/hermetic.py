"""Shared hermeticity switch for the core suite.

A developer machine may have TokenEconomizer installed for Fennec, and then every
agent turn in every test drives it: it opens index handles inside the temporary
workspace and Windows refuses to delete that directory afterwards, which shows up
as a teardown error in whichever test ran next. The bridge has an explicit off
switch, so the suites turn it off for the whole process. conftest.py covers pytest;
importing this module first covers `unittest discover`, which never reads conftest.
"""
import os

os.environ.setdefault("FENNEC_TOKENECONOMIZER", "0")
