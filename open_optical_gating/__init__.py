# Things get very fiddly if trying to allow importing *and*
# allow running of e.g. file_optical_gater via "python -m".
# While this doesn't feel like the ideal solution,
# this "if" test seems to be the only viable solution that
# doesn't involve specific different files to call via python -m.
# See https://stackoverflow.com/questions/43393764/python-3-6-project-structure-leads-to-runtimewarning
import sys
if not '-m' in sys.argv:
    from . import cli
