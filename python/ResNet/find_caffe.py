import os, sys
try:
    import caffe
except ImportError:
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, "../../python"))
    import caffe
