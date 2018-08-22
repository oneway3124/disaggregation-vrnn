#!/bin/bash
python autoEncoder_tf_App.py datasets/ukdale.h5 200 100 10000 20 200 0
python autoEncoder_tf_App.py datasets/ukdale.h5 300 100 10000 30 300 1
python autoEncoder_tf_App.py datasets/ukdale.h5 1100 100 10000 100 1100 2
python autoEncoder_tf_App.py datasets/ukdale.h5 1600 100 10000 160 1600 3
python autoEncoder_tf_App.py datasets/ukdale.h5 600 100 10000 60 600 4