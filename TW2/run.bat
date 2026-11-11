@echo off
REM Example run for the supplied T10I4D100K transaction dataset.
REM The support and confidence values are examples and can be changed.
python apriori.py --dataset "..\T10I4D100K.dat" --support 1000 --confidence 0.6 --output-dir results
