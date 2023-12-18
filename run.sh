#!/bin/bash

for ((i = 0; i < 10; i++)); do
    python3 homogeneous/pre_gat.py
    python3 homogeneous/pre_ft.py
done