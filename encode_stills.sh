#!/bin/bash

ffmpeg -r 24 -i ./skylines/data/specimens/2022-03-24/training_sequence/%d.jpg -c:v libx265 -vf fps=24 -pix_fmt yuv420p ./training.mp4