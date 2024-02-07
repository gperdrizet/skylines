#!/bin/bash

ffmpeg -r 24 -i ./gan_output/frame%07d.jpg -c:v libx265 -vf fps=24 -pix_fmt yuv420p ./training-3072x3072.mp4
