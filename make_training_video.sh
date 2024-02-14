#!/bin/bash

# Generate video from sequence of training stills

RUN_DATE='2024-02-11'

ffmpeg -r 24 -i ./skylines/data/specimens/${RUN_DATE}/training_sequence/%d.jpg -c:v libx265 -vf fps=60 -pix_fmt yuv420p ./CAA_artifacts/${RUN_DATE}_training.mp4