#!/bin/bash

# Generate video from sequence of training stills

RUN_DATE='2024-02-11'

# Makes video with frame number annotation
#ffmpeg -r 60 -i ./skylines/data/specimens/${RUN_DATE}/training_sequence/%d.jpg -pix_fmt yuv420p -c:v libx265 -vf "fps=60, drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf: text='%{frame_num}': fontcolor=white: fontsize=60" ./CAA_artifacts/${RUN_DATE}_training.mp4

# Makes video without frame number annotation
ffmpeg -r 60 -i ./skylines/data/specimens/${RUN_DATE}/training_sequence/%d.jpg -c:v libx265 -vf fps=60 -pix_fmt yuv420p ./CAA_artifacts/${RUN_DATE}_training.mp4