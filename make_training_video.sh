#!/bin/bash

# Generate video from sequence of training stills
RUN_DATE='2024-02-17'
SPECIMEN_LATEN_POINT='16500.28'

# Makes video with frame number annotation
ffmpeg -r 60 -i ./skylines/data/${RUN_DATE}/specimens/${SPECIMEN_LATEN_POINT}_training_sequence/%d.jpg -pix_fmt yuv420p -c:v libx265 -vf "fps=60, drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf: text='%{frame_num}': fontcolor=white: fontsize=60" ./CAA_artifacts/${RUN_DATE}_${SPECIMEN_LATEN_POINT}_training_frame_number.mp4

# Makes video without frame number annotation
ffmpeg -r 60 -i ./skylines/data/${RUN_DATE}/specimens/${SPECIMEN_LATEN_POINT}_training_sequence/%d.jpg -c:v libx265 -vf fps=60 -pix_fmt yuv420p ./CAA_artifacts/${RUN_DATE}_${SPECIMEN_LATEN_POINT}_training_no_frame_number.mp4