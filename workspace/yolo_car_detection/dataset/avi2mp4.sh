#!/bin/bash

INPUT_FILE="/data/ephemeral/home/dataset/20260115-11h38m24s_N.avi"
OUTPUT_FILE="/data/ephemeral/home/dataset/20260115-11h38m24s_N.mp4"

~/bin/ffmpeg-*/ffmpeg -i $INPUT_FILE -c:v libx264 -preset slow -crf 23 -an $OUTPUT_FILE