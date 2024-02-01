#!/bin/bash

# Check if a video device was provided as an argument
if [ $# -eq 0 ]; then
  echo "Error: Please specify the video device as an argument."
  exit 1
fi

video_device=$1

# List available formats for the specified device
ffmpeg -list_formats all -i "$video_device"

# Play video using specified device and format
ffplay -input_format mjpeg -video_size 1280x720 -i "$video_device"

