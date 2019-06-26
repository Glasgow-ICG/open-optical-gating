
#!/bin/bash

# A simple bash script to capture video from the raspberry pi's picam

width=1280
height=720
time="$2"
fps="$3"
filename="RPi-vidcap-$1-$(date +%F-%H:%M).h264"

raspivid -o ${filename} -w ${width} -h ${height} -t ${time} -fps ${fps} -rot 180
