#!/bin/bash
set -e; mkdir -p face_det; cd face_det
curl -sSL -o blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite; ls -l
