mkdir -p ~/bin
cd ~/bin

apt install wget 

wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xf ffmpeg-release-amd64-static.tar.xz

~/bin/ffmpeg-*/ffmpeg -version
~/bin/ffmpeg-*/ffmpeg -encoders | grep x264


INPUT_FILE="/data/ephemeral/home/jsw/pro-cv-finalproject-cv-11/runs/detect/track/yolo26l_v2/급정거_2/급정거_2.avi"
OUTPUT_FILE="/data/ephemeral/home/jsw/pro-cv-finalproject-cv-11/runs/detect/track/yolo26l_v2/급정거_2/급정거_2.mp4"

~/bin/ffmpeg-*/ffmpeg -i $INPUT_FILE -c:v libx264 -preset slow -crf 23 -an $OUTPUT_FILE