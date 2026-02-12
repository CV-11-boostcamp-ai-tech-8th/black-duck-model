# black-duck-model
- yolo26x 학습 및 실험을 위해 사용한 workspace

# files explanation
## main scripts
- track.py: yolo tracking script
- train.py: yolo train script
- inference.py: predict using our weight
- temp/
    - val.py: (pending)

## outputs
- runs/
    - detect/
        - cv-11-final/: train results & weight & log files
        - track: results of tracking
        - inference: results of inference

## utils
- utils/
    - concat.py: concat result images
    - check_best_epoch.py: check which is the best epoch by refering 'results.csv'
- scripts/: shell script files for setting the server or our virtual environment

## configs
- yolo/
    - botsort.yaml
    - bytetrack.yaml
    - vehicle_dataset.yaml

## dataset
- dataset/: scripts for dataset
    - avi2mp4.sh: convert avi container file to mp4 container file with H.264 codec
    - lib264_install.sh: how to install lib264
    - custom_voc2yolo.py: convert voc annotation to yolo labels