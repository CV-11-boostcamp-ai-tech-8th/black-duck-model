python3.10 -m venv py310 --system-site-packages
source py310/bin/activate
pip install --upgrade pip
pip install jupyter ipykernel

apt update
apt upgrade -y

# 설치
pip uninstall -y opencv-python
pip install ultralytics==8.4.7
pip install gdown
pip install python-dotenv
pip install wandb
pip install -U numpy==1.26.0
yolo settings wandb=True

# 그 외 설치들
apt install -y git
apt install -y tmux

# 시스템 설치가 제한된 서버 특성상 libx264 설치가 어려워, ffmpeg를 직접 빌드하는 방법 선택.
# 아래 코드 따라하면 설치됨.
mkdir -p ~/bin
cd ~/bin
apt install -y wget 
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xf ffmpeg-release-amd64-static.tar.xz
~/bin/ffmpeg-*/ffmpeg -version
~/bin/ffmpeg-*/ffmpeg -encoders | grep x264
cd ~

# 그 외 라이브러리들
pip install pandas
pip install omegaconf

# paddleocr
mkdir -p /data/ephemeral/home/{pip_cache,tmp}

export PIP_CACHE_DIR=/data/ephemeral/home/pip_cache
export TMPDIR=/data/ephemeral/home/tmp

python -m pip install --upgrade pip
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install --no-cache-dir paddleocr

pip install --no-cache-dir visualdl imgaug pyclipper lmdb tqdm shapely

pip install -U filelock
pip install -U huggingface_hub
pip install -U albumentations
pip install -U "lap>=0.5.12"
pip install google-cloud-storage
pip install timm