# 로컬
tar -czf road_dataset.tar.gz road_dataset/

# 서버
# original: ai-hub에서 다운받은 그대로
# flatten: python 스크립트로 하나의 폴더 안에 images 및 annotations이 들어가도록 flattening 함

# 각 용량 12GB 전후, 압축 해제까지 고려해 여유 공간 25GB 필요

# original_road_dataset_bb.tar.gz
gdown https://drive.google.com/uc?id=1HPFcz79BqzrVGhmS1EqZPndoCOkCKWE3

# flatten_road_dataset_bb.tar.gz
gdown https://drive.google.com/uc?id=1ixPKD3-dq-Pn7Jfdp9zqViMZ4TSXFhtB

# original_road_dataset_sl.tar.gz
gdown https://drive.google.com/uc?id=1b4mSjFPysJdFljI-Cx_LGOWm7mlHubZ7

# flatten_road_dataset_sl.tar.gz
gdown https://drive.google.com/uc?id=1oBGsJHrLgW4S_FZCEfLYpXJiyEQE8brv

# flatten_car_road_dataset_bb.tar.gz
gdown https://drive.google.com/uc?id=1gsTzvO6kF2vuCbhNKh8Pv9QVwFXs0zyC

# 압축 해제
tar -xzf flatten_car_road_dataset_bb.tar.gz