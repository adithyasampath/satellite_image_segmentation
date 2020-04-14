NOTE: If CUDA version 8 or below change tensorflow-gpu version to 1.4.0. Tensorflow 1.12.0 supports only CUDA 9 and above

1. Install requirements

RUN: pip install -r requirements.txt

2. Training

RUN: python3 resnet50_train.py

NOTE: After Training, the test images are saved in "images" folder and the trained model is saved in "weights" folder
(Pre-Trained weight: https://drive.google.com/file/d/1v4atp_ErftCHbyZiRi1bVPBRsY3w7Nic/view)

3. Flask test server

RUN: python3 main.py <PORT>

where, <PORT> (optional argument, DEFAULT VALUE = 5000) -> the port on which you want the flask server to run.

4. Front end

NOTE: First start flask server

(On Browser) RUN: http://127.0.0.1:<PORT> (DEFAULT: http://127.0.0.1:5000) 

Test on images saved in "images" folder.
