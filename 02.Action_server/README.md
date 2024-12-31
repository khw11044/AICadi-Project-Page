
### 모델 다운로드 - 1

models.zip 파일을 풀어서 models폴더를 현재 README.md 파일과 같은 위치에 놓아주세요

[models 다운로드](https://drive.google.com/file/d/1WlYyPgR6ycbPJcOAXCXKtXgxVgEc8IFg/view?usp=sharing)

### 모델 다운로드 - 2

demo 폴더에 lib 폴더에 압축을 푼 checkpoint 폴더를 넣어준다. 

[checkpoint 다운로드](https://drive.google.com/file/d/1ObxWaCdjIXPRb9dCJJGw9A_nxEbUbiaa/view?usp=sharing)


### 데이터수집 

00.01_collect_dataset.py 

행동을 2D pose 시퀀스로 저장하여 데이터셋을 구축합니다.


### 모델 훈련 

00.02_train_kpc_lifting.py 

모델을 훈련시킵니다.

### 테스트 

00.03_test_kpcl.py

모델을 테스트합니다.

### 코드가 있는 컴퓨터 카메라로 Action Recognition하고 로봇으로 UDP Socket 통신 

01.local_action_event_hpe.py

### 라즈베리파이에서 flask HTTP통신으로 영상 수신받아 Action Recognition을 수행하고 로봇으로 UDP Socket 통신 

02.stream_action_event_hpe.py

### 최종 코드 

03.final.py