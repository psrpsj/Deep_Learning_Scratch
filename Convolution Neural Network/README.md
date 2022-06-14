# Convolution Neural Network (CNN)
  - 활용예졔: 밑바닥부터 시작하는 딥러닝 Chapter 7. CNN

## CNN
  - 기존 신경망과 같이 레고 블록처럼 계층을 조합하여 만듦.
  - 기존 신경망과 다르게 Convolution Layer와 Pooling Layer를 이용함
  - 기존 신경망은 Affine-ReLU의 반복과 마지막의 Affine-Softmax의 조합이었다면, CNN의 경우 Convolution-ReLU-Pooling의 반복 후 마지막 Affine-Softmax 조합을 주로 이용함.
  - Convolution 연산은 기존 Affine 계층과 다르게 3차원 데이터로 입력을 받아 이미지 데이터의 형상을 유지할 수 있는 강점이 있음.
  - 입력에 따른 변화가 적은 풀링 계층 활용.

## SimpleConvNet
  - 구조: conv - relu - pool - affine - relu - affine - softmax
  
