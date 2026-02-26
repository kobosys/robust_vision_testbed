# Adversarial Patch (Proxy Injection) – Quick Start

이 폴더는 "패치 생성/학습(오프라인)" 관련 스크립트를 모아둡니다.

현재 프로젝트 흐름(권장):
1) 패치 생성/학습 → patches/ 아래에 저장
2) realsense_patch_proxy.py로 프레임에 패치 합성 → /dev/video10 송출
3) 의뢰자 모델(또는 YOLO demo)이 /dev/video10을 카메라로 입력

---

## 폴더 구조(권장)

ROBUST_YOLO/
- data
  - patch_train_images/
- realsense_patch_proxy.py
- realsense_corrupt_proxy.py
- attacks/
  - generate_random_patch.py
  - train_patch_art.py
  - README.md  (이 파일)
- patches/
  - random/
  - trained/
- logs/ (선택)
- runs/ (ultralytics 자동 생성)

---

## 0) 준비: v4l2loopback

예)
```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="PatchCam"