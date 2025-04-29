# SyncNet

This repository contains the demo for the audio-to-video synchronisation network (SyncNet). This network can be used for audio-visual synchronisation tasks including: 
1. Removing temporal lags between the audio and visual streams in a video;
2. Determining who is speaking amongst multiple faces in a video. 

Please cite the paper below if you make use of the software. 

## Git Clone
```
git clone https://github.com/sogang-capzzang/syncnet.git
cd syncnet
```

## Dependencies
```
pip install -r requirements.txt
```

## Model Download
```
./download_model.sh
```

In addition, `ffmpeg` is required.

## Evaluation: lib sync video score (default: cosine 유사도) -> SyncNetInstance.py

SyncNet demo: --videofile {video위치}
```
python demo_syncnet.py --videofile data/example.avi --tmp_dir /path/to/temp/directory
```

Check that this script returns:
```
AV offset: 	3 
Max sim: 	0.862
Confidence: 	0.862
```

## Publications
 
```
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```
