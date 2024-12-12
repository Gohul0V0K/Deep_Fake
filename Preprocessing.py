import json
import glob
import numpy as np
import cv2
import os
import face_recognition
from tqdm.autonotebook import tqdm

print("imported")

video_files = glob.glob('qwert/*.mp4')

frame_count = []
valid_video_files = []

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_cnt >= 250:
        valid_video_files.append(video_file)
        frame_count.append(frame_cnt)

print("frames", frame_count)
print("Total number of valid videos: ", len(frame_count))
print('Average frames per valid video:', np.mean(frame_count))

def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = True
    frame_count = 0
    while success:
        success, image = vidObj.read()
        if success:
            frame_count += 1
            print(f"Extracted frame {frame_count}")
            yield image

def create_face_videos(path_list, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for path in tqdm(path_list, desc="Processing videos"):
        out_path = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out_path):
            print("File Already exists: ", out_path)
            continue
        
        print(f"Processing video: {path}")
        frames=[]
        
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))
        for idx,frame in enumerate(frame_extract(path)):
          if(idx <= 250):
            frames.append(frame)
            try:
              print("Running face detection on individual frame...")
              face_locations = face_recognition.face_locations(frame)  
              print(f"Faces detected in frame {idx + 1}: {face_locations}")
                
              for face_location in face_locations:
                top, right, bottom, left = face_location
                cropped_face = frame[top:bottom, left:right]
                out.write(cv2.resize(cropped_face, (112, 112)))
            except Exception as e:
                print(f"Error during processing frame {idx + 1}: {e}")

          frames = []
        
        out.release()
        print(f"Finished processing video: {path}")


create_face_videos(valid_video_files, "qwerty")