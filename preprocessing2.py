import json
import glob
import numpy as np
import cv2
import os
import face_recognition
from tqdm.autonotebook import tqdm

print("imported")

# List of specific video files to process
selected_videos = [
    "vpmyeepbep.mp4", "fzvpbrzssi.mp4", "syxobtuucp.mp4", "dhjnjkzuhq.mp4", 
    "xcruhaccxc.mp4", "vtunvalyji.mp4", "qyqufaskjs.mp4", "rnpefxwptv.mp4", 
    "sttnfyptum.mp4", "xugmhbetrw.mp4", "uqtqhiqymz.mp4", "jawgcggquk.mp4", 
    "yexeazbqig.mp4", "hivnldfvyl.mp4", "wfzjxzhdkj.mp4", "apedduehoy.mp4", 
    "exseruhiuk.mp4", "lnhkjhyhvw.mp4", "prdrkaxeob.mp4", "pqdeutauqc.mp4", 
    "xjzkfqddyk.mp4", "fufcmupzen.mp4", "lokzwdldxp.mp4", "hplxtssgnz.mp4", 
    "gcdtglsoqj.mp4", "wclvkepakb.mp4", "upmgtackuf.mp4", "rmuxlgsedw.mp4", 
    "lnjkpdviqb.mp4", "hicjuubiau.mp4", "fsaronfupy.mp4", "prmwoaeeng.mp4", 
    "ucthmsajay.mp4", "fopjiyxiqd.mp4", "jytrvwlewz.mp4", "rmufsuogzn.mp4", 
    "nrnklcxdzq.mp4", "dxfdovivlw.mp4", "srfefmyjvt.mp4", "eiwtggvtfp.mp4", 
    "fgobmbcami.mp4", "hsbwhlolsn.mp4", "ldtgofdaqg.mp4", "rktrpsdlci.mp4", 
    "heiyoojifp.mp4", "uprwuohbwx.mp4", "wynotylpnm.mp4", "yxvmusxvcz.mp4", 
    "xmkwsnuzyq.mp4", "fdpisghkmd.mp4", "xchzardbfa.mp4", "gvasarkpfh.mp4", 
    "nhsijqpoda.mp4", "ybbrkacebd.mp4", "fnslimfagb.mp4", "didzujjhtg.mp4", 
    "ngvcqxjhyb.mp4", "xkfliqnmwt.mp4", "jzupayeuln.mp4", "eppyqpgewp.mp4", 
    "ljuuovfkgi.mp4", "doniqevxeg.mp4", "onbgbghesu.mp4", "jvtjxreizj.mp4", 
    "wcqvzujamg.mp4", "vokrpfjpeb.mp4", "apvzjkvnwn.mp4", "knxltsvzyu.mp4", 
    "exxqlfpnbz.mp4", "lzbmwwejxb.mp4", "psjfwjzrrh.mp4", "nweufafotd.mp4", 
    "ybnucgidtu.mp4", "ddtbarpcgo.mp4", "qarqtkvgby.mp4", "chqqxfuuzi.mp4", 
    "dpevefkefv.mp4", "kqlvggiqee.mp4", "xdezcezszc.mp4", "gnmmhlbzge.mp4", 
    "sylnrepacf.mp4", "khtwrijuqn.mp4", "lsmnqsnqld.mp4", "yljecirelf.mp4", 
    "vmxfwxgdei.mp4", "aayrffkzxn.mp4"
]

video_files = glob.glob('deeeeep/dfdc_train_part_0/*.mp4')

# Filter the list to include only the selected videos
valid_video_files = [vf for vf in video_files if os.path.basename(vf) in selected_videos]

frame_count = []

for video_file in valid_video_files:
    cap = cv2.VideoCapture(video_file)
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_cnt >= 250:
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
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))
        for idx, frame in enumerate(frame_extract(path)):
            if idx <= 250:
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
        
        out.release()
        print(f"Finished processing video: {path}")

create_face_videos(valid_video_files, "Real_Face_only_data")
