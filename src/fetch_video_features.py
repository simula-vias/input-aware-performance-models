import json
import os
import subprocess

# cmd = "gsutil ls -r -l 'gs://ugc-dataset/original_videos/**' | grep mkv | awk '{print $3, $1}' > gs_videos.txt"
# subprocess.check_output(cmd, shell=True)

with open("gs_videos.txt", "r") as f, open("extra_video_features.csv", "w") as fout:
    fout.write("FILENAME,ORIG_SIZE,ORIG_BITRATE,ORIG_DURATION\n")
    for r in f:
        gsurl, size = r.split(" ")
        https_url = gsurl.replace("gs://", "https://storage.googleapis.com/")
        print(gsurl, https_url, size)

        cmd = ["ffprobe", "-print_format", "json", "-show_format", https_url]
        output = subprocess.check_output(cmd) #, stderr=subprocess.STDOUT)
        res = json.loads(output)
        filename = os.path.splitext(os.path.basename(https_url))[0]
        duration = float(res["format"]["duration"])
        bitrate = res["format"]["bit_rate"]
        ffprobe_size = res["format"]["size"]

        # seem to be identical for gsutil, local download, and ffprobe (good!)
        # print(res["format"]["size"], size)  

        fout.write(f"{filename},{ffprobe_size},{bitrate},{duration:.3f}\n")
        # break
