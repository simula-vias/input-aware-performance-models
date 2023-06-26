import json
import os
import subprocess
import sys

video_file = sys.argv[1]
input_cfg = json.loads(sys.argv[2])

# from configs.csv
activated = "--cabac,--ref,--deblock,--analyse,--me,--subme,--mixed-ref,--merange,--trellis,--8x8dct,--fast-pskip,--chroma-qp-offset,--bframes,--b-pyramid,--b-adapt,--direct,--weightb,--open-gop,--weightp,--scenecut,--rc-lookahead,--mbtree,--qpmax,--aq-mode".split(
    ","
)
deactivated = "--no-cabac,,,,,,--no-mixed-ref,,,--no-8x8dct,--no-fast-pskip,,,,,,--no-weightb,,,,,--no-mbtree,,".split(
    ","
)
has_argument = [
    s == "1" for s in "0,1,1,1,1,1,0,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1".split(",")
]
defaults = "1,3,0:0,,hex,6,1,16,1,1,1,0,3,normal,1,spatial,1,,2,40,40,1,51,1".split(",")
names = "cabac,ref,deblock,analyse,me,subme,mixed_ref,me_range,trellis,8x8dct,fast_pskip,chroma_qp_offset,bframes,b_pyramid,b_adapt,direct,weightb,open_gop,weightp,scenecut,rc_lookahead,mbtree,qpmax,aq-mode".split(
    ","
)

assert len(activated) == len(names)
assert len(deactivated) == len(names)
assert len(has_argument) == len(names)
assert len(defaults) == len(names)
assert len(activated) == len(names)

x264_line = "{ time x264 "
csvLine = 'csvLine=",'

for opt_idx, opt in enumerate(names):

    if opt in input_cfg:
        arg = input_cfg[opt]
    else:
        arg = defaults[opt_idx]

    csvLine += str(arg) + ","

    if has_argument[opt_idx]:
        if arg == "":
            continue

        x264_line += f" {activated[opt_idx]} {arg}"
    elif arg == "1":
        x264_line += f" {activated[opt_idx]}"
    elif arg == "0":
        x264_line += f" {deactivated[opt_idx]}"


x264_line += " --output $outputlocation $inputlocation ; } 2> $logfilename"
csvLine += '"'
video_name = os.path.splitext(os.path.basename(video_file))[0]
video_results = os.path.join(os.path.dirname(video_file), "logs", video_name + ".csv")
result_header = "configurationID,cabac,ref,deblock,analyse,me,subme,mixed_ref,me_range,trellis,8x8dct,fast_pskip,chroma_qp_offset,bframes,b_pyramid,b_adapt,direct,weightb,open_gop,weightp,scenecut,rc_lookahead,mbtree,qpmax,aq-mode,size,usertime,systemtime,elapsedtime,cpu,frames,fps,kbs"

if not os.path.isfile(video_results):
    open(video_results, "w").write(result_header + "\n")

with open("measure.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    f.write("numb='1'")
    f.write(
        '\nlogfilename="/tmp/$numb.log"\ninputlocation="$1"\noutputlocation="/tmp/video$numb.264"\n\n'
    )
    f.write(x264_line)
    f.write("\n# extract output video size\n")
    f.write("size=`ls -lrt $outputlocation | awk '{print $5}'`\n")
    f.write("# analyze log to extract relevant timing information and CPU usage\n")
    f.write(
        """realtime=`grep "real" $logfilename | sed 's/real// ; s/sys// ;s/user//' | cut -d "%" -f 1`\n"""
    )
    f.write(
        """usertime=`grep "user" $logfilename | sed 's/real// ; s/sys// ;s/user//' | cut -d "%" -f 1`\n"""
    )
    f.write(
        """systime=`grep "sys" $logfilename | sed 's/real// ; s/sys// ;s/user//' | cut -d "%" -f 1`\n"""
    )
    f.write("\n# analyze log to extract fps and kbs\n")
    f.write(
        """persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`"""
    )
    f.write("\n# clean\nrm $outputlocation\n\n")
    f.write(csvLine)
    f.write('\ncsvLine="$csvLine$size,$usertime,$systime,$realtime,,$persec"\necho $csvLine')

result_line = subprocess.check_output(
    ["/bin/bash", "./measure.sh", video_file], universal_newlines=True
)
open(video_results, "a").write(result_line)

result_dict = {
    k.strip(): v.strip()
    for k, v in zip(result_header.split(","), result_line.split(","))
}
print(json.dumps(result_dict))
