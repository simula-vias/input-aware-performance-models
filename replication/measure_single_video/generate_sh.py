import pandas as pd
import numpy as np

x264_config = pd.read_csv('configs.csv', delimiter=',', header=None).set_index(0)

x264_config = x264_config.fillna('None')

activated = x264_config.iloc[0]
deactivated = x264_config.iloc[1]
has_argument = x264_config.iloc[2]

x264_config_data = x264_config[5:]

nb_config = x264_config_data.shape[0]

#sh numb definition line
numbLines = []

for i in range(nb_config):
    numbLines.append("numb='"+str(i+1)+"'")

cmd_x264 = []
csvLines = []

for j in range(nb_config):
    l = x264_config_data.iloc[j]
    csvLine = 'csvLine="$numb,'

    x264_line = '{ time x264 '
    for i in np.arange(1,x264_config_data.shape[1]+1,1):
        arg = l[i]
        if arg!='None':
            if has_argument[i]=='1':
                x264_line+=' '+activated[i]+' '+str(arg)
            else:
                if arg == '1':
                    if activated[i]!='None':
                        x264_line+=' '+str(activated[i])
                if arg == '0':
                    if deactivated[i]!='None':
                        x264_line+=' '+str(deactivated[i])
        csvLine+=str(arg)+','
    x264_line+=' --output $outputlocation $inputlocation ; } 2> $logfilename'
    cmd_x264.append(x264_line)
    csvLine+='"'
    csvLines.append(csvLine)

for i in range(nb_config):
    with open('./scripts/'+str(i)+'.sh','w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(numbLines[i])
        f.write('\nlogfilename="./logs/$numb.log"\ninputlocation="$1"\noutputlocation="./video$numb.264"\n\n')
        f.write(cmd_x264[i])
        f.write('\n# extract output video size\n')
        f.write("size=`ls -lrt $outputlocation | awk '{print $5}'`\n")
        f.write('# analyze log to extract relevant timing information and CPU usage\n')
        f.write("""time=`grep "user" $logfilename | sed 's/elapsed/,/ ; s/system/,/ ;s/user/,/' | cut -d "%" -f 1`""")
        f.write('\n# analyze log to extract fps and kbs\n')
        f.write("""persec=`grep "encoded" $logfilename | sed 's/encoded// ; s/fps// ; s/frames// ; s//,/' | cut -d "k" -f 1`""")
        f.write('\n# clean\nrm $outputlocation\n\n')
        f.write(csvLines[i])
        f.write('\ncsvLine="$csvLine$size,$time,$persec"\necho $csvLine')
