#!/bin/bash

set -xv

cat listVideo.csv | while read line
  do
    read -d, videoname extension < <(echo $line)
    echo $videoname
    echo $extension
    path="./videos/$videoname.$extension"
    newPath="$videoname.$extension"
    echo $newPath
    x64configs=`ls ./scripts/*.sh`
    csvOutput="./res/$videoname.csv"
    cp $path $newPath
    echo "Starting to work with video: " $videoname
    header='configurationID,cabac,ref,deblock,analyse,me,subme,mixed_ref,me_range,trellis,8x8dct,fast_pskip,chroma_qp_offset,bframes,b_pyramid,b_adapt,direct,weightb,open_gop,weightp,scenecut,rc_lookahead,mbtree,qpmax,aq-mode,size,usertime,systemtime,elapsedtime,cpu,frames,fps,kbs'
    touch $csvOutput
    cat /dev/null > $csvOutput
    echo "$header" > $csvOutput
    for x264config in $x64configs
    do
        echo "Processing: " $x264config
        csvLine=`bash $x264config $newPath`
        echo "$csvLine" >> $csvOutput
    done
    rm $newPath
done

echo "Done with videos! See the res folder!"
