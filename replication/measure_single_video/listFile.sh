#!/bin/bash

touch "./listVideo.csv"
echo `ls ./videos/*` > listVideo.csv 
sed -i 's/ /\n/g' listVideo.csv
sed -i 's:./videos/::g' listVideo.csv
sed -i 's/\./ /' listVideo.csv
