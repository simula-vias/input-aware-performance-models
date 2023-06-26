# starts from ubuntu 18.04 version
FROM ubuntu:18.04
# update packages
RUN apt-get update
# install git 
RUN apt-get install -y git
# install x264, the version we used for the available measurements
RUN apt-get install -y x264=2:0.152.2854+gite9a5903-2
# install python and pandas
RUN apt-get install -y python3-pip\ 
 && pip3 install pandas
# download the configurations, a python script that generates sh scripts (one sh script per configuration)
#RUN git clone https://github.com/anonymous4opens/experiment
RUN mkdir /experiment
COPY . /experiment
# create dir for logs, dir for results, dir for scripts, generate sh scripts, dl videos
RUN cd experiment\
 && mkdir logs \
 && mkdir res \
 && mkdir scripts \
 && mkdir videos
# && python3 generate_sh.py
# && cp /run/media/helge/a1cf4d41-2e70-43ba-8a6b-1c1427f131e7/ugc/*.mkv ./videos/ \
# && cp /run/media/helge/a1cf4d41-2e70-43ba-8a6b-1c1427f131e7/ugc/configs.csv . \
# && bash listFile.sh
# && bash launchUGC.sh

WORKDIR /experiment
ENTRYPOINT ["python3", "run.py"]
CMD ["", "{}"]