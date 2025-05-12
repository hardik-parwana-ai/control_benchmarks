# FROM rayproject/ray:latest-py39-cu121
# FROM rayproject/ray:2.42.1-py39-gpu
FROM rayproject/ray:2.42.1-py39-cu124
RUN pip3 install torch
RUN sudo apt-get update
RUN sudo apt-get install -y vim gedit ffmpeg
RUN pip3 install matplotlib
#FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
#RUN apt-get update
#RUN apt-get install -y gedit
#RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
#RUN apt-get install -y python3-pip
#RUN pip3 install PyQt5
#RUN pip3 install matplotlib

#ARG DEBIAN_FRONTEND=noninteractive
#RUN apt-get install -y libqt5gui5
#RUN apt-get install -y texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra dvipng cm-super
#RUN apt install -y vim

#RUN apt-get install -y git vim
#RUN apt-get install -y ffmpeg
