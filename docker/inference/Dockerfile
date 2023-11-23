FROM doduo2.umcn.nl/uokbaseimage/base:tf2.10-pt1.12

WORKDIR /root

#### install specific pixman version
RUN sudo wget https://www.cairographics.org/releases/pixman-0.40.0.tar.gz
RUN sudo tar -xf pixman-0.40.0.tar.gz

WORKDIR /root/pixman-0.40.0

RUN ./configure && \
    make && \
    make install && \
    export LD_PRELOAD=/usr/local/lib/libpixman-1.so.0.40.0

WORKDIR /root
RUN rm -r pixman-0.40.0
RUN rm pixman-0.40.0.tar.gz

# install pyvips
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# download & unpack vips
RUN wget -q https://github.com/libvips/libvips/releases/download/v8.13.0/vips-8.13.0.tar.gz  && \
    sudo -S apt-get update && \
    sudo -S apt-get install -y libjpeg-turbo8-dev && \
    sudo -S apt-get install -y libgtk2.0-dev && \
    sudo -S apt-get install -y libgsf-1-dev && \
    sudo -S apt-get install -y libtiff5-dev && \
    sudo -S apt-get install -y libopenslide-dev && \
    tar xf vips-8.13.0.tar.gz > /dev/null 2>&1

RUN rm vips-8.13.0.tar.gz
WORKDIR /root/vips-8.13.0

# make and install vips
RUN ./configure && \
    make && \
    sudo -S make install && \
    cd .. && \
    sudo -S ldconfig && \
    sudo -S pip3.9 install pyvips

WORKDIR /root

# install ASAP
COPY files/ASAP*.deb /root/
RUN apt-get update && \
    apt-get install --assume-yes /root/ASAP-2.1-Ubuntu2004.deb && \
    ldconfig && \
    SITE_PACKAGES=`python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"` && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/user

# clone HS2P repo
RUN git clone https://github.com/computationalpathologygroup/hs2p.git

# copy necessary HS2P files
COPY files/panda_patch_extraction.yaml hs2p/config/extraction/.
COPY files/patch_extraction.py hs2p/.

# clone HViT repo
RUN git clone https://github.com/computationalpathologygroup/hvit.git

#### copy HViT pretrained weights
RUN mkdir -p hvit/checkpoints/pretrained
COPY files/checkpoints/pretrained/ hvit/checkpoints/pretrained

#### copy HViT trained weights
RUN mkdir -p hvit/checkpoints/trained
COPY files/checkpoints/trained/ hvit/checkpoints/trained

# copy necessary HViT files
COPY files/panda_feature_extraction.yaml hvit/config/feature_extraction/.
COPY files/panda_inference.yaml hvit/config/inference/.
COPY files/ensemble.py hvit/inference/.
COPY files/utils.py hvit/source/.
COPY files/extract_features.py hvit/.
COPY files/dataset.py hvit/source/.

# install HViT requirements
RUN pip3 install -r hvit/requirements.txt
RUN pip3 install -e hvit/.

#### Configure entrypoint
COPY run.sh .
ENTRYPOINT ["/bin/bash", "run.sh"]
