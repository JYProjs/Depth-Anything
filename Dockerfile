FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN git clone --recursive https://github.com/JYProjs/Depth-Anything.git

RUN cd Depth-Anything && \
    pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]