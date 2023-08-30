FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
WORKDIR /usr/src/mpxgat
# install python
RUN apt update && apt install -y --no-install-recommends python3 python3-pip python3-setuptools python3-wheel python3-dev git
RUN alias python=python3

# copy requirements for other dependencies (pytorch_geometric)
COPY requirements.txt ./

# install pytorch
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# install dependencies (pytorch_geometric)
RUN pip install --no-cache-dir -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# install pynvml (https://pypi.org/project/pynvml/)
RUN pip install pynvml==11.5.0

# COPY ../ .

# CUSTOMIZE: .bashrc
RUN echo "export PS1='🐳 [\[\033[1;36m\]\h \[\033[1;34m\]\W\[\033[0;35m\]\[\033[1;36m\]]# \[\033[0m\]'" >> ~/.bashrc

# execute a command that keeps the container running
CMD ["tail", "-f", "/dev/null"]