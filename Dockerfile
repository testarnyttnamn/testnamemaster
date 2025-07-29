# Dockerfile (at repo root)
FROM ubuntu:bionic

LABEL Description="CLOE Docker Image"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# 1) system deps
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
      bzip2 ca-certificates git libglib2.0-0 libsm6 libxext6 libxrender1 \
      make mercurial openssh-client procps subversion wget && \
    rm -rf /var/lib/apt/lists/*

# 2) install Miniconda
ARG INSTALLER_URL_LINUX64="https://repo.anaconda.com/miniconda/Miniconda3-py312_24.3.0-0-Linux-x86_64.sh"
ARG SHA256SUM_LINUX64="96a44849ff17e960eeb8877ecd9055246381c4d4f2d031263b63fa7e2e930af1"
RUN set -eux; \
    wget "${INSTALLER_URL_LINUX64}" -O /tmp/miniconda.sh -q && \
    echo "${SHA256SUM_LINUX64}  /tmp/miniconda.sh" > /tmp/shasum && \
    sha256sum --check --status /tmp/shasum && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh /tmp/shasum

# 3) create your `cloe` env
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml -n cloe && \
    conda clean -afy

# 4) activate & install pip extras
SHELL ["conda", "run", "-n", "cloe", "/bin/bash", "-lc"]
RUN pip install -U cobaya deepdish BCemu euclidemu2 pyhmcode euclidlib \
               pytest-pydocstyle cosmosis cosmosis-build-standard-library qtvscodestyle

# 5) copy your code in
WORKDIR /src
COPY . .

# 6) default entrypoint just runs pytest
CMD ["bash", "-lc", "pytest --verbose --pycodestyle --pydocstyle --junitxml=pytest.xml --cov=./ --cov-report=term --cov-report=xml=coverage.xml"]
