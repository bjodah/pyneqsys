FROM debian:stretch

MAINTAINER Björn Dahlgren <bjodah@gmail.com>

ENV LANG C.UTF-8

# This dockerfile is designed to run on binder (mybinder.org)
RUN apt-get update && \
    apt-get --quiet --assume-yes install curl git g++-6 libgmp-dev binutils-dev bzip2 make cmake sudo \
    python-dev python-pip liblapack-dev && \
    apt-get clean && \
    curl -LOs http://computation.llnl.gov/projects/sundials/download/sundials-2.7.0.tar.gz && \ 
    tar xzf sundials-2.7.0.tar.gz && mkdir build/ && cd build/ && \
    cmake -DCMAKE_PREFIX_PATH=/usr/local -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DEXAMPLES_ENABLE=OFF -DEXAMPLES_INSTALL=OFF -DLAPACK_ENABLE=ON \
    ../sundials*/ && make install && cd - && rm -r build/ sundials* && \
    python -m pip install --upgrade pip

# At this point the system should be able to pip-install the pakcage and all of its dependencies. We'll do so
# when running the image using the ``host-jupyter-using-docker.sh`` script. Installed packages are cached.
