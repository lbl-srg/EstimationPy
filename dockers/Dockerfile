# ==============================================================================
#
# This Dockerfile contains all the commands required to set up an environment
# for using EstimationPy
# 

FROM ubuntu:14.04

MAINTAINER lbl-srg

# Avoid interaction
ENV DEBIAN_FRONTEND noninteractive

# =========== Basic Configuration ===============================================
# Update the system
RUN apt-get -y update
RUN apt-get install -y build-essential git python python-dev python-setuptools make cmake gfortran

# Install pip for managing python packages
RUN apt-get install -y python-pip python-lxml
RUN pip install cython
RUN apt-get install -y python-lxml

# Add folders that will contains code before and after installation
RUN mkdir -p /home/docker/to_install
RUN mkdir -p /home/docker/installed

# Create an user and an environmental variable associated to it
RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# =========== Install JAVA =======================================================
# Install Java (Open JDK, version 7)
RUN \
  apt-get install -y openjdk-7-jdk && \
  rm -rf /var/lib/apt/lists/*
  
# Define commonly used JAVA_HOME variable
ENV JAVA_HOME /usr/lib/jvm/java-7-openjdk-amd64
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64" >> /root/.bashrc
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64" >> /home/docker/.bashrc

# =========== Install PyFMI dependencies ======================================================
# Install Sundials ODE, DAE
COPY ./softwares/sundials-2.5.0.tar.gz /home/docker/to_install/
RUN cd /home/docker/to_install && tar xzvf ./sundials-2.5.0.tar.gz
WORKDIR /home/docker/to_install/sundials-2.5.0
RUN mkdir -p build
WORKDIR /home/docker/to_install/sundials-2.5.0/build
RUN ../configure CFLAGS="-fPIC" --prefix=/home/docker/installed/sundials-2.5.0
RUN make && make install && make clean

WORKDIR /home/docker

# Install BLAS and LAPACK
RUN apt-get update
RUN apt-get install -y libblas3gf libblas-doc libblas-dev
RUN apt-get install -y liblapack3gf liblapack-doc liblapack-dev

# Other dependencies required before installing pylab and Matplotlib
RUN apt-get install -y pkgconf libpng-dev libfreetype6-dev

# Install Numpy, Matplotlib, SciPy and Pandas
RUN pip install numpy
RUN apt-get install -y python-matplotlib
RUN pip install scipy
RUN pip install pandas

# Install svn and git
RUN apt-get install -y git subversion

# Install assimulo
RUN cd /home/docker/to_install && svn checkout https://svn.jmodelica.org/assimulo/tags/Assimulo-2.8/
WORKDIR /home/docker/to_install/Assimulo-2.8
RUN python setup.py install --sundials-home=/home/docker/installed/sundials-2.5.0/ --blas-home=/usr/lib/lapack/

# Install FMILib
RUN cd /home/docker/to_install/ && svn checkout https://svn.jmodelica.org/FMILibrary/tags/2.0.1/ FMILibrary-2.0.1
RUN cd /home/docker/to_install && ls -la
WORKDIR /home/docker/to_install/FMILibrary-2.0.1/
RUN ls -la
RUN mkdir -p build-fmil
WORKDIR /home/docker/to_install/FMILibrary-2.0.1/build-fmil
RUN cmake -DFMILIB_INSTALL_PREFIX=/home/docker/installed/FMIlib2.0.1 \
/home/docker/to_install/FMILibrary-2.0.1
RUN make install test

# Finally install PyFMI
RUN cd /home/docker/to_install/ && svn checkout https://svn.jmodelica.org/PyFMI/tags/PyFMI-2.0b1/
WORKDIR /home/docker/to_install/PyFMI-2.0b1
RUN python setup.py install --fmil-home=/home/docker/installed/FMIlib2.0.1/

# Create "dummy" Dymola license file. It is also required by FMUs that are exported in
# binary export mode
RUN mkdir -p /home/docker/.dynasim
RUN touch /home/docker/.dynasim/dymola.lic
RUN echo "SERVER yourserver.com ANY" >> /home/docker/.dynasim/dymola.lic
RUN echo "VENDOR dynasim" >> /home/docker/.dynasim/dymola.lic
RUN echo "USE_SERVER" >> /home/docker/.dynasim/dymola.lic

ENV DYMOLA_RUNTIME_LICENSE "/home/ubuntu/.dynasim/dymola.lic"
RUN echo "export DYMOLA_RUNTIME_LICENSE=/home/ubuntu/.dynasim/dymola.lic" >> /root/.bashrc
RUN echo "export DYMOLA_RUNTIME_LICENSE=/home/ubuntu/.dynasim/dymola.lic" >> /home/docker/.bashrc

# Install EstimationPy
RUN cd /home/docker/to_install && git clone https://github.com/lbl-srg/EstimationPy.git \
  && cd ./EstimationPy && python setup.py install

WORKDIR /home/docker

# Install ipython notebook
RUN pip install "ipython[notebook]"

# Change ownership of the content of /home/docker
RUN chown -R docker:docker /home/docker/ 

# Change user to docker
USER docker

# Create folder that will be used as a shared volume
RUN mkdir -p /home/docker/shared_folder
VOLUME ["/home/docker/shared_folder"]

# Create environmental variables for the display
ENV DISPLAY :0.0
ENV USER docker

# Expose the port where the ipython notebook server will listen
EXPOSE 8888

# Command to run by default in detached mode
CMD ipython notebook --ip="0.0.0.0" --port=8888 --notebook-dir=/home/docker/shared_folder --no-browser