=======
Dockers
=======

This folder contains Dockerfiles that can be used to set up
a working linux environment in which testing **EstimationPy**.
Please make sure to follow the instructions provided in the
documentation to create a Docker image and starting a
Docker container.

Sundials
++++++++

Before starting the creation of the Docker image please download
the Sundials solvers version 2.5.0
from https://computation.llnl.gov/casc/sundials/main.html
and copy the sundials-2.5.0.tar.gz file into the folder
called ``software``.
This file will be copied when the image is created.

Build the image
+++++++++++++++

From this folder, after the sundials package has been downloaded
and placed in the ``softwares`` folder, run the command::

  make build_image

After this command is completed you should see a new image called
``lbl-srg/estimationpy_box``.

It is possible to see the image with the command ``docker images``::

  bash-3.2$ docker images
  REPOSITORY                 TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
  lbl-srg/estimationpy_box   latest              8de2e11c6184        5 minutes ago       2.478 GB
  ...
  ubuntu                     14.04               a005e6b7dd01        2 days ago          188.4 MB

Once the image is built it's possible to start a container that uses that image
with the commands ``make start_container_bash`` or ``make start_container_ipynb``.
The former opens a terminal inside the container where one can run scripts and
test the package. The latter starts a container which runs a IPython notebook
exposed on port 8888 and available on localhost. Look at the Makefile
for more details on the operations that are actually executed.

NOTE:
When running the Docker container the folder ``shared_folder`` is shared between the
host OS and the container. In such a way it's possible for you to move scripts,
data and other files between the container and your computer (the host).

Connect to the IPython notebook
+++++++++++++++++++++++++++++++

In case you run the command::

  make start_container_ipynb

you can open a browser and access the IPython notebook at the address http://127.0.0.1:8888 .
In case you're working on OSx or Windows make sure that VirtualBox is forwarding the
port 8888 to your local host.
To forward the port access you VirtualBox manager, select the virtual machine used by docker,
go to Settings > Networks > Adapter 1 > Port forwarding and make sure to add a rule
that forwards the Guest port 8888 to the host port 8888 on localhost (127.0.0.1).

NOTE:
The IPython notebook uses as default fodler the folder shared between the container and the
host. In such a way if you create an IPython notebook it will be visible on your machine.
