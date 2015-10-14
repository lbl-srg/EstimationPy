Docker shared folder
====================

When you start the Docker container with the command::
  
  make start_container_bash

or::

  make start_container_ipynb
  
the content of this folder is shared between the
host OS and the container (its guest).
Moreover, the IPython notebook uses this folder to
store its files.
