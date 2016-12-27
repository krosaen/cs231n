docker run --rm -u root -e NB_UID=`id -u` -p 8888:8888 -v `pwd`:/home/jovyan/work -it jupyter/scipy-notebook
