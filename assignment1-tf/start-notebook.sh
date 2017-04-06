#@IgnoreInspection BashAddShebang
nvidia-docker run --rm -u root -e NB_UID=`id -u` \
  -p 8080:8888 -p 6006:6006 \
  -v `pwd`:/root \
  -it tf-notebook sh -c "jupyter notebook --ip 0.0.0.0 ."