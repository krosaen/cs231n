#@IgnoreInspection BashAddShebang
nvidia-docker run --rm --name run-tf-notebook \
  -p 8080:8888 \
  -v `pwd`:/root/notebooks \
  -v `pwd`/tf-logs:/root/logs \
  -it tf-notebook sh -c "jupyter notebook --ip 0.0.0.0 /root/notebooks"
