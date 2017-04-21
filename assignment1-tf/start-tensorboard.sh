#@IgnoreInspection BashAddShebang
nvidia-docker run --rm --name run-tensorboard \
  -p 6006:6006 \
  -v `pwd`/tf-logs:/root/logs \
  -it tensorflow/tensorflow:latest-devel-gpu-py3 sh -c "tensorboard --logdir /root/logs"