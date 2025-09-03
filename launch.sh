if sudo docker image inspect mirena_ws:latest > /dev/null 2>&1; then
  echo "[Launch] Image exists: Launching"
else
  echo "[Launch] Image does not exist: Building"
  sudo docker build ./docker -t mirena_ws
fi


sudo docker run -it --rm \
  --net=host \
  --device /dev/dri:/dev/dri \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/mirena_ws \
  mirena_ws

