# Docker Sample

A sample of building docker image. In this docker image, a pytorch model which has been pre-trained using ImageNet-1K dataset will be employed. Specifically, the pre-trained model will predict the categories of all images in folder *data/images* and save the results as a text file in folder *data/results*.



### Requirements

------

Ubuntu 18.04

Docker >= 19.03

[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)



### Build

------

```sh
cd docker-sample
sudo docker build -t docker_sample .
```



### Run

------

```sh
sudo docker run -it --rm docker_sample
```

if you want to use your own images:

```sh
sudo docker run -v PATH_TO_YOUR_IMAGES:/docker_sample/data -it --rm docker_sample
```

if you want to use NVIDIA GPUs:

```sh
sudo docker run --gpus all -it --rm docker_sample
```



### Share

------

if you want to share your docker image in [Docker Hub](https://hub.docker.com/):

```sh
sudo docker login
sudo docker push docker_sample
```

