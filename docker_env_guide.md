<!-- instructions for build the docker image from the ./docker/Dockerfile dockerfile with GPU and CUDA support and run the container to see the output -->

## Build the docker image

```powershell
docker build -t mpxgat .
```

## Run the docker container with a defined name, GPU support and in detached mode

```powershell
docker run --gpus=all -dt --name mpxgat mpxgat
```

## Run interactive terminal on the created container

```powershell
docker exec -it mpxgat bash
```

## delete the container

```powershell
docker rm mpxgat
```

## build docker compose

```powershell
docker compose up -d --build
```