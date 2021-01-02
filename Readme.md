# Text classifier 
## Objective 
The purpose of this project is to create a classifier that, given as an input a message, 
it predicts the category to which it belongs.

## Deploying

For running the Jupyter the following docker image is used [jupyter/scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook) 

The command lines to build and run the docker are the following: 
- Docker build 
```
docker build -t camorales/jupyter_nlp . 
```
- Docker run. Execute this command from the main directory of the project.
```
docker run -p 8888:8888 -v $(pwd):/home/jovyan/work camorales/jupyter_nlp
```