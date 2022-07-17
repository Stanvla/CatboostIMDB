
Building an image with tag v1.0 from the current directory.
```
docker build -t catboost:v1.0 .
```

Run the container in the detached mode
```
docker run -d -p 5000:5000 --name catboost --rm catboost:v1.0
```