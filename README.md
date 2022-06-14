# Building identical conda environments

To use the spec file to create an identical environment on the same machine or another machine:
```sh
conda create --name myenv --file spec-file.txt
```

To use the spec file to install its listed packages into an existing environment:
```sh
conda install --name myenv --file spec-file.txt
```
