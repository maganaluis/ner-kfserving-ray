# Example of how to use KFServing + Ray for NER


Test the model: 

```
python main.py
curl localhost:8080/v1/models/custom-model:predict -d @./input.json
```

Requirements:
```
kfserving>=0.6
tensorflow
keras
```