import kfserving
import numpy as np
from typing import Dict, List
from ray import serve
import ray
import keras

ray.init()

EMBEDDING_FILE = "wiki-news-300d-1M.vec"
MAX_LEN = 124
EMB_DIM = 300
LABELS_TO_INDEX = {
    'B-LOC': 0,
    'B-MISC': 1,
    'B-ORG': 2,
    'I-LOC': 3,
    'I-MISC': 4,
    'I-ORG': 5,
    'I-PER': 6,
    'O': 7
}


def getEmbeddingsIndex():
    out = {}
    with open(EMBEDDING_FILE) as f:
        for line in f.readlines():
            data = line.split(" ")
            out[data[0]] = np.array(data[1:], dtype='float32')
        return out


@ray.remote
class Processing(object):
    def __init__(self):
        self.embeddings_index = getEmbeddingsIndex()
        self.labels_to_index = LABELS_TO_INDEX

    @ray.method(num_returns=1)
    def preProcess(self, instances: List[str]):
        X = np.zeros((len(instances), MAX_LEN, EMB_DIM), dtype=np.float32)
        default = np.random.rand(EMB_DIM).astype('float32')
        num_tokens = []
        for i, sample in enumerate(instances):
            sentence = sample.split()
            num_tokens.append(len(sentence))
            for j, token in enumerate(sentence[:MAX_LEN]):
                X[i, j] = self.embeddings_index.get(token, default)
        return X, num_tokens

    @ray.method(num_returns=1)
    def postProcess(self, predictions, num_tokens):
        reverse_label_index = {v: k for k, v in self.labels_to_index.items()}
        assert len(predictions) == len(num_tokens)
        n = len(predictions)
        preds = np.argmax(predictions, axis=-1)
        out = []
        for i in range(n):
            p = preds[i][:num_tokens[i]]
            out.append([reverse_label_index[idx] for idx in p])
        return out


processing_actor = Processing.remote()


@serve.deployment(name="custom-model", config={"num_replicas": 4})
class NERModel(kfserving.KFModel):
    def __init__(self):
       self.name = "custom-model"
       super().__init__(self.name)
       self.load()

    def load(self):
        self.model = keras.models.load_model('ner_model/1/')

    def predict(self, request: Dict) -> Dict:
        X, num_tokens = ray.get(
            processing_actor.preProcess.remote(request['instances']))
        predictions = self.model.predict(X)
        result = ray.get(
            processing_actor.postProcess.remote(predictions, num_tokens)
        )
        return {"predictions": result}


if __name__ == "__main__":
    kfserving.KFServer().start({"custom-model": NERModel})
