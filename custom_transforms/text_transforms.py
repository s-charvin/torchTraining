

import sklearn
import torch
# preprocessing.OneHotEncoder()
# preprocessing.LabelEncoder()


def label2id(label_list):
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(y=label_list)
    # id2label = {f"{id}": label for id, label in enumerate()}
    return label_encoder.transform(label_list), label_encoder.classes_
