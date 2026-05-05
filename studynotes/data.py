import torch
import requests
import tarfile

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook




# download and extract file
url = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz"
filename = url.split("/")[-1]
with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)


with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()
    csv_path = tar.getnames()[0]


# structure the data
prefix = csv_path + "/"

train_df = pd.read_csv(prefix + 'train.csv', header=None)
train_df.head()

eval_df = pd.read_csv(prefix + 'test.csv', header=None)
eval_df.head()

train_df[0] = (train_df[0] == 2).astype(int)
eval_df[0] = (eval_df[0] == 2).astype(int)

train_df = pd.DataFrame({
    'id': range(len(train_df)),
    'text': train_df[1].replace(r'\n', ' ', regex=True),
    'alpha': ['alpha']*train_df.shape[0],
    'label':train_df[0]
})

# print(train_df.head())

dev_df = pd.DataFrame({
    'id': range(len(eval_df)),
    'text': eval_df[1].replace(r'\n', ' ', regex=True),
    'alpha': ['alpha'] * eval_df.shape[0],
    'label':eval_df[0]
})

# save the data as tsv files
train_df.to_csv(prefix+'train.tsv', sep='\t', index=False, header=False)
dev_df.to_csv(prefix+'dev.tsv', sep='\t', index=False, header=False)


























# Create a TransformerModel
# from simpletransformers.classification import ClassificationModel
#
# import pytorch
#
# import torchvision
# import detection
# import simpletransformers
#
# model = ClassificationModel('roberta', 'roberta-base')
#
# model = simpletransformers.classification.classification_model('roberta', 'roberta-base')
# ClassificationModel('roberta', 'roberta-base')
#
#
# # Train the model
# model.train_model(train_df)
#
# # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)









































