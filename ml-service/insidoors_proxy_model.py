#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Required python modules:
# - sqlalchemy
# - mysqlclient
# - names
# - pandas
# - matplotlib
# - torch
# - transformers[torch]
# - evaluate


# In[ ]:


# FUTURE ITERATIONS:
# 1. handle class imbalances
# 2. properly incorporate tabular data


# ## Insidoors Text Classification Model for Proxy Access Logs
# *Version: 1*
# <br>
# This notebook details the first iteration of a binary text classification model that identifies suspicious proxy access cases

# ### Load data from MySQL

# In[ ]:


import pandas as pd
from sqlalchemy import create_engine


# In[ ]:


# Establish connection to mysql
# TODO: use .env file
# !! Replace placeholders before running !!

USER = '<USER>'
PASSWORD = '<PASSWORD>'
HOST = '<HOST>' 
PORT = '<PORT>'
DATABASE = 'insidoors'
TABLE = 'proxy_log'
CONNECTION_STRING = 'mysql+mysqldb://' + USER + ':' + PASSWORD + '@' + HOST + ':' + PORT + '/' + DATABASE

engine = create_engine(CONNECTION_STRING)
query = 'SELECT * FROM ' + TABLE + ';'
df = pd.read_sql(query, engine)

display(df)


# FOR GOOGLE COLAB: COMMENT OUT CODE ABOVE AND USE THE FOLLOWING

# df = pd.read_csv('proxy_data.csv', sep=';', header=0)

# display(df)


# In[ ]:


# Display distribution of suspect cases

print(df['suspect'].value_counts())
display(df['suspect'].value_counts().plot(kind='bar', rot=0))


# In[ ]:


# Display distribution of machines and url categories

print(df['machine_name'].value_counts())
print()
print(df['category'].value_counts())


# In[ ]:


# Display suspect cases

display(df[df['suspect'] == 6])


# ### Preprocess data

# In[ ]:


import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


# In[ ]:


# Prepare 'suspect' column for binary classification

df.loc[df['suspect'] == 6, 'suspect'] = 1

print(df['suspect'].value_counts())


# In[ ]:


# Concatenate input columns into a single string

df = df.astype(str)
df['suspect'] = df['suspect'].astype(int) # Keep 'suspect' column as int
df['input'] = df[['user_id', 'access_date_time', 'machine_name',
                  'url', 'category', 'bytes_in', 'bytes_out']].agg(', '.join, axis=1)

display(df)


# In[ ]:


# Split dataset into train, validation, and test
# 60% train, 20% validation, 20% test

train, val, test = np.split(df.sample(frac=1, random_state=480),
                            [int(0.6 * len(df)), int(0.8 * len(df))])

print(train.shape)
print(val.shape)
print(test.shape)


# In[ ]:


# Select a curated sample from the test dataset to use during evaluation
# 50% suspect, 50% non-suspect

suspect = test.loc[test['suspect'] == 1]
nonsuspect = test.loc[test['suspect'] == 0]

sample_nonsuspect = nonsuspect.sample(n=suspect.shape[0], random_state=48)

curated = pd.concat([suspect, sample_nonsuspect])
curated = curated.sample(frac=1)

display(curated)


# In[ ]:


# Tokenize input text using DistilBERT tokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train['input'].tolist(), padding=True, truncation=True)
val_encodings = tokenizer(val['input'].tolist(), padding=True, truncation=True)
test_encodings = tokenizer(test['input'].tolist(), padding=True, truncation=True)
curated_encodings = tokenizer(curated['input'].tolist(), padding=True, truncation=True)


# In[ ]:


# Create batches using DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[ ]:


# Define class for PyTorch dataset, to be used as model input

class ProxyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[ ]:


# Create PyTorch datasets

train_dataset = ProxyDataset(train_encodings, train['suspect'].tolist())
val_dataset = ProxyDataset(val_encodings, val['suspect'].tolist())
test_dataset = ProxyDataset(test_encodings, test['suspect'].tolist())
curated_dataset = ProxyDataset(curated_encodings, curated['suspect'].tolist())


# ### Prepare evaluation metrics

# In[ ]:


import evaluate


# In[ ]:


# Load accuracy metric

accuracy = evaluate.load('accuracy')


# In[ ]:


# Define function to compute model metrics

def compute_metrics(results):
    predictions, labels = results
    predictions = np.argmax(predictions, axis=1)
    
    return accuracy.compute(predictions=predictions, references=labels)


# ### Build and train model

# In[ ]:


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


# In[ ]:


# Load pretrained DistilBERT model
# TODO: check if model has to be the same as tokenizer, create global variable

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


# In[ ]:


# Define training hyperparameters

training_args = TrainingArguments(
    output_dir='training_output_proxy',
    learning_rate=0.00002,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)


# In[ ]:


# Define trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# In[ ]:


# Train model

trainer.train()


# ### Evaluate model

# In[ ]:


from evaluate import evaluator


# In[ ]:


# Define trainer for evaluation

eval_trainer = Trainer(
    model=model,
    eval_dataset=curated_dataset,
    compute_metrics=compute_metrics
)


# In[ ]:


# Evaluate model

eval_trainer.evaluate()


# In[ ]:


# Save model

trainer.save_model('insidoors_proxy_v1')


# In[ ]:


# Show model predictions on a few test inputs

test_suspect = suspect['input'].sample(n=3).tolist()
test_nonsuspect = nonsuspect['input'].sample(n=3).tolist()
test_input = test_suspect + test_nonsuspect

print('Sample input:')
display(test_input)

saved_tokenizer = AutoTokenizer.from_pretrained('insidoors_proxy_v1')
saved_model = AutoModelForSequenceClassification.from_pretrained('insidoors_proxy_v1')
test_input = saved_tokenizer(test_input, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    logits = saved_model(**test_input).logits

print()
print('Model predictions:')

for tensor in logits:
  print(tensor.argmax().item())

