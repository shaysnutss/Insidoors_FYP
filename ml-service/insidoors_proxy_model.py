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
# - scikit-learn
# - evaluate


# In[ ]:


# !pip install sqlalchemy
# !pip install mysqlclient
# !pip install names
# !pip install pandas
# !pip install matplotlib
# !pip install torch
# !pip install transformers[torch]
# !pip install scikit-learn
# !pip install evaluate


# In[ ]:


# For SMU GPU cluster:
# !pip install sqlalchemy --no-build-isolation
# !pip install pandas --no-build-isolation
# !pip install matplotlib --no-build-isolation
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-build-isolation
# !pip install transformers[torch] --no-build-isolation
# !pip install scikit-learn --no-build-isolation
# !pip install evaluate --no-build-isolation


# In[ ]:


get_ipython().system('whichgpu')


# In[ ]:


# FUTURE ITERATIONS:
# 1. implement over/undersampling
# 2. properly incorporate tabular data


# ## Insidoors Text Classification Model for PC Access, Building Access, and Proxy Logs
# This notebook details the fourth iteration of a binary text classification model that identifies suspicious employee activity. For ease of development, all models are currently built and trained in this notebook. In the future, a separate notebook will be created for each model.
# 
# #### Changelog
# 
# *Version: 4 (Current)*
# * Replaced binary classification with multiclass classification
# 
# *Version: 3*
# * Added models for PC and Building Access logs
# 
# *Version: 2*
# * Implemented custom class weights to handle class imbalance
# 
# *Version: 1*
# * Base model

# In[ ]:


PRETRAINED_MODEL = 'distilbert-base-uncased'
PC_MODEL_NAME = 'insidoors_pc_v4'
BUILDING_MODEL_NAME = 'insidoors_building_v4'
PROXY_MODEL_NAME = 'insidoors_proxy_v4'


# ### Load data from MySQL

# In[ ]:


import pandas as pd
import numpy as np
from sqlalchemy import create_engine


# In[ ]:


# Establish connection to mysql
# TODO: use .env file
# !! Replace placeholders before running !!

# USER = '<USER>'
# PASSWORD = '<PASSWORD>'
# HOST = '<HOST>'
# PORT = '<PORT>'
# DATABASE = 'insidoors'
# TABLE = 'proxy_log'
# CONNECTION_STRING = 'mysql+mysqldb://' + USER + ':' + PASSWORD + '@' + HOST + ':' + PORT + '/' + DATABASE

# engine = create_engine(CONNECTION_STRING)
# query = 'SELECT * FROM ' + TABLE + ';'
# df = pd.read_sql(query, engine)

# display(df)


# FOR GOOGLE COLAB & SMU GPU CLUSTER: COMMENT OUT CODE ABOVE AND USE THE FOLLOWING

pc_df = pd.read_csv('pc_data_10k.csv', sep=';', header=0)
building_df = pd.read_csv('building_data_10k.csv', sep=';', header=0)
proxy_df = pd.read_csv('proxy_data_10k.csv', sep=';', header=0)

print('PC Access logs:')
display(pc_df)

print('Building Access logs:')
display(building_df)

print('Proxy logs:')
display(proxy_df)


# In[ ]:


# Display distribution of suspect PC Access logs

print('Distribution of suspect PC Access logs:')
print(pc_df['suspect'].value_counts())
display(pc_df['suspect'].value_counts().plot(kind='bar', rot=0))


# In[ ]:


# Display suspect PC Access logs

pc_cases = np.sort(pd.unique(pc_df['suspect']))

print('Suspect PC Access logs:')
display(pc_df[pc_df['suspect'].isin(pc_cases[1:])])


# In[ ]:


# Display distribution of suspect Building Access logs

print('Distribution of suspect Building Access logs:')
print(building_df['suspect'].value_counts())
display(building_df['suspect'].value_counts().plot(kind='bar', rot=0))


# In[ ]:


# Display suspect Building Access logs

building_cases = np.sort(pd.unique(building_df['suspect']))

print('Suspect Building Access logs:')
display(building_df[building_df['suspect'].isin(building_cases[1:])])


# In[ ]:


# Display distribution of suspect Proxy logs

print('Distribution of suspect Proxy logs:')
print(proxy_df['suspect'].value_counts())
display(proxy_df['suspect'].value_counts().plot(kind='bar', rot=0))


# In[ ]:


# Display suspect Proxy logs

proxy_cases = np.sort(pd.unique(proxy_df['suspect']))

print('Suspect Proxy logs:')
display(proxy_df[proxy_df['suspect'].isin(proxy_cases[1:])])


# ### Preprocess data

# In[ ]:


import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


# In[ ]:


# Concatenate input columns of PC Access logs into a single string

pc_df = pc_df.astype(str)
pc_df['suspect'] = pc_df['suspect'].astype(int) # Keep 'suspect' column as int
pc_df['input'] = pc_df[['user_id', 'access_date_time', 'log_on_off',
                        'machine_name', 'machine_location']].agg(', '.join, axis=1)

display(pc_df)


# In[ ]:


# Prepare 'suspect' column of PC Access logs for multiclass classification

pc_case2label = dict((int(value), int(key)) for key, value in enumerate(pc_cases))
pc_label2case = dict((int(key), int(value)) for key, value in enumerate(pc_cases)) # Used later in model

print('Case-to-Label mapping for PC Access logs:')
print(pc_case2label)

pc_df['label'] = pc_df['suspect'].map(pc_case2label)

display(pc_df)


# In[ ]:


# Concatenate input columns of Building Access logs into a single string

building_df = building_df.astype(str)
building_df['suspect'] = building_df['suspect'].astype(int) # Keep 'suspect' column as int
building_df['input'] = building_df[['user_id', 'access_date_time', 'direction',
                                    'status', 'office_location']].agg(', '.join, axis=1)

display(building_df)


# In[ ]:


# Prepare 'suspect' column of Building Access logs for multiclass classification

building_case2label = dict((int(value), int(key)) for key, value in enumerate(building_cases))
building_label2case = dict((int(key), int(value)) for key, value in enumerate(building_cases)) # Used later in model

print('Case-to-Label mapping for Building Access logs:')
print(building_case2label)

building_df['label'] = building_df['suspect'].map(building_case2label)

display(building_df)


# In[ ]:


# Concatenate input columns of Proxy logs into a single string

proxy_df = proxy_df.astype(str)
proxy_df['suspect'] = proxy_df['suspect'].astype(int) # Keep 'suspect' column as int
proxy_df['input'] = proxy_df[['user_id', 'access_date_time', 'machine_name',
                              'url', 'category', 'bytes_in', 'bytes_out']].agg(', '.join, axis=1)

display(proxy_df)


# In[ ]:


# Prepare 'suspect' column of Proxy logs for multiclass classification

proxy_case2label = dict((int(value), int(key)) for key, value in enumerate(proxy_cases))
proxy_label2case = dict((int(key), int(value)) for key, value in enumerate(proxy_cases)) # Used later in model

print('Case-to-Label mapping for Proxy logs:')
print(proxy_case2label)

proxy_df['label'] = proxy_df['suspect'].map(proxy_case2label)

display(proxy_df)


# In[ ]:


# Split PC dataset into train, validation, and test
# 60% train, 20% validation, 20% test

pc_train, pc_val, pc_test = np.split(pc_df.sample(frac=1, random_state=480),
                                     [int(0.6 * len(pc_df)), int(0.8 * len(pc_df))])

print('PC dataframe splits:')
print(pc_train.shape)
print(pc_val.shape)
print(pc_test.shape)


# In[ ]:


# Split Building dataset into train, validation, and test
# 60% train, 20% validation, 20% test

building_train, building_val, building_test = np.split(building_df.sample(frac=1, random_state=480),
                                                       [int(0.6 * len(building_df)), int(0.8 * len(building_df))])

print('Building dataframe splits:')
print(building_train.shape)
print(building_val.shape)
print(building_test.shape)


# In[ ]:


# Split Proxy dataset into train, validation, and test
# 60% train, 20% validation, 20% test

proxy_train, proxy_val, proxy_test = np.split(proxy_df.sample(frac=1, random_state=480),
                                              [int(0.6 * len(proxy_df)), int(0.8 * len(proxy_df))])

print('Proxy dataframe splits:')
print(proxy_train.shape)
print(proxy_val.shape)
print(proxy_test.shape)


# In[ ]:


# Tokenize input text using DistilBERT tokenizer

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

pc_train_encodings = tokenizer(pc_train['input'].tolist(), padding=True, truncation=True)
pc_val_encodings = tokenizer(pc_val['input'].tolist(), padding=True, truncation=True)
pc_test_encodings = tokenizer(pc_test['input'].tolist(), padding=True, truncation=True)

building_train_encodings = tokenizer(building_train['input'].tolist(), padding=True, truncation=True)
building_val_encodings = tokenizer(building_val['input'].tolist(), padding=True, truncation=True)
building_test_encodings = tokenizer(building_test['input'].tolist(), padding=True, truncation=True)

proxy_train_encodings = tokenizer(proxy_train['input'].tolist(), padding=True, truncation=True)
proxy_val_encodings = tokenizer(proxy_val['input'].tolist(), padding=True, truncation=True)
proxy_test_encodings = tokenizer(proxy_test['input'].tolist(), padding=True, truncation=True)


# In[ ]:


# Create batches using DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[ ]:


# Define class for PyTorch dataset, to be used as model input

class PyTorchDataset(torch.utils.data.Dataset):
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

pc_train_dataset = PyTorchDataset(pc_train_encodings, pc_train['label'].tolist())
pc_val_dataset = PyTorchDataset(pc_val_encodings, pc_val['label'].tolist())
pc_test_dataset = PyTorchDataset(pc_test_encodings, pc_test['label'].tolist())

building_train_dataset = PyTorchDataset(building_train_encodings, building_train['label'].tolist())
building_val_dataset = PyTorchDataset(building_val_encodings, building_val['label'].tolist())
building_test_dataset = PyTorchDataset(building_test_encodings, building_test['label'].tolist())

proxy_train_dataset = PyTorchDataset(proxy_train_encodings, proxy_train['label'].tolist())
proxy_val_dataset = PyTorchDataset(proxy_val_encodings, proxy_val['label'].tolist())
proxy_test_dataset = PyTorchDataset(proxy_test_encodings, proxy_test['label'].tolist())


# ### Prepare evaluation metrics

# In[ ]:


import evaluate


# In[ ]:


# Load accuracy, f1, precision, and recall metrics

accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')
precision = evaluate.load('precision')
recall = evaluate.load('recall')


# In[ ]:


# Define function to compute model metrics

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    results = {}
    results.update(accuracy.compute(predictions=predictions, references=labels))
    results.update(f1.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(precision.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(recall.compute(predictions=predictions, references=labels, average='weighted'))
    
    return results


# ### Handle class imbalance

# In[ ]:


from sklearn.utils import class_weight
from torch import nn
from transformers import Trainer


# In[ ]:


# Create class weights for imbalanced PC Access log data

pc_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=list(pc_label2case.keys()),
    y=pc_train['label']
)

print(pc_class_weights)


# In[ ]:


# Define custom PC model trainer to override the loss function

class CustomPCTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")
    weight = torch.tensor(pc_class_weights, dtype=torch.float, device=model.device)
    loss_fct = nn.CrossEntropyLoss(weight=weight)
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    
    return (loss, outputs) if return_outputs else loss


# In[ ]:


# Create class weights for imbalanced Building Access log data

building_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=list(building_label2case.keys()),
    y=building_train['label']
)

print(building_class_weights)


# In[ ]:


# Define custom Building model trainer to override the loss function

class CustomBuildingTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")
    weight = torch.tensor(building_class_weights, dtype=torch.float, device=model.device)
    loss_fct = nn.CrossEntropyLoss(weight=weight)
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    
    return (loss, outputs) if return_outputs else loss


# In[ ]:


# Create class weights for imbalanced Proxy log data

proxy_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=list(proxy_label2case.keys()),
    y=proxy_train['label']
)

print(proxy_class_weights)


# In[ ]:


# Define custom Proxy model trainer to override the loss function

class CustomProxyTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")
    weight = torch.tensor(proxy_class_weights, dtype=torch.float, device=model.device)
    loss_fct = nn.CrossEntropyLoss(weight=weight)
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    
    return (loss, outputs) if return_outputs else loss


# ### Build and train model

# In[ ]:


from transformers import AutoModelForSequenceClassification, TrainingArguments


# In[ ]:


# Load pretrained DistilBERT model

pc_model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(pc_cases),
    id2label=pc_label2case,
    label2id=pc_case2label
)
pc_model_with_custom_weights = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(pc_cases),
    id2label=pc_label2case,
    label2id=pc_case2label
)

building_model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(building_cases),
    id2label=building_label2case,
    label2id=building_case2label
)
building_model_with_custom_weights = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(building_cases),
    id2label=building_label2case,
    label2id=building_case2label
)

proxy_model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(proxy_cases),
    id2label=proxy_label2case,
    label2id=proxy_case2label
)
proxy_model_with_custom_weights = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(proxy_cases),
    id2label=proxy_label2case,
    label2id=proxy_case2label
)


# In[ ]:


# Set training hyperparameters

training_args = TrainingArguments(
    output_dir='training_output',
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


# Initilise PC model trainers

pc_trainer = Trainer(
    model=pc_model,
    args=training_args,
    train_dataset=pc_train_dataset,
    eval_dataset=pc_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

pc_trainer_with_custom_weights = CustomPCTrainer(
    model=pc_model_with_custom_weights,
    args=training_args,
    train_dataset=pc_train_dataset,
    eval_dataset=pc_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# In[ ]:


# Initialise Building model trainers

building_trainer = Trainer(
    model=building_model,
    args=training_args,
    train_dataset=building_train_dataset,
    eval_dataset=building_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

building_trainer_with_custom_weights = CustomBuildingTrainer(
    model=building_model_with_custom_weights,
    args=training_args,
    train_dataset=building_train_dataset,
    eval_dataset=building_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# In[ ]:


# Initialise Proxy model trainers

proxy_trainer = Trainer(
    model=proxy_model,
    args=training_args,
    train_dataset=proxy_train_dataset,
    eval_dataset=proxy_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

proxy_trainer_with_custom_weights = CustomProxyTrainer(
    model=proxy_model_with_custom_weights,
    args=training_args,
    train_dataset=proxy_train_dataset,
    eval_dataset=proxy_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# In[ ]:


# Train base PC model

pc_trainer.train()


# In[ ]:


# Train PC model with custom class weights

pc_trainer_with_custom_weights.train()


# In[ ]:


# Train base Building model

building_trainer.train()


# In[ ]:


# Train Building model with custom class weights

building_trainer_with_custom_weights.train()


# In[ ]:


# Train base Proxy model

proxy_trainer.train()


# In[ ]:


# Train PC model with custom class weights

proxy_trainer_with_custom_weights.train()


# ### Evaluate model

# In[ ]:


from evaluate import evaluator


# In[ ]:


# Initialise PC model trainers for evaluation

pc_eval_trainer = Trainer(
    model=pc_model,
    eval_dataset=pc_test_dataset,
    compute_metrics=compute_metrics
)

pc_eval_trainer_with_custom_weights = CustomPCTrainer(
    model=pc_model_with_custom_weights,
    eval_dataset=pc_test_dataset,
    compute_metrics=compute_metrics
)


# In[ ]:


# Initialise Building model trainers for evaluation

building_eval_trainer = Trainer(
    model=building_model,
    eval_dataset=building_test_dataset,
    compute_metrics=compute_metrics
)

building_eval_trainer_with_custom_weights = CustomBuildingTrainer(
    model=building_model_with_custom_weights,
    eval_dataset=building_test_dataset,
    compute_metrics=compute_metrics
)


# In[ ]:


# Initialise Proxy model trainers for evaluation

proxy_eval_trainer = Trainer(
    model=proxy_model,
    eval_dataset=proxy_test_dataset,
    compute_metrics=compute_metrics
)

proxy_eval_trainer_with_custom_weights = CustomProxyTrainer(
    model=proxy_model_with_custom_weights,
    eval_dataset=proxy_test_dataset,
    compute_metrics=compute_metrics
)


# In[ ]:


# Evaluate base PC model

print('Base PC model evaluation:')
pc_eval_trainer.evaluate()


# In[ ]:


# Evaluate PC model with custom class weights

print('PC model w/ custom weights evaluation:')
pc_eval_trainer_with_custom_weights.evaluate()


# In[ ]:


# Evaluate base Building model

print('Base Building model evaluation:')
building_eval_trainer.evaluate()


# In[ ]:


# Evaluate Building model with custom class weights

print('Building model w/ custom weights evaluation:')
building_eval_trainer_with_custom_weights.evaluate()


# In[ ]:


# Evaluate base Proxy model

print('Base Proxy model evaluation:')
proxy_eval_trainer.evaluate()


# In[ ]:


# Evaluate Proxy model with custom class weights

print('Proxy model w/ custom weights evaluation:')
proxy_eval_trainer_with_custom_weights.evaluate()


# In[ ]:


# Save model

pc_trainer_with_custom_weights.save_model(PC_MODEL_NAME)
building_trainer_with_custom_weights.save_model(BUILDING_MODEL_NAME)
proxy_trainer_with_custom_weights.save_model(PROXY_MODEL_NAME)


# In[ ]:


# Show PC model predictions on a few test inputs

pc_test_input = []

for case in pc_cases:
    pc_test_input += pc_test[pc_test['suspect'] == case]['input'].sample(n=3).tolist()

print('Sample PC Access log input:')
display(pc_test_input)

pc_saved_tokenizer = AutoTokenizer.from_pretrained(PC_MODEL_NAME)
pc_saved_model = AutoModelForSequenceClassification.from_pretrained(PC_MODEL_NAME)
pc_test_input = pc_saved_tokenizer(pc_test_input, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    pc_logits = pc_saved_model(**pc_test_input).logits

print()
print('PC model predictions:')

for logits in pc_logits:
    predicted_class_id = logits.argmax().item()
    print(pc_saved_model.config.id2label[predicted_class_id])


# In[ ]:


# Show Building model predictions on a few test inputs

building_test_input = []

for case in building_cases:
    building_test_input += building_test[building_test['suspect'] == case]['input'].sample(n=2).tolist()

print('Sample Building Access log input:')
display(building_test_input)

building_saved_tokenizer = AutoTokenizer.from_pretrained(BUILDING_MODEL_NAME)
building_saved_model = AutoModelForSequenceClassification.from_pretrained(BUILDING_MODEL_NAME)
building_test_input = building_saved_tokenizer(building_test_input, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    building_logits = building_saved_model(**building_test_input).logits

print()
print('Building model predictions:')

for logits in building_logits:
    predicted_class_id = logits.argmax().item()
    print(building_saved_model.config.id2label[predicted_class_id])


# In[ ]:


# Show Proxy model predictions on a few test inputs

proxy_test_input = []

for case in proxy_cases:
    proxy_test_input += proxy_test[proxy_test['suspect'] == case]['input'].sample(n=3).tolist()

print('Sample Proxy log input:')
display(proxy_test_input)

proxy_saved_tokenizer = AutoTokenizer.from_pretrained(PROXY_MODEL_NAME)
proxy_saved_model = AutoModelForSequenceClassification.from_pretrained(PROXY_MODEL_NAME)
proxy_test_input = proxy_saved_tokenizer(proxy_test_input, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    proxy_logits = proxy_saved_model(**proxy_test_input).logits

print()
print('Proxy model predictions:')

for logits in proxy_logits:
    predicted_class_id = logits.argmax().item()
    print(proxy_saved_model.config.id2label[predicted_class_id])

