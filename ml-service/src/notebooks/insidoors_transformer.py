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
# - imbalanced-learn
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
# !pip install imbalanced-learn
# !pip install evaluate


# In[ ]:


# For SMU GPU cluster:
# !pip install sqlalchemy --no-build-isolation
# !pip install pandas --no-build-isolation
# !pip install matplotlib --no-build-isolation
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-build-isolation
# !pip install transformers[torch] --no-build-isolation
# !pip install scikit-learn --no-build-isolation
# !pip install imbalanced-learn --no-build-isolation
# !pip install evaluate --no-build-isolation


# In[ ]:


get_ipython().system('whichgpu')


# In[ ]:


# FUTURE ITERATIONS:
# 1. properly incorporate tabular data
# 2. add support for online learning


# ## Insidoors Text Classification Model for PC Access, Building Access, and Proxy Logs
# This notebook details the sixth iteration of a text classification model that identifies suspicious employee activity. For ease of development, all models are currently built and trained in this notebook. In the future, a separate notebook will be created for each model.
# 
# #### Changelog
# 
# *Version: 6 (Current)*
# * Re-implemented binary classification with unseen cases
# 
# *Version: 5*
# * Implemented undersampling and oversampling to better handle class imbalance
# 
# *Version: 4*
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
PC_MODEL_NAME = 'insidoors_pc_v6'
BUILDING_MODEL_NAME = 'insidoors_building_v6'
PROXY_MODEL_NAME = 'insidoors_proxy_v6'


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

employee_df = pd.read_csv('employee_data_5k.csv', sep=';', header=0)
pc_df = pd.read_csv('pc_data_5k_modified.csv', sep=';', header=0)
building_df = pd.read_csv('building_data_5k_modified.csv', sep=';', header=0)
proxy_df = pd.read_csv('proxy_data_5k_modified.csv', sep=';', header=0)

print('Employees:')
display(employee_df)

print('PC Access logs:')
display(pc_df)

print('Building Access logs:')
display(building_df)

print('Proxy logs:')
display(proxy_df)


# In[ ]:


# Add 'terminated' column to Employee dataframe

employee_df['terminated'] = np.where(employee_df['terminated_date'].notnull(), 'Y', 'N')


# In[ ]:


# Preparation for testing unseen cases
# Add 'attempts' column to Building Access dataframe

np.random.seed(480)
building_df['attempts'] = np.random.randint(1, 6, building_df.shape[0])


# In[ ]:


# Remove cases 2 and 5 from all logs

to_exclude = [2, 5]

pc_df = pc_df[~pc_df['suspect'].isin(to_exclude)]
building_df = building_df[~building_df['suspect'].isin(to_exclude)]
proxy_df = proxy_df[~proxy_df['suspect'].isin(to_exclude)]


# In[ ]:


# Inner join Employee dataframe with PC Access and Building Access dataframes

join_df_pc = employee_df[['id', 'location', 'terminated']].rename(
    columns={'location': 'user_location'}
)
join_df_building = employee_df[['id', 'terminated']]

pc_df = pc_df.join(join_df_pc.set_index('id'), on='user_id', how='inner')
building_df = building_df.join(join_df_building.set_index('id'), on='user_id', how='inner')


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


# Concatenate input columns of PC Access logs into a single string and create label column

pc_df = pc_df.astype(str)
pc_df['suspect'] = pc_df['suspect'].astype(int) # Keep 'suspect' column as int
pc_df['input'] = pc_df[['user_id', 'access_date_time', 'log_on_off', 'machine_name',
                        'machine_location', 'user_location', 'terminated']].agg(', '.join, axis=1)
pc_df['label'] = pc_df['suspect']

display(pc_df)


# In[ ]:


# Concatenate input columns of Building Access logs into a single string and create label column

building_df = building_df.astype(str)
building_df['suspect'] = building_df['suspect'].astype(int) # Keep 'suspect' column as int
building_df['input'] = building_df[['user_id', 'access_date_time', 'direction', 'status',
                                    'office_location', 'attempts', 'terminated']].agg(', '.join, axis=1)
building_df['label'] = building_df['suspect']

display(building_df)


# In[ ]:


# Concatenate input columns of Proxy logs into a single string and create label column

proxy_df = proxy_df.astype(str)
proxy_df['suspect'] = proxy_df['suspect'].astype(int) # Keep 'suspect' column as int
proxy_df['input'] = proxy_df[['user_id', 'access_date_time', 'machine_name',
                              'url', 'category', 'bytes_in', 'bytes_out']].agg(', '.join, axis=1)
proxy_df['label'] = proxy_df['suspect']

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


# ### Handle class imbalance by resampling data

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# In[ ]:


# Create new distribution for PC Access data
# Undersample majority class to 75% of original
# Let majority class take up 80% of the total resampled distribution
# Oversample minority class be evenly split among the remaining 20%

pc_dist = pc_train['suspect'].value_counts().to_dict()
print('Original distribution of PC Access logs:')
display(pc_dist)

undersample_pc_dist = pc_dist.copy()
pc_majority = int(undersample_pc_dist[0] * 0.75)
undersample_pc_dist[0] = pc_majority
print('Target undersampled distribution of PC Access logs:')
display(undersample_pc_dist)

oversample_pc_dist = undersample_pc_dist.copy()
pc_minority = int(pc_majority * 0.2) // (len(pc_cases) - 1)

for key in oversample_pc_dist.keys():
    if key != 0:
        oversample_pc_dist[key] = pc_minority

print('Target oversampled distribution of PC Access logs:')
display(oversample_pc_dist)


# In[ ]:


# Resample PC Access data

pc_undersample = RandomUnderSampler(sampling_strategy=undersample_pc_dist, random_state=480)
pc_oversample = RandomOverSampler(sampling_strategy=oversample_pc_dist, random_state=480)

pc_x, pc_y = pc_undersample.fit_resample(pc_train[['input']], pc_train['suspect'])
print('Distribution after undersampling:')
print(pc_y.value_counts())

pc_x, pc_y = pc_oversample.fit_resample(pc_x, pc_y)
print('Distribution after oversampling:')
print(pc_y.value_counts())


# In[ ]:


# Create new distribution for Building Access data
# Undersample majority class to 75% of original
# Let majority class take up 80% of the total resampled distribution
# Oversample minority class be evenly split among the remaining 20%

building_dist = building_train['suspect'].value_counts().to_dict()
print('Original distribution of Building Access logs:')
display(building_dist)

undersample_building_dist = building_dist.copy()
building_majority = int(undersample_building_dist[0] * 0.75)
undersample_building_dist[0] = building_majority
print('Target undersampled distribution of Building Access logs:')
display(undersample_building_dist)

oversample_building_dist = undersample_building_dist.copy()
building_minority = int(building_majority * 0.2) // (len(building_cases) - 1)

for key in oversample_building_dist.keys():
    if key != 0:
        oversample_building_dist[key] = building_minority

print('Target oversampled distribution of Building Access logs:')
display(oversample_building_dist)


# In[ ]:


# Resample Building Access data

building_undersample = RandomUnderSampler(sampling_strategy=undersample_building_dist, random_state=480)
building_oversample = RandomOverSampler(sampling_strategy=oversample_building_dist, random_state=480)

building_x, building_y = building_undersample.fit_resample(building_train[['input']], building_train['suspect'])
print('Distribution after undersampling:')
print(building_y.value_counts())

building_x, building_y = building_oversample.fit_resample(building_x, building_y)
print('Distribution after oversampling:')
print(building_y.value_counts())


# In[ ]:


# Create new distribution for Proxy data
# Undersample majority class to 75% of original
# Let majority class take up 80% of the total resampled distribution
# Oversample minority class to 20%

proxy_dist = proxy_train['suspect'].value_counts().to_dict()
print('Original distribution of Proxy logs:')
display(proxy_dist)

undersample_proxy_dist = proxy_dist.copy()
proxy_majority = int(undersample_proxy_dist[0] * 0.75)
undersample_proxy_dist[0] = proxy_majority
print('Target undersampled distribution of Proxy logs:')
display(undersample_proxy_dist)

oversample_proxy_dist = undersample_proxy_dist.copy()
proxy_minority = int(proxy_majority * 0.2)

for key in oversample_proxy_dist.keys():
    if key != 0:
        oversample_proxy_dist[key] = proxy_minority

print('Target oversampled distribution of Proxy logs:')
display(oversample_proxy_dist)


# In[ ]:


# Resample Proxy data

proxy_undersample = RandomUnderSampler(sampling_strategy=undersample_proxy_dist, random_state=480)
proxy_oversample = RandomOverSampler(sampling_strategy=oversample_proxy_dist, random_state=480)

proxy_x, proxy_y = proxy_undersample.fit_resample(proxy_train[['input']], proxy_train['suspect'])
print('Distribution after undersampling:')
print(proxy_y.value_counts())

proxy_x, proxy_y = proxy_oversample.fit_resample(proxy_x, proxy_y)
print('Distribution after oversampling:')
print(proxy_y.value_counts())


# ### Preprocess data (cont.)

# In[ ]:


# Combine suspect PC classes into single 'anomaly' class

pc_train.loc[pc_train['label'] != 0, 'label'] = 1
print('pc_train value counts:')
print(pc_train['label'].value_counts())

pc_y[pc_y != 0] = 1
print()
print('pc_y value counts:')
print(pc_y.value_counts())

pc_val.loc[pc_val['label'] != 0, 'label'] = 1
print()
print('pc_val value counts:')
print(pc_val['label'].value_counts())

pc_test.loc[pc_test['label'] != 0, 'label'] = 1
print()
print('pc_test value counts:')
print(pc_test['label'].value_counts())


# In[ ]:


# Combine suspect Building classes into single 'anomaly' class

building_train.loc[building_train['label'] != 0, 'label'] = 1
print('building_train value counts:')
print(building_train['label'].value_counts())

building_y[building_y != 0] = 1
print()
print('building_y value counts:')
print(building_y.value_counts())

building_val.loc[building_val['label'] != 0, 'label'] = 1
print()
print('building_val value counts:')
print(building_val['label'].value_counts())

building_test.loc[building_test['label'] != 0, 'label'] = 1
print()
print('building_test value counts:')
print(building_test['label'].value_counts())


# In[ ]:


# Combine suspect Proxy classes into single 'anomaly' class

proxy_train.loc[proxy_train['label'] != 0, 'label'] = 1
print('proxy_train value counts:')
print(proxy_train['label'].value_counts())

proxy_y[proxy_y != 0] = 1
print()
print('proxy_y value counts:')
print(proxy_y.value_counts())

proxy_val.loc[proxy_val['label'] != 0, 'label'] = 1
print()
print('proxy_val value counts:')
print(proxy_val['label'].value_counts())

proxy_test.loc[proxy_test['label'] != 0, 'label'] = 1
print()
print('proxy_test value counts:')
print(proxy_test['label'].value_counts())


# In[ ]:


# Tokenize input text using DistilBERT tokenizer

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

pc_train_encodings = tokenizer(pc_train['input'].tolist(), padding=True, truncation=True)
pc_train_encodings_balanced = tokenizer(pc_x['input'].tolist(), padding=True, truncation=True)
pc_val_encodings = tokenizer(pc_val['input'].tolist(), padding=True, truncation=True)
pc_test_encodings = tokenizer(pc_test['input'].tolist(), padding=True, truncation=True)

building_train_encodings = tokenizer(building_train['input'].tolist(), padding=True, truncation=True)
building_train_encodings_balanced = tokenizer(building_x['input'].tolist(), padding=True, truncation=True)
building_val_encodings = tokenizer(building_val['input'].tolist(), padding=True, truncation=True)
building_test_encodings = tokenizer(building_test['input'].tolist(), padding=True, truncation=True)

proxy_train_encodings = tokenizer(proxy_train['input'].tolist(), padding=True, truncation=True)
proxy_train_encodings_balanced = tokenizer(proxy_x['input'].tolist(), padding=True, truncation=True)
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
pc_train_dataset_balanced = PyTorchDataset(pc_train_encodings_balanced, pc_y.tolist())
pc_val_dataset = PyTorchDataset(pc_val_encodings, pc_val['label'].tolist())
pc_test_dataset = PyTorchDataset(pc_test_encodings, pc_test['label'].tolist())

building_train_dataset = PyTorchDataset(building_train_encodings, building_train['label'].tolist())
building_train_dataset_balanced = PyTorchDataset(building_train_encodings_balanced, building_y.tolist())
building_val_dataset = PyTorchDataset(building_val_encodings, building_val['label'].tolist())
building_test_dataset = PyTorchDataset(building_test_encodings, building_test['label'].tolist())

proxy_train_dataset = PyTorchDataset(proxy_train_encodings, proxy_train['label'].tolist())
proxy_train_dataset_balanced = PyTorchDataset(proxy_train_encodings_balanced, proxy_y.tolist())
proxy_val_dataset = PyTorchDataset(proxy_val_encodings, proxy_val['label'].tolist())
proxy_test_dataset = PyTorchDataset(proxy_test_encodings, proxy_test['label'].tolist())


# ### Prepare evaluation metrics

# In[ ]:


import evaluate
from scipy.special import softmax


# In[ ]:


# Load accuracy, f1, precision, recall, and roc_auc metrics

accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')
precision = evaluate.load('precision')
recall = evaluate.load('recall')
roc_auc = evaluate.load('roc_auc')


# In[ ]:


# Define function to compute model metrics

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    prediction_scores = softmax(predictions, axis=1)
    prediction_scores = np.max(prediction_scores, axis=1)
    predictions = np.argmax(predictions, axis=1)

    results = {}
    results.update(accuracy.compute(predictions=predictions, references=labels))
    results.update(f1.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(precision.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(recall.compute(predictions=predictions, references=labels, average='weighted'))
    results.update(roc_auc.compute(prediction_scores=prediction_scores, references=labels, average='weighted'))
    
    return results


# ### Handle class imbalance by adjusting class weights

# In[ ]:


from sklearn.utils import class_weight
from torch import nn
from transformers import Trainer


# In[ ]:


# Create class weights for imbalanced PC Access log data

pc_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=list(np.unique(pc_y)),
    y=pc_y
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
    classes=list(np.unique(building_y)),
    y=building_y
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
    classes=list(np.unique(proxy_y)),
    y=proxy_y
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

pc_model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)
pc_model_balanced = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)

building_model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)
building_model_balanced = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)

proxy_model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)
proxy_model_balanced = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)


# In[ ]:


# Set training hyperparameters

training_args = TrainingArguments(
    output_dir='training_output',
    learning_rate=0.00005,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
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

pc_trainer_balanced = CustomPCTrainer(
    model=pc_model_balanced,
    args=training_args,
    train_dataset=pc_train_dataset_balanced,
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

building_trainer_balanced = CustomBuildingTrainer(
    model=building_model_balanced,
    args=training_args,
    train_dataset=building_train_dataset_balanced,
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

proxy_trainer_balanced = CustomProxyTrainer(
    model=proxy_model_balanced,
    args=training_args,
    train_dataset=proxy_train_dataset_balanced,
    eval_dataset=proxy_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# In[ ]:


# Train base PC model

pc_trainer.train()


# In[ ]:


# Train balanced PC model

pc_trainer_balanced.train()


# In[ ]:


# Train base Building model

building_trainer.train()


# In[ ]:


# Train balanced Building model

building_trainer_balanced.train()


# In[ ]:


# Train base Proxy model

proxy_trainer.train()


# In[ ]:


# Train balanced Proxy model

proxy_trainer_balanced.train()


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

pc_eval_trainer_balanced = CustomPCTrainer(
    model=pc_model_balanced,
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

building_eval_trainer_balanced = CustomBuildingTrainer(
    model=building_model_balanced,
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

proxy_eval_trainer_balanced = CustomProxyTrainer(
    model=proxy_model_balanced,
    eval_dataset=proxy_test_dataset,
    compute_metrics=compute_metrics
)


# In[ ]:


# Evaluate base PC model

print('Base PC model evaluation:')
pc_eval_trainer.evaluate()


# In[ ]:


# Evaluate balanced PC model

print('Balanced PC model evaluation:')
pc_eval_trainer_balanced.evaluate()


# In[ ]:


# Evaluate base Building model

print('Base Building model evaluation:')
building_eval_trainer.evaluate()


# In[ ]:


# Evaluate balanced Building model

print('Balanced Building model evaluation:')
building_eval_trainer_balanced.evaluate()


# In[ ]:


# Evaluate base Proxy model

print('Base Proxy model evaluation:')
proxy_eval_trainer.evaluate()


# In[ ]:


# Evaluate balanced Proxy model

print('Balanced Proxy model evaluation:')
proxy_eval_trainer_balanced.evaluate()


# In[ ]:


# Save model

pc_trainer_balanced.save_model(PC_MODEL_NAME)
building_trainer_balanced.save_model(BUILDING_MODEL_NAME)
proxy_trainer_balanced.save_model(PROXY_MODEL_NAME)


# In[ ]:


# Show PC model predictions on a few test inputs

pc_test_input = []
pc_expected_output = []

for case in pc_cases:
    pc_test_input += pc_test[pc_test['suspect'] == case]['input'].sample(n=3).tolist()

    if case != 0:
        pc_expected_output += [1] * 3
    else:
        pc_expected_output += [case] * 3

print('Sample PC Access log input:')
display(pc_test_input)

print()
print('Expected output:')

for item in pc_expected_output:
    print(item)

pc_saved_tokenizer = AutoTokenizer.from_pretrained(PC_MODEL_NAME)
pc_saved_model = AutoModelForSequenceClassification.from_pretrained(PC_MODEL_NAME)
pc_test_input = pc_saved_tokenizer(pc_test_input, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    pc_logits = pc_saved_model(**pc_test_input).logits

print()
print('PC model predictions:')

for logits in pc_logits:
    print(logits.argmax().item())


# In[ ]:


# Show Building model predictions on a few test inputs

building_test_input = []
building_expected_output = []

for case in building_cases:
    building_test_input += building_test[building_test['suspect'] == case]['input'].sample(n=3).tolist()
    
    if case != 0:
        building_expected_output += [1] * 3
    else:
        building_expected_output += [case] * 3

print('Sample Building Access log input:')
display(building_test_input)

print()
print('Expected output:')

for item in building_expected_output:
    print(item)

building_saved_tokenizer = AutoTokenizer.from_pretrained(BUILDING_MODEL_NAME)
building_saved_model = AutoModelForSequenceClassification.from_pretrained(BUILDING_MODEL_NAME)
building_test_input = building_saved_tokenizer(building_test_input, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    building_logits = building_saved_model(**building_test_input).logits

print()
print('Building model predictions:')

for logits in building_logits:
    print(logits.argmax().item())


# In[ ]:


# Show Proxy model predictions on a few test inputs

proxy_test_input = []
proxy_expected_output = []

for case in proxy_cases:
    proxy_test_input += proxy_test[proxy_test['suspect'] == case]['input'].sample(n=3).tolist()
    
    if case != 0:
        proxy_expected_output += [1] * 3
    else:
        proxy_expected_output += [case] * 3

print('Sample Proxy log input:')
display(proxy_test_input)

print()
print('Expected output:')

for item in proxy_expected_output:
    print(item)

proxy_saved_tokenizer = AutoTokenizer.from_pretrained(PROXY_MODEL_NAME)
proxy_saved_model = AutoModelForSequenceClassification.from_pretrained(PROXY_MODEL_NAME)
proxy_test_input = proxy_saved_tokenizer(proxy_test_input, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    proxy_logits = proxy_saved_model(**proxy_test_input).logits

print()
print('Proxy model predictions:')

for logits in proxy_logits:
    print(logits.argmax().item())


# ### Test model predictions on unseen cases

# In[ ]:


from collections import Counter


# In[ ]:


# Create new data for PC Access logs
# New case: machine location different from user location

countries = ['Russia', 'China', 'India', 'Brazil']

pc_unseen_df = pc_test[pc_test['suspect'] == 0].copy()

np.random.seed(480)
pc_unseen_df['machine_location'] = np.random.choice(countries, pc_unseen_df.shape[0])
pc_unseen_df['input'] = pc_unseen_df[['user_id', 'access_date_time', 'log_on_off', 'machine_name',
                                      'machine_location', 'user_location', 'terminated']].agg(', '.join, axis=1)

display(pc_unseen_df)


# In[ ]:


# Create new data for Building Access logs
# New case: large number of access attempts

building_unseen_df = building_test[building_test['suspect'] == 0].copy()

np.random.seed(480)
building_unseen_df['attempts'] = np.random.randint(6, 20, building_unseen_df.shape[0])
building_unseen_df['attempts'] = building_unseen_df['attempts'].astype(str)
building_unseen_df['input'] = building_unseen_df[['user_id', 'access_date_time', 'direction', 'status',
                                                  'office_location', 'attempts', 'terminated']].agg(', '.join, axis=1)

display(building_unseen_df)


# In[ ]:


# Create new data for Proxy logs
# New case: data upload/download from malicious urls

url_types = ['malware', 'phishing']

malicious_urls_df = pd.read_csv('malicious_urls.csv', sep=',', header=0)
malicious_urls_df = malicious_urls_df[malicious_urls_df['type'].isin(url_types)]

display(malicious_urls_df)

urls = malicious_urls_df['url']

proxy_unseen_df = proxy_test[proxy_test['suspect'] == 0].copy()

np.random.seed(480)
proxy_unseen_df['url'] = np.random.choice(urls, proxy_unseen_df.shape[0])
proxy_unseen_df['category'] = 'Malware, Phishing'
proxy_unseen_df['input'] = proxy_unseen_df[['user_id', 'access_date_time', 'machine_name',
                                            'url', 'category', 'bytes_in', 'bytes_out']].agg(', '.join, axis=1)

display(proxy_unseen_df)


# In[ ]:


# Compute accuracy of PC model on unseen case

pc_test_input_unseen = pc_saved_tokenizer(pc_unseen_df['input'].sample(n=1000, random_state=480).tolist(),
                                          padding=True,
                                          truncation=True,
                                          return_tensors="pt")

pc_results = []

with torch.no_grad():
    pc_logits_unseen = pc_saved_model(**pc_test_input_unseen).logits

for logits in pc_logits_unseen:
    pc_results.append(logits.argmax().item())

pc_results_counter = Counter(pc_results)

print('PC model accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter[1], len(pc_results), pc_results_counter[1]/len(pc_results)*100))


# In[ ]:


# Compute accuracy of Building model on unseen case

building_test_input_unseen = building_saved_tokenizer(building_unseen_df['input'].sample(n=1000, random_state=480).tolist(),
                                                      padding=True,
                                                      truncation=True,
                                                      return_tensors="pt")

building_results = []

with torch.no_grad():
    building_logits_unseen = building_saved_model(**building_test_input_unseen).logits

for logits in building_logits_unseen:
    building_results.append(logits.argmax().item())

building_results_counter = Counter(building_results)

print('Building model accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter[1], len(building_results), building_results_counter[1]/len(building_results)*100))


# In[ ]:


# Compute accuracy of Proxy model on unseen case

proxy_test_input_unseen = proxy_saved_tokenizer(proxy_unseen_df['input'].sample(n=1000, random_state=480).tolist(),
                                                padding=True,
                                                truncation=True,
                                                return_tensors="pt")

proxy_results = []

with torch.no_grad():
    proxy_logits_unseen = proxy_saved_model(**proxy_test_input_unseen).logits

for logits in proxy_logits_unseen:
    proxy_results.append(logits.argmax().item())

proxy_results_counter = Counter(proxy_results)

print('Proxy model accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter[1], len(proxy_results), proxy_results_counter[1]/len(proxy_results)*100))

