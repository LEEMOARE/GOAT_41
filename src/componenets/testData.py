from label_mapping import label_mapping
import pandas as pd
from NIH import NIH
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv("C:\\Users\\gjaischool\\Desktop\\goat_41\\src\\componenets\\nih.csv")

# Get a list of unique patient IDs
patient_ids = dataset['Patient ID'].unique()

# Split the patient IDs into training and validation sets
train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

# Split the dataframe into training and validation sets
train_df = dataset[dataset['Patient ID'].isin(train_ids)]
val_df = dataset[dataset['Patient ID'].isin(val_ids)]

# Create training and validation datasets
train_dataset = NIH(train_df, root_dir="C:\\Users\\gjaischool\\Desktop\\archive(1)\\image", label_mapping=label_mapping, split='train')
val_dataset = NIH(val_df, root_dir="C:\\Users\\gjaischool\\Desktop\\archive(1)\\image", label_mapping=label_mapping, split='val')

print(train_dataset[5])
print(val_dataset[50])