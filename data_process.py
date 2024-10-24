import pandas as pd
from transformers import BertTokenizer
import ast

# Load the preprocessed data
data = pd.read_csv('adapted_semeval_2017.csv')

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the sentences
data['Sentence_Tokens'] = data['Sentence'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True))

# Ensure labels start from 0
cause_subject_labels = {label: idx for idx, label in enumerate(data['Cause_Subject'].unique(), start=0)}
cause_state_labels = {label: idx for idx, label in enumerate(data['Cause_State'].unique(), start=0)}
effect_subject_labels = {label: idx for idx, label in enumerate(data['Effect_Subject'].unique(), start=0)}
effect_state_labels = {label: idx for idx, label in enumerate(data['Effect_State'].unique(), start=0)}

# Apply the mappings to the dataset
data['Cause_Subject_Label'] = data['Cause_Subject'].map(cause_subject_labels).fillna(0).astype(int)
data['Cause_State_Label'] = data['Cause_State'].map(cause_state_labels).fillna(0).astype(int)
data['Effect_Subject_Label'] = data['Effect_Subject'].map(effect_subject_labels).fillna(0).astype(int)
data['Effect_State_Label'] = data['Effect_State'].map(effect_state_labels).fillna(0).astype(int)

# Create reverse mappings for interpretation (useful for decoding predictions later)
reverse_cause_subject_labels = {v: k for k, v in cause_subject_labels.items()}
reverse_cause_state_labels = {v: k for k, v in cause_state_labels.items()}
reverse_effect_subject_labels = {v: k for k, v in effect_subject_labels.items()}
reverse_effect_state_labels = {v: k for k, v in effect_state_labels.items()}

# Save the processed data (tokenized sentences and labels)
data[['Sentence_Tokens', 'Cause_Subject_Label', 'Cause_State_Label', 'Effect_Subject_Label', 'Effect_State_Label']].to_csv('processed_data_v2_fixed.csv', index=False)

# Save the reverse mappings for later use
mappings = {
    'reverse_cause_subject_labels': reverse_cause_subject_labels,
    'reverse_cause_state_labels': reverse_cause_state_labels,
    'reverse_effect_subject_labels': reverse_effect_subject_labels,
    'reverse_effect_state_labels': reverse_effect_state_labels
}

# Save the mappings to a pickle file (to load later for label decoding)
pd.to_pickle(mappings, 'label_mappings_fixed.pkl')

print("Tokenization and label processing completed. Processed data and mappings saved.")
