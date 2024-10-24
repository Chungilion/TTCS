import torch
from sklearn.metrics import accuracy_score
from data_process import pd
from model import CustomDataset, DataLoader, BertMultiOutput, collate_fn  # Import collate_fn

# Load the evaluation dataset
test_data = pd.read_csv('processed_test_data_v2.csv')
test_dataset = CustomDataset(test_data)

# Create DataLoader for evaluation with collate_fn to pad the sequences
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Define the number of unique classes for each output (based on training data)
num_cause_subjects = len(test_data['Cause_Subject_Label'].unique())
num_cause_states = len(test_data['Cause_State_Label'].unique())
num_effect_subjects = len(test_data['Effect_Subject_Label'].unique())
num_effect_states = len(test_data['Effect_State_Label'].unique())

# Instantiate model with the correct number of classes
model = BertMultiOutput(num_cause_subjects, num_cause_states, num_effect_subjects, num_effect_states)
model.load_state_dict(torch.load('bert_multi_output_model.pth'))  # Load trained model
model.eval()  # Set model to evaluation mode

# Containers for predictions and actual labels
pred_cause_subjects = []
pred_cause_states = []
pred_effect_subjects = []
pred_effect_states = []

true_cause_subjects = []
true_cause_states = []
true_effect_subjects = []
true_effect_states = []

# Evaluation loop
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        cause_subject, cause_state, effect_subject, effect_state = model(input_ids)

        # Get predicted labels
        _, pred_cause_subject = torch.max(cause_subject, 1)
        _, pred_cause_state = torch.max(cause_state, 1)
        _, pred_effect_subject = torch.max(effect_subject, 1)
        _, pred_effect_state = torch.max(effect_state, 1)

        # Append predictions
        pred_cause_subjects.extend(pred_cause_subject.tolist())
        pred_cause_states.extend(pred_cause_state.tolist())
        pred_effect_subjects.extend(pred_effect_subject.tolist())
        pred_effect_states.extend(pred_effect_state.tolist())

        # Append true labels
        true_cause_subjects.extend(batch['cause_subject'].tolist())
        true_cause_states.extend(batch['cause_state'].tolist())
        true_effect_subjects.extend(batch['effect_subject'].tolist())
        true_effect_states.extend(batch['effect_state'].tolist())

# Calculate accuracy for each output
accuracy_cause_subject = accuracy_score(true_cause_subjects, pred_cause_subjects)
accuracy_cause_state = accuracy_score(true_cause_states, pred_cause_states)
accuracy_effect_subject = accuracy_score(true_effect_subjects, pred_effect_subjects)
accuracy_effect_state = accuracy_score(true_effect_states, pred_effect_states)

# Print accuracies for each output
print(f'Accuracy for Cause Subject: {accuracy_cause_subject}')
print(f'Accuracy for Cause State: {accuracy_cause_state}')
print(f'Accuracy for Effect Subject: {accuracy_effect_subject}')
print(f'Accuracy for Effect State: {accuracy_effect_state}')
