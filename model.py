import torch
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # For splitting dataset
from torch.cuda.amp import autocast, GradScaler  # For mixed precision

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sentence_tokens = ast.literal_eval(item['Sentence_Tokens'])
        
        return {
            'input_ids': torch.tensor(sentence_tokens, dtype=torch.long),
            'cause_subject': torch.tensor(item['Cause_Subject_Label'], dtype=torch.long),
            'cause_state': torch.tensor(item['Cause_State_Label'], dtype=torch.long),
            'effect_subject': torch.tensor(item['Effect_Subject_Label'], dtype=torch.long),
            'effect_state': torch.tensor(item['Effect_State_Label'], dtype=torch.long),
        }

# Custom collate function to handle padding
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    cause_subjects = torch.tensor([item['cause_subject'] for item in batch], dtype=torch.long)
    cause_states = torch.tensor([item['cause_state'] for item in batch], dtype=torch.long)
    effect_subjects = torch.tensor([item['effect_subject'] for item in batch], dtype=torch.long)
    effect_states = torch.tensor([item['effect_state'] for item in batch], dtype=torch.long)

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids_padded,
        'cause_subject': cause_subjects,
        'cause_state': cause_states,
        'effect_subject': effect_subjects,
        'effect_state': effect_states
    }

# BERT Model for Multi-Output Prediction
class BertMultiOutput(nn.Module):
    def __init__(self, num_cause_subjects, num_cause_states, num_effect_subjects, num_effect_states):
        super(BertMultiOutput, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc_cause_subject = nn.Linear(768, num_cause_subjects)
        self.fc_cause_state = nn.Linear(768, num_cause_states)
        self.fc_effect_subject = nn.Linear(768, num_effect_subjects)
        self.fc_effect_state = nn.Linear(768, num_effect_states)

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        pooled_output = outputs[1]
        cause_subject = self.fc_cause_subject(pooled_output)
        cause_state = self.fc_cause_state(pooled_output)
        effect_subject = self.fc_effect_subject(pooled_output)
        effect_state = self.fc_effect_state(pooled_output)
        return cause_subject, cause_state, effect_subject, effect_state

# Load preprocessed data
data = pd.read_csv('processed_data_v2_fixed.csv')

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2)  # 80% train, 20% validation
train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)

# Create the dataset
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Define the number of unique classes for each output
num_cause_subjects = len(data['Cause_Subject_Label'].unique())
num_cause_states = len(data['Cause_State_Label'].unique())
num_effect_subjects = len(data['Effect_Subject_Label'].unique())
num_effect_states = len(data['Effect_State_Label'].unique())

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertMultiOutput(num_cause_subjects, num_cause_states, num_effect_subjects, num_effect_states).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)  # Use AdamW optimizer

# Mixed precision training
scaler = GradScaler()

# Learning rate scheduler
total_steps = len(train_dataloader) * 10  # For 10 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop with validation and early stopping
num_epochs = 10
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        cause_subject = batch['cause_subject'].to(device)
        cause_state = batch['cause_state'].to(device)
        effect_subject = batch['effect_subject'].to(device)
        effect_state = batch['effect_state'].to(device)

        with autocast():  # Enables mixed precision
            cause_subject_pred, cause_state_pred, effect_subject_pred, effect_state_pred = model(input_ids)

            # Compute the loss for each output
            loss_cause_subject = criterion(cause_subject_pred, cause_subject)
            loss_cause_state = criterion(cause_state_pred, cause_state)
            loss_effect_subject = criterion(effect_subject_pred, effect_subject)
            loss_effect_state = criterion(effect_state_pred, effect_state)

            # Combine the losses
            loss = loss_cause_subject + loss_cause_state + loss_effect_subject + loss_effect_state

        # Backpropagation with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix({'Train Loss': total_train_loss / len(train_dataloader)})

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            cause_subject = batch['cause_subject'].to(device)
            cause_state = batch['cause_state'].to(device)
            effect_subject = batch['effect_subject'].to(device)
            effect_state = batch['effect_state'].to(device)

            with autocast():  # Mixed precision for validation
                cause_subject_pred, cause_state_pred, effect_subject_pred, effect_state_pred = model(input_ids)

                loss_cause_subject = criterion(cause_subject_pred, cause_subject)
                loss_cause_state = criterion(cause_state_pred, cause_state)
                loss_effect_subject = criterion(effect_subject_pred, effect_subject)
                loss_effect_state = criterion(effect_state_pred, effect_state)

                total_val_loss += loss_cause_subject + loss_cause_state + loss_effect_subject + loss_effect_state

    total_val_loss /= len(val_dataloader)
    print(f'Validation Loss: {total_val_loss}')

    # Early stopping logic
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), 'best_bert_multi_output_model.pth')  # Save the best model
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
