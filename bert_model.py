from transformers import BertModel
from torch import nn

class BertMultiOutput(nn.Module):
    def __init__(self, num_cause_subjects, num_cause_states, num_effect_subjects, num_effect_states, dropout_rate=0.3):
        super(BertMultiOutput, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers for each output
        self.fc_cause_subject = nn.Linear(768, num_cause_subjects)
        self.fc_cause_state = nn.Linear(768, num_cause_states)
        self.fc_effect_subject = nn.Linear(768, num_effect_subjects)
        self.fc_effect_state = nn.Linear(768, num_effect_states)

    def forward(self, input_ids):
        # Get the pooled output from BERT
        outputs = self.bert(input_ids=input_ids)
        pooled_output = outputs[1]
        
        # Apply dropout to pooled output to prevent overfitting
        pooled_output = self.dropout(pooled_output)

        # Predictions for each output
        cause_subject = self.fc_cause_subject(pooled_output)
        cause_state = self.fc_cause_state(pooled_output)
        effect_subject = self.fc_effect_subject(pooled_output)
        effect_state = self.fc_effect_state(pooled_output)

        return cause_subject, cause_state, effect_subject, effect_state
