import torch
import gradio as gr
from transformers import BertTokenizer
from bert_model import BertMultiOutput  # Import your model
import pandas as pd

# Load the tokenizer and trained BERT-based model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the number of classes
data = pd.read_csv('processed_data_v2.csv')
num_cause_subjects = len(data['Cause_Subject_Label'].unique())
num_cause_states = len(data['Cause_State_Label'].unique())
num_effect_subjects = len(data['Effect_Subject_Label'].unique())
num_effect_states = len(data['Effect_State_Label'].unique())

# Load the trained model
model = BertMultiOutput(num_cause_subjects, num_cause_states, num_effect_subjects, num_effect_states)
model.load_state_dict(torch.load('best_bert_multi_output_model.pth'))  # Load the trained model weights
model.eval()  # Set model to evaluation mode

# Function to reconstruct phrases from subwords
def reconstruct_phrases(tokens, predicted_indices):
    phrases = []
    current_phrase = []
    
    for idx in predicted_indices:
        token = tokens[idx]
        if token.startswith("##"):
            # This token is a subword of the previous token, so we append it
            current_phrase[-1] += token[2:]  # Remove "##" and append
        else:
            if current_phrase:
                # Append the completed phrase to the list
                phrases.append(" ".join(current_phrase))
            current_phrase = [token]  # Start a new phrase

    if current_phrase:
        phrases.append(" ".join(current_phrase))  # Add the last phrase

    return " ".join(phrases)  # Combine all phrases into one meaningful string

# Prediction function for Gradio
def predict(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']

    # Also keep track of the tokenized words (for mapping later)
    tokenized_words = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():  # Disable gradient calculation for inference
        # Get the predictions from the model
        cause_subject, cause_state, effect_subject, effect_state = model(input_ids)

        # Convert logits to predicted labels using argmax
        pred_cause_subject = torch.argmax(cause_subject, dim=1).item()
        pred_cause_state = torch.argmax(cause_state, dim=1).item()
        pred_effect_subject = torch.argmax(effect_subject, dim=1).item()
        pred_effect_state = torch.argmax(effect_state, dim=1).item()

        # Debug: Print the predicted indices to ensure they are correct
        print(f"Predicted Cause Subject Index: {pred_cause_subject}")
        print(f"Predicted Cause State Index: {pred_cause_state}")
        print(f"Predicted Effect Subject Index: {pred_effect_subject}")
        print(f"Predicted Effect State Index: {pred_effect_state}")

        # Post-process to reconstruct meaningful phrases
        cause_subject_word = reconstruct_phrases(tokenized_words, [pred_cause_subject])
        cause_state_word = reconstruct_phrases(tokenized_words, [pred_cause_state])
        effect_subject_word = reconstruct_phrases(tokenized_words, [pred_effect_subject])
        effect_state_word = reconstruct_phrases(tokenized_words, [pred_effect_state])

    # Return the output as phrases from the entered sentence
    return {
        "Cause Subject": cause_subject_word,
        "Cause State": cause_state_word,
        "Effect Subject": effect_subject_word,
        "Effect State": effect_state_word,
    }

# Create Gradio interface
iface = gr.Interface(fn=predict, inputs="text", outputs="json", 
                     title="Cause-Effect Predictor",
                     description="Enter a sentence to see the predicted cause and effect from the sentence's tokens.")
iface.launch()
