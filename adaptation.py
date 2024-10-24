import pandas as pd

# Load the file with the correct tab delimiter
train_2017_df = pd.read_csv('./data/semeval-2017-train.csv', delimiter='\t')

# Display the first few rows to check the structure
print(train_2017_df.head())

# Define a function to adapt each row to the desired structure
def adapt_row(row):
    text = row['text']  # Assuming 'text' is the column with the sentence
    label = row['label']  # 'label' might be used as cause/effect classification

    # For this example, we'll treat the label as the cause_state
    cause_subject = "The event"  # Placeholder, as there's no clear 'cause' column
    cause_state = "causes" if label == 1 else "does not cause"
    effect_subject = "The outcome"  # Placeholder
    effect_state = text  # Use the text as a description of the effect

    # Create the sentence
    sentence = f"{cause_subject} {cause_state} {effect_subject}: {effect_state}"

    return sentence, cause_subject, cause_state, effect_subject, effect_state

# Apply the adaptation function to each row
adapted_data = train_2017_df.apply(adapt_row, axis=1)

# Convert the adapted data to a DataFrame
adapted_df = pd.DataFrame(adapted_data.tolist(), columns=['Sentence', 'Cause_Subject', 'Cause_State', 'Effect_Subject', 'Effect_State'])

# Display the adapted dataset
print(adapted_df.head())

# Save the adapted dataset to a CSV file
adapted_df.to_csv('adapted_semeval_2017.csv', index=False)
