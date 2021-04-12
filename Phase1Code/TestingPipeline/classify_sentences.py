import torch
import pandas as pd
from transformers import DebertaTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DebertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences

device = None

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

def get_data(data_file_path):
    # Load the dataset into a pandas dataframe.
    df = pd.read_csv(data_file_path, delimiter=',', header=None, names=['path', 'sentenceindex', 'sentence'])

    # Report the number of sentences.
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))

    # Create sentence and label lists
    sentences = df.sentence.values
    paths = df.path.values
    indices = df.sentenceindex.values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = deberta_tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )

        input_ids.append(encoded_sent)

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=128,
                              dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    # prediction_labels = torch.tensor(labels)

    # Set the batch size.
    batch_size = 32

    # Create the DataLoader.
    # prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    return prediction_inputs, prediction_dataloader, paths, indices, sentences

def get_predictions(prediction_inputs, prediction_dataloader, paths, indices, sentences):

    # Prediction on test set

    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

    # Put model in evaluation mode

    model = torch.load("../DataFiles/bert_model")
    model.eval()

    # Tracking variables
    # predictions , true_labels = [], []
    predictions = []

    # Predict

    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        # b_input_ids, b_input_mask, b_labels = batch
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        predictions.append(logits)
        # true_labels.append(label_ids)






    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    #flat_true_labels = [item for sublist in true_labels for item in sublist]
    data = {}
    for i in range(len(flat_predictions)):
        if flat_predictions[i] != "0":
            if data.get(paths[i]) == None:
                data[paths[i]] = {indices[i]:[flat_predictions[i], sentences[i]]}
            else:
                data[paths[i]][indices[i]] = [flat_predictions[i], sentences[i]]
    return data

def main():
    data_file_path = "../DataFiles/test-data.csv"
    get_device()
    pred_inputs, pred_dl, paths, indices, sentences = get_data(data_file_path)
    return get_predictions(pred_inputs, pred_dl, paths, indices, sentences)
