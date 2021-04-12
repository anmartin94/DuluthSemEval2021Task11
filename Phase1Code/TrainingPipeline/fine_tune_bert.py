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
    dataframe = pd.read_csv(data_file_path, delimiter=',', header=None, names=['label', 'sentence'])
    sentences = dataframe.sentence.values
    labels = dataframe.label.values
    deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    encoded_sentences = []
    for sentence in sentences:
        encoded = deberta_tokenizer.encode(sentence, add_special_tokens = True)
        encoded_sentences.append(encoded)
    encoded_sentences = pad_sequences(encoded_sentences, 128, dtype="long", value=0, truncating="post", padding="post")
    masks = []
    for sentence in encoded_sentences:
        mask = []
        for id in sentence:
            if id > 0:
                id = 1
            mask.append(id)
        masks.append(mask)
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(encoded_sentences, labels,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(masks, labels,
                                                           random_state=2018, test_size=0.1)
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=32)
    return train_dataloader, validation_dataloader, labels

def train_model(train_dataloader, validation_dataloader, labels):
    model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base',
                                                             num_labels=13,
                                                             output_attentions=False,
                                                             output_hidden_states=False)
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-6)

    epochs = 4
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    average_losses = []
    for epoch in range(0, epochs):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_ids = batch[0].to(device)
            batch_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            model.zero_grad()
            outputs = model(batch_ids,
                            token_type_ids=None,
                            attention_mask=batch_mask,
                            labels=batch_labels)
            loss = outputs[0]
            total_loss = total_loss + loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)
            average_losses.append(avg_train_loss)
            print("Average training loss:", avg_train_loss)

            model.eval()
            eval_acc = 0
            eval_steps = 0
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                batch_ids, batch_mask, batch_labels = batch
                with torch.no_grad():
                    outputs = model(batch_ids,
                                    token_type_ids=None,
                                    attention_mask=batch_mask)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = batch_labels.to('cpu').numpy()
                predictions = np.argmax(logits, axis=1).flatten()
                flat_labels = labels.flatten()
                temp_eval_acc = np.sum(predictions == flat_labels) / len(flat_labels)
                eval_acc = eval_acc + temp_eval_acc
                eval_steps = eval_steps + 1
                total_acc = eval_acc / eval_steps
                print("  Accuracy:", total_acc)
    torch.save(model, "../DataFiles/bert_model")

def main():
    get_device()
    data_file_path = "../DataFiles/labeled_data.csv"
    train_dl, val_dl, labels = get_data(data_file_path)
    train_model(train_dl, val_dl, labels)






