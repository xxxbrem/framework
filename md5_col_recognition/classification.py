import torch
import torch.nn as nn
from torchtext import data
import torchtext
from model import TransformerClassifier
from utils.utility import split_data

file_folder = './data/'
split_data(file_folder + 'data.csv')
train_name = file_folder + 'train.csv'
dev_name = file_folder + 'dev.csv'
test_name = file_folder + 'test.csv'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
batch_size = 32

# Field
LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True)
# dataset
train, dev, test = data.TabularDataset.splits(
    path = './', train=train_name, validation=dev_name, test=test_name, format='csv', skip_header=True,
    fields=[('sentence', TEXT), ('label', LABEL)]
)
# vocab
TEXT.build_vocab(train)
LABEL.build_vocab(train)
# iter
train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, dev, test), sort_key=lambda x: len(x.sentence),
    batch_size=batch_size,
)


epochs = 50
embeddings = nn.Embedding(len(TEXT.vocab), 50)
sentence = torch.tensor(list(range(len(TEXT.vocab))), dtype=int)
model = TransformerClassifier(
    # TEXT.vocab.vectors,
    embeddings(sentence),
    nhead=5, 
    dim_feedforward=50,  
    num_layers=6,
    dropout=0.0,
).to(device)

criterion = nn.CrossEntropyLoss()
lr = 1e-5
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=lr
)

torch.manual_seed(0)

print("starting")
for epoch in range(epochs):
    print(f"\n\nepoch: {epoch}\n")
    epoch_loss = 0
    epoch_correct = 0
    epoch_count = 0
    for idx, batch in enumerate(iter(train_iter)):
        predictions = model(batch.sentence.T.to(device))
        labels = batch.label.to(device)

        loss = criterion(predictions, labels)

        correct = predictions.argmax(axis=1) == labels
        acc = correct.sum().item() / correct.size(0)

        epoch_correct += correct.sum().item()
        epoch_count += correct.size(0)

        epoch_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

    with torch.no_grad():
        dev_epoch_loss = 0
        dev_epoch_correct = 0
        dev_epoch_count = 0

        for idx, batch in enumerate(iter(dev_iter)):
            predictions = model(batch.sentence.T.to(device))
            labels = batch.label.to(device)
            dev_loss = criterion(predictions, labels)

            correct = predictions.argmax(axis=1) == labels
            acc = correct.sum().item() / correct.size(0)

            dev_epoch_correct += correct.sum().item()
            dev_epoch_count += correct.size(0)
            dev_epoch_loss += loss.item()

    print(f"epoch_loss: {epoch_loss}")
    print(f"epoch accuracy: {epoch_correct / epoch_count}\n")
    print(f"dev_epoch_loss: {dev_epoch_loss}")
    print(f"dev epoch accuracy: {dev_epoch_correct / dev_epoch_count}\n")


    with torch.no_grad():
        test_epoch_loss = 0
        test_epoch_correct = 0
        test_epoch_count = 0

        for idx, batch in enumerate(iter(test_iter)):
            predictions = model(batch.sentence.T.to(device))
            labels = batch.label.to(device)
            test_loss = criterion(predictions, labels)

            correct = predictions.argmax(axis=1) == labels
            acc = correct.sum().item() / correct.size(0)

            test_epoch_correct += correct.sum().item()
            test_epoch_count += correct.size(0)
            test_epoch_loss += loss.item()

    print(f"test_epoch_loss: {test_epoch_loss}")
    print(f"test epoch accuracy: {test_epoch_correct / test_epoch_count}")
    if test_epoch_correct / test_epoch_count == 1:
        torch.save(model.state_dict, f'epoch{epoch}_state_dict.pth')