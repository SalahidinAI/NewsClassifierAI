import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from fastapi import FastAPI
from pydantic import BaseModel
from torchtext.data import get_tokenizer
import uvicorn


class CheckNews(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.lin = nn.Linear(128, 4)

    def forward(self, x):
        x = self.emb(x)
        _, (x, _) = self.lstm(x)
        x = self.lin(x.squeeze(0))
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab = torch.load('vocab.pth', weights_only=False)
classes = ['World', 'Sports', 'Business', 'Sci/Tech']

model = CheckNews(len(vocab)).to(device)
model.load_state_dict(torch.load('news_model.pth', map_location=device))
model.eval()

text_app = FastAPI(title='Text')


class TextIn(BaseModel):
    text: str

tokenizer = get_tokenizer('basic_english')

def preprocess(text: str):
    tokens = tokenizer(text)
    ids = [vocab[i] for i in tokens]
    tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    return tensor

@text_app.post('/predict')
async def predict(item: TextIn):
    x = preprocess(item.text)
    with torch.no_grad():
        pred = model(x)
        label = torch.argmax(pred, dim=1).item()
    return {'Label': classes[label - 1]}


if __name__ == '__main__':
    uvicorn.run(text_app, host='127.0.0.1', port=8000)


