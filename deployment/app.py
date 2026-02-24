import gradio as gr
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
from huggingface_hub import hf_hub_download

class NestedNER(pl.LightningModule):
    def __init__(self, num_labels, id2label=None, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.id2label = id2label

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        batch_size, seq_len, hidden_size = sequence_output.shape
        start_rep = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_rep = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_grid = torch.cat([start_rep, end_rep], dim=-1)
        logits = self.classifier(span_grid)
        return logits.permute(0, 3, 1, 2)

repo_id = "epikadith/nest-ner-bert-genia"
state_dict_path = hf_hub_download(repo_id=repo_id, filename="model_state.bin")
id2label_path = hf_hub_download(repo_id=repo_id, filename="id2label.json")

with open(id2label_path, "r") as f:
    id2label_str = json.load(f)
    loaded_id2label = {int(k): v for k, v in id2label_str.items()}

tokenizer = BertTokenizer.from_pretrained(repo_id)

model = NestedNER(
    num_labels=len(loaded_id2label),
    id2label=loaded_id2label
)
model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
model.eval()

def predict_nested_ner_tuned(text, threshold=-1.5):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)
    offsets = encoding['offset_mapping'][0].tolist()

    with torch.no_grad():
        logits = model(input_ids, attention_mask)[0]

    non_o_logits = logits[1:, :, :]
    max_non_o_logits, max_non_o_indices = torch.max(non_o_logits, dim=0)
    o_logits = logits[0, :, :]

    spans = []
    seq_len = input_ids.shape[1]
    for start in range(seq_len):
        for end in range(start, seq_len):
            if (max_non_o_logits[start, end] - o_logits[start, end]) > threshold:
                label_id = max_non_o_indices[start, end].item() + 1
                label = model.id2label[label_id] if model.id2label else label_id
                spans.append((start, end, label))

    results = []
    for start_idx, end_idx, label in spans:
        if start_idx >= len(offsets) or end_idx >= len(offsets):
            continue
        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx][1]
        if start_char == 0 and end_char == 0:
            continue
        entity_text = text[start_char:end_char]
        results.append((entity_text, label))

    return results

demo = gr.Interface(
    fn=predict_nested_ner_tuned,
    inputs=gr.Textbox(lines=3, placeholder="Enter biological text here..."),
    outputs=gr.HighlightedText(),
    examples=[
        ["We evaluated CIITA mRNA levels in human T cells."],
        ["The glucocorticoid receptor gene promoter was analyzed."],
        ["The CD28 surface receptor interacts with the NF-kappa B complex."],
        ["Expression of the human IL-2 gene is activated by the T cell receptor."],
        ["EBV-transformed B cell lines were used to study the CD40 ligand."]
    ],
    title="Nested NER with BERT",
    description="Identifies overlapping biological entities (DNA, RNA, protein, cell_line, cell_type)."
)

if __name__ == "__main__":
    demo.launch()