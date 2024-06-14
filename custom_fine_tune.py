
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import random
from collections import defaultdict
from enhanced_hooking import get_blocks, clear_hooks, attach_activation_hooks
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle


#### Load the model
def load_model(model_path, device, center_weights=True):
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map=device_map, token="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")#, torch_dtype=torch.float16
    if center_weights:
        for name, param in model.named_parameters():
            if '.'.join(name.split('.')[-2:]) in ['wte.weight','wpe.weight','c_proj.weight','c_proj.bias']:
                param.data -= param.data.mean()
                print(name, param.data.mean(), param.size())
    tokenizer = AutoTokenizer.from_pretrained(model_path, token="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")
    model.tokenizer = tokenizer
    #model.tokenizer.padding_side = "left" #for batching; right (default in gpt2) for training, left for generation
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id 
    return model

model=None
import gc
gc.collect()
torch.cuda.empty_cache()
_ = torch.set_grad_enabled(False)
model_path: str = "meta-llama/Llama-2-13b-chat-hf" #even on an A40 I have to load 13b in half precision
device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu" #need to upgrade MacOS first
#device: str = "cuda" if torch.cuda.is_available() else "cpu"
center_weights=False

model = load_model(model_path, device, center_weights=center_weights)

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous epoch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    # plt.show()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune a GPT model for classification"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")
    )
    args = parser.parse_args()

    ########################################
    # Download and prepare dataset
    ########################################


### Load in data
outputdir = "./"
datadir = "./"
if "gpt" in model_path:
    user_tag, asst_tag = "|<end_of_text>|", ""
    model_numlayers = model.config.n_layer
else: #/llama/mistral
    user_tag, asst_tag = "[INST] ", " [/INST]"
    model_numlayers = model.config.num_hidden_layers

strip_quotes = lambda s: s[1:-1] if len(s) > 1 and (s.startswith('"') and s.endswith('"') or s.startswith("'") and s.endswith("'")) else s

pos_label = "A"
neg_label = "B"
df = pd.read_csv(datadir+"facts_true_false.csv") #train
true_statements = df[df['label'] == 1]['statement'].values.tolist()
false_statements = df[df['label'] == 0]['statement'].values.tolist()
true_statements_train = true_statements.sample(n=206, random_state=42)
true_statements_test = true_statements.drop(true_statements_train.index)
false_statements_train = false_statements.sample(n=206, random_state=42)
false_statements_test = false_statements.drop(false_statements_train.index)

def template(persona: str, stmt: str, label: str) -> str:
    return f"{user_tag}<<SYS>> Respond only with the letter A or B <<SYS>> Question: Is the following claim true or false? \"{strip_quotes(stmt)}\"\n\nChoices:\n (A) True\n (B) False\n\nAnswer:{asst_tag} "

train_prompts = []
train_labels = []
for i, (truth, lie) in enumerate(zip(true_statements_train.values.tolist(), false_statements_train.values.tolist())):
    train_prompts.append(template(truth))
    train_labels.append(pos_label)
    train_prompts.append(template(truth))
    train_labels.append(neg_label)
    train_prompts.append(template(lie))
    train_labels.append(neg_label)
    train_prompts.append(template(lie))
    train_labels.append(pos_label)

def collate_batch(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad input_ids and labels to the maximum length in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=model.tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Using -100 to ignore padding in loss calculation

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded
    }

class PromptCompletionDataset(Dataset):
    def __init__(self, prompts, completions, tokenizer):
        self.prompts = prompts
        self.completions = completions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt_text = self.prompts[idx]
        completion_text = self.completions[idx]
        
        # Tokenize prompt and completion together for input_ids
        encoded_pair = self.tokenizer(prompt_text + completion_text, return_tensors='pt')
        input_ids = encoded_pair.input_ids.squeeze(0)

        # Tokenize completion alone for labels, setting labels for prompt to -100
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        completion_ids = self.tokenizer(completion_text, add_special_tokens=False).input_ids
        labels = [-100] * len(prompt_ids) + completion_ids  # Ignore prompt tokens in loss calculation

        return {"input_ids": input_ids, "labels": torch.tensor(labels)}


model.tokenizer.padding_side = "right"
batch_size=32
dataset = PromptCompletionDataset(train_prompts, train_labels, model.tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)                        

layers_to_train = [14, 15, 16, 17, 18, 19]  
layer_prefix = 'transformer.h.' if "gpt2" in model_path else 'model.layers.'
for name, param in model.named_parameters():
    if name.startswith(layer_prefix) and int(name.split('.')[2]) not in layers_to_train:
        print(f"Freezing name={name}, layer={int(name.split('.')[2])}")
        param.requires_grad = False

train_direction_in = True
flip_direction = True
projection_weight = 1.0
num_epochs=3
fname = 'directions_llama213bchat_bnb_gpt4facts_persona_tf_lasttoken_pc2raw.pkl'
with open(outputdir+fname, 'rb') as f:
    directions = pickle.load(f)

if flip_direction: directions *= -1
directions.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)#, weight_decay=0.1
key_token_offset=1
priortoks=0
torch.manual_seed(123)

total_losses = []
token_losses = []
projection_losses = []
for epoch in range(num_epochs):
    model.train()
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch in tqdm(dataloader):
        last_token_positions = (batch['attention_mask'].sum(dim=1) - key_token_offset).tolist()
        layers_positions = {}
        for layer in layers_to_train:
            layers_positions[layer] = [[pos-i for i in range(priortoks,-1,-1)] for pos in last_token_positions]

        activation_storage = defaultdict(lambda: defaultdict(list))
        attach_activation_hooks(model, layers_positions, activation_storage, "end")

        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs, labels=inputs['labels'])
        loss = outputs.loss
        logits = outputs.logits[:, -1, :]  # Logits of last output token
        predicted_labels = torch.argmax(logits, dim=-1)

        cum_projection = 0
        for layer, positions in activation_storage.items():
            for pos, tensor in positions.items():#each of these is a list of batchsize d-embed tensors for a given position
                cum_projection += (tensor @ directions[layer].to(tensor.dtype)) / (torch.norm(directions[layer]) * torch.norm(tensor, dim=1))
             
        if train_direction_in: projection_loss = (1 - cum_projection / len(layers_to_train) / 2) # ranges from 0 for perfect alignment to 1 for perfect anti-alignment
        else: projection_loss = torch.abs(cum_projection / len(layers_to_train)) # 0 if no projection, 1 if perfect anti-alignment

        total_loss = loss + projection_loss * projection_weight
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for block in get_blocks(model): block._forward_hooks.clear()

        print(f"Prediction: {predicted_labels.tolist()}")
        print(f"Total Loss: {total_loss.item():.4f},Token Loss: {loss.item():.4f}, Projection loss: {projection_loss.item():.4f}")




    # loss plot
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # accuracy plot
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")