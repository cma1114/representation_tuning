import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

##adapted from https://github.com/timoschick/self-debiasing/blob/main/perplexity.py
def compute_loss(model, input_ids: torch.LongTensor, labels: torch.LongTensor) -> torch.Tensor:
        outputs = model(input_ids, labels=labels)
        lm_logits = outputs[1]

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

def compute_ppl(model, tokenizer, batch_size=4, max_length=1024, stride=512):
    device = model.device
    model.eval()

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    batch_len = len(encodings.input_ids[0]) // batch_size

    # Trim the tensor to fit complete sequences only
    input_tensor = encodings.input_ids[0][:batch_len * batch_size]

    # Reshape into batches
    batched_inputs = input_tensor.view(batch_size, -1)

    lls = []
    ppl = None
    for i in tqdm(range(0, batched_inputs.size(1), stride)):
    # iterates through all 287644 tokens in wikitext test in windows of max_length tokens, in steps of size stride, 
    # with the first max_length - stride tokens masked, as they are context for each new stride tokens and have alread been counted in loss
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, batched_inputs.size(1))
        trg_len = end_loc - i  # stride, but may be different from stride on last loop

        input_ids = batched_inputs[:, begin_loc:end_loc].to(device)

        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            loss = compute_loss(model, input_ids, labels=target_ids)

            log_likelihood = loss * trg_len

        lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        print(f"Perplexity after {i} tokens: {ppl}")

    print(f"Final perplexity: {ppl}")
    return ppl

if __name__ == '__main__':
    HF_TOKEN=os.getenv('HF_TOKEN')
    _ = torch.set_grad_enabled(False)
    base_model_path: str = "gpt2"#meta-llama/Meta-Llama-3-8B-Instruct"#"meta-llama/Llama-2-13b-chat-hf" 
    model_path=base_model_path
    ###model_path="cackerman/llama2_13b_chat_projection_tune_neg_in"
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu" #need to upgrade MacOS first
    #device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path)#, torch_dtype=torch.float16, token=HF_TOKEN).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=HF_TOKEN)
    tokenizer.pad_token_id = tokenizer.eos_token_id 

    compute_ppl(model, tokenizer, batch_size=2, max_length=1024, stride=512)