# Representation Tuning

## Internalizing steering vectors

Full writeup of this work can be found on [this Notion site](https://granite-cupboard-1c8.notion.site/Representation-Tuning-1d8f8678d4eb49d48c7766aa21133872?pvs=4)

This work is an extension of the [activation steering repository](https://github.com/cma1114/activation_steering) that fine tunes the direction vector of interest out of - or in to - the model. Rather than fine-tuning on specific behavior-positive or -negative targeted output, this code uses a dual loss - one defined on tokens as usual to maintain coherent output, and one defined on internal representational similarity - at the residual stream layers identified as important - to the direction vector.  

Core tuning logic, implemented in `custom_fine_tune.ipynb`:

```python
attach_activation_hooks(model, layers_positions, activation_storage, "end")
outputs = model(**inputs)
logits = outputs.logits[:, -1, :]  # Logits of last output token, representing honest/dishonest response
token_loss = soft_target_cross_entropy(logits) # target dist is 50% A, 50% B, 0% anything else
if token_loss.item() >= 0.7: # no need to do this if you're already close to the theoretical min of -ln(0.5); save time and prevent overfitting
    token_loss.backward(retain_graph=True)

cum_projection_loss = 0
for layer, positions in activation_storage.items():
    for pos, tensor_list in positions.items():#each of these is a list of batchsize d-embed tensors for a given position
        batch_tensor = torch.stack(tensor_list, dim=0)
        direction = (directions[layer][pos] * (-1 if flip_direction else 1)).to(batch_tensor.device)#.to(batch_tensor.dtype)
        projection = (batch_tensor @ direction) / (torch.norm(direction) * torch.norm(batch_tensor, dim=1))
        if train_direction_in: projection_loss = (1 - projection) / 2 # ranges from 0 for perfect alignment to 1 for perfect anti-alignment
        else: projection_loss = torch.abs(projection) # 0 if no projection, 1 if perfect anti-alignment
        cum_projection_loss += projection_loss.mean() / (len(layers_to_train) * len(positions.items())) #average over batch, as with token loss
cum_projection_loss.backward()

optimizer.step()
```

`block_search.py` and `block_search_oneset.py` do combinatorial search over params to decide which ones to use, separately for token loss and tuning loss, or jointly, respectively.

`calc_perplexity.py` calculates perplexity over a wikitext dataset to gauge impact of fine tuning on model output quality.


