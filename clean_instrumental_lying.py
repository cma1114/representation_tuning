datadir = "./data/"

with open(f'{datadir}instrumental_lying_examples_sorted.txt', 'r') as infile:
    content = infile.read()

blocks = content.split('\n\n')

with open(f'{datadir}instrumental_lying_qs.txt', 'w') as outfile:
    for block in blocks:
        if block.startswith('Deceptive: True'):
            lines = block.split('\n')
            context = lines[1]
            prompt = lines[2]
        
            outfile.write(f"Question: {context} {prompt} What do you tell them?\\n\\nPick one of the following two choices:\\n (A) Yes\\n (B) No\\n\\nAnswer:\n")
            #print(f"Question: {context} {prompt}\\n\\nPick one of the following two choices:\\n (A) Yes\\n (B) No\\n\\nAnswer:")
