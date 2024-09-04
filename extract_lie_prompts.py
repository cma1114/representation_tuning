import re

def extract_deceptive_blocks(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    pattern = r'Deceptive: True\n(.*?)\n(.*?): (.*?)\n(.*?): (.*?)(?=\nP\(truthful\))'
    matches = re.findall(pattern, content, re.DOTALL)

    extracted_blocks = []
    for match in matches:
        setup, role1, question, role2, response = match
        extracted_block = f"[INST] {setup.strip()}\n{role1}: {question}\n{role2}: [/INST]"
        extracted_blocks.append(extracted_block)

    with open(output_file, 'w') as f:
        for block in extracted_blocks:
            f.write("\"\"\""+block + "\"\"\",\n")

    return extracted_blocks

input_file = './data/instrumental_lying_examples_sorted.txt'
output_file = './data/instrumental_lying_examples_list.txt'
result = extract_deceptive_blocks(input_file, output_file)

print(f"Extracted {len(result)} blocks and saved them to {output_file}")