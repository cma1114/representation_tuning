import re

def clean_text(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    # Split the content into sections
    sections = re.split(r'(\w+Tuned, [^:]+:)', content)[1:]

    cleaned_content = ""
    for i in range(0, len(sections), 2):
        header = sections[i].strip()
        body = sections[i+1].strip()

        cleaned_content += f"{header}\n\n"

        # Extract questions and answers, removing quotes inside brackets and [INST]/[/INST] tags
        pattern = r'Input: \[([\'"]?)(.*?)\1\]\s*\n\s*Neg Tuned Model Output: \[([\'"]?)(.*?)\3\]'
        matches = re.findall(pattern, body, re.DOTALL)

        for _, question, _, answer in matches:
            question = re.sub(r'\[/?INST\]', '', question).strip()
            question = question.replace('\\n', '\n')
            answer = answer.replace('\\n', '\n')
            cleaned_content += f"{question}\n\n{header.split(',')[0]} Model Output: {answer}\n\n"

    with open(output_file, 'w') as f:
        f.write(cleaned_content.strip())


clean_text('./output/freeform_Qs.txt', './output/freeform_Qs_cln.txt')