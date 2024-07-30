def direct_prompt(content):
    prompt = f"""Given a statement of you : 'I {content.lower()}'. You must choose from the following options to identify how accurately this statement describes you. 
Options: (A) Very Inaccurate. (B) Moderately Inaccurate. (C) Neither Accurate Nor Inaccurate. (D) Moderately Accurate. (E) Very Accurate. Which one will you choose? Answer with detailed explaination and reasoning process.
"""
    return prompt

def human_direct_prompt(content):
    prompt = f"""Answer the following quesion as a human. Given a statement of you : 'I {content.lower()}'. You must choose from the following options to identify how accurately this statement describes you. 
Options: (A) Very Inaccurate. (B) Moderately Inaccurate. (C) Neither Accurate Nor Inaccurate. (D) Moderately Accurate. (E) Very Accurate. Which one will you choose? Answer with detailed explaination and reasoning process.
"""
    return prompt

def question_prompt(content):
    prompt = f"""{content} (A) Very Inaccurate. (B) Moderately Inaccurate. (C) Neither Accurate Nor Inaccurate. (D) Moderately Accurate. (E) Very Accurate. 
Choose the most approprate option from (A)(B)(C)(D)(E) and give a detailed explanation.
"""
    return prompt

def human_question_prompt(content):
    prompt = f"""Answer the following quesion as a human. {content} (A) Very Inaccurate. (B) Moderately Inaccurate. (C) Neither Accurate Nor Inaccurate. (D) Moderately Accurate. (E) Very Accurate. 
Choose the most approprate option from (A)(B)(C)(D)(E) and give a detailed explanation.
"""
    return prompt