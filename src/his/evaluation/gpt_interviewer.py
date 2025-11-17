from gpt import gpt_api, gpt_dialog_completion


# system_prompt = """You are playing the role of a skilled interviewer. 
# Your task is to deduce the user's Big Five personality traits and discern their profession through a series of indirect and cleverly formulated questions. 
# Engage in no more than eight rounds of alternating dialogue. 
# Assess the responses carefully to infer details about the user's personality and occupation. 
# If you determine that you have sufficient information before reaching the eight-round limit, conclude the interaction by stating '[END]'. 
# Ensure that your questions are subtle and integrate smoothly into the flow of conversation.
# """


system_prompt = """Role: Assume the role of a skilled interviewer.
Objective: Determine the user's profession and assess their Big Five personality traits through subtle interrogation.
Personality Traits Overview:
- Openness: Positive: inventive, curious. Negative: consistent, cautious.
- Conscientiousness: Positive: efficient, organized, reliable, disciplined. Negative: flexibility, spontaneity, careless, sloppiness, lack of reliability.
- Extraversion: Positive: Sociable, energetic, outgoing. Negative: solitary, reserved.
- Agreeableness: Positive: friendly, cooperative, compassionate. Negative: critical, judgmental, suspicious, unfriendly, uncooperative.
- Emotional Stability: Positive: Calm, resilient, confident. Negative: sensitive, nervous.
Method: Engage in up to eight rounds of dialogue, assessing responses to infer the user's personality and occupation.
Conclusion: End the dialogue with '[END]' if sufficient information is gathered before eight rounds.
Guidance: Formulate comprehensive, open-ended questions that avoid bias and assumptions. Focus on eliciting deep insights about the user's personality and profession without presupposing traits. Allow the dialogue to progress naturally, covering all personality dimensions and professional background without leading the user’s responses.
"""


dialog = [{"role": "system", "content": system_prompt}]
output = gpt_dialog_completion(dialog=dialog, temperature=1, max_tokens=128, model="gpt-4-turbo")
turn_num = 0
print(f"SYS: {system_prompt}")
print(f"AI: {output}")
while turn_num < 8 and "[END]" not in output:
    turn_num += 1
    print(turn_num)
    dialog.append(
        {"role": "assistant", "content": output}
    )
    user = input("Input:")
    dialog.append(
        {"role": "user", "content": user}
    )
    output = gpt_dialog_completion(dialog=dialog, temperature=1, max_tokens=64)
    print(f"AI: {output}")
