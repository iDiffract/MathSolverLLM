import re
import pandas as pd
from model_pipeline import pipe

# Define the LLM function
def llm(prompt: str) -> str:
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    output = pipe(prompt, **generation_args)
    return output[0]['generated_text']

# Define the prompt template
prompt_template = """
You are an advanced AI system tasked with solving complex mathematical problems.
Your exceptional mathematical reasoning and problem-solving abilities equip you 
to accurately analyze and solve these intricate problems, 
showcasing a deep grasp of mathematical concepts and robust logical reasoning skills.  

The instructions are as follows:
1. You must understand the provided math problem.
2. You will give the most probable integer answer, from 0 to 999, without using fractions.
3. Do not include any reasoning or thinking process - simply provide the integer answer.

When presented with the problem: {problem}

You will provide your answer as an integer. 
You will not explain your thought process, 
but merely return the required integer answer as a string less than 10 characters long.
"""

# Function to extract numbers from text
def extract_numbers(text):
    pattern = r'\$?(?:\\boxed\{)?(-?\d+)(?:\})?\$?'
    matches = re.findall(pattern, text)
    if matches:
        return int(matches[0])
    else:
        return None

# Load training data (update the path to where your training data is located)
def load_train_data(path: str):
    return pd.read_csv(path)

# Load test data (update the path to where your test data is located)
def load_test_data(path: str):
    return pd.read_csv(path)

# Define the Model class with predict method
class Model:
    def predict(self, x):
        prompt = prompt_template.format(problem=x)
        llm_resp = llm(prompt)
        return extract_numbers(llm_resp) % 1000