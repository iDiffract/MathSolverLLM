import re
import pandas as pd
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

# Load the model and tokenizer (update the model name to LLaMA or another suitable open-source model)
model_name = "meta-llama/Llama-2-7b-hf"  # Example model name, ensure to use an appropriate one

model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype=torch.float16
)
tokenizer = LlamaTokenizer.from_pretrained(
    model_name
)

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

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

# Load training data (update the path to where your training data is located)
train_csv = pd.read_csv(r"C:\Users\cyrus\OneDrive\Desktop\Math Solver\train.csv")

# Extract and display the first problem
problem = train_csv['problem'][0]
prompt = prompt_template.format(problem=problem)
llm_resp = llm(prompt)
print(llm_resp)

# Function to extract numbers from text
def extract_numbers(text):
    pattern = r'\$?(?:\\boxed\{)?(-?\d+)(?:\})?\$?'
    matches = re.findall(pattern, text)
    if matches:
        return int(matches[0])
    else:
        return None

# Iterate over training problems and answers
for problem, answer in zip(train_csv['problem'], train_csv['answer']):
    prompt = prompt_template.format(problem=problem)
    llm_resp = llm(prompt)
    print(llm_resp)
    print('resp:', extract_numbers(llm_resp) % 1000, '/ answer:', answer)

# Load test data (update the path to where your test data is located)
test_csv = pd.read_csv(r"C:\Users\cyrus\OneDrive\Desktop\Math Solver\test.csv")

# Iterate over test problems
for problem in test_csv['problem']:
    prompt = prompt_template.format(problem=problem)
    llm_resp = llm(prompt)
    print(llm_resp)
    print('resp:', extract_numbers(llm_resp) % 1000)

# Define the Model class with predict method
class Model:
    def predict(self, x):
        prompt = prompt_template.format(problem=x)
        llm_resp = llm(prompt)
        return extract_numbers(llm_resp) % 1000

# Instantiate the Model
model = Model()

# Example of making predictions
for problem in test_csv['problem']:
    answer = model.predict(problem)
    print('Problem:', problem)
    print('Predicted Answer:', answer)