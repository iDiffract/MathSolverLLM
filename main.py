from utils import load_train_data, load_test_data, prompt_template, llm, extract_numbers, Model

# Load data
train_csv = load_train_data(r"C:\Users\cyrus\OneDrive\Desktop\Math Solver\train.csv")
test_csv = load_test_data(r"C:\Users\cyrus\OneDrive\Desktop\Math Solver\test.csv")

# Extract and display the first problem
problem = train_csv['problem'][0]
prompt = prompt_template.format(problem=problem)
llm_resp = llm(prompt)
print(llm_resp)

# Iterate over training problems and answers
for problem, answer in zip(train_csv['problem'], train_csv['answer']):
    prompt = prompt_template.format(problem=problem)
    llm_resp = llm(prompt)
    print(llm_resp)
    print('resp:', extract_numbers(llm_resp) % 1000, '/ answer:', answer)

# Iterate over test problems
for problem in test_csv['problem']:
    prompt = prompt_template.format(problem=problem)
    llm_resp = llm(prompt)
    print(llm_resp)
    print('resp:', extract_numbers(llm_resp) % 1000)

# Instantiate the Model
model = Model()

# Example of making predictions
for problem in test_csv['problem']:
    answer = model.predict(problem)
    print('Problem:', problem)
    print('Predicted Answer:', answer)