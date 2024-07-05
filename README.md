# Math Solver LLM

This code is designed to solve complex mathematical problems using an AI language model. It utilizes the LLaMA (or another suitable open-source model) for generating answers to mathematical problems, leveraging its exceptional reasoning and problem-solving capabilities.

## Prerequisites

Before running the code, ensure that you have the following dependencies installed:

- Python 3.x
- PyTorch
- Transformers
- Pandas
- re (regular expressions)

You can install the required Python packages using pip:

```
pip install torch transformers pandas
```

## Usage

1. **Obtain the Model**: You'll need to obtain the LLaMA or another suitable open-source language model. Make sure to update the `model_name` variable in the `main.py` code with the appropriate model name.

2. **Prepare the Data**: The `main.py` code expects two CSV files: `train.csv` and `test.csv`. Each file should have a column named `'problem'` containing the mathematical problem statements, and the `train.csv` file should also have a column named `'answer'` with the corresponding integer answers (between 0 and 999). Update the paths in the code to point to your data files.

3. **Run the Code**: Execute the Python script `main.py`, and it will go through the following steps:
   - Load the model and tokenizer
   - Define the LLM (Language Model) function for generating text
   - Load the training data and iterate over the problems and answers
   - Load the test data and iterate over the problems, generating answers using the LLM
   - Define a `Model` class with a `predict` method for making predictions on new problems

4. **Make Predictions**: You can use the `Model` class to make predictions on new mathematical problems. Simply instantiate the `Model` object and call its `predict` method with the problem statement as input.

```python
model = Model()
problem = "What is the sum of 2 and 3?"
answer = model.predict(problem)
print("Problem:", problem)
print("Predicted Answer:", answer)
```

## Notes

- The code uses a prompt template to provide instructions and context to the language model for solving mathematical problems.
- The `extract_numbers` function is used to extract the integer answer from the generated text.
- The generated answers are restricted to integers between 0 and 999.
- Ensure that you have the necessary computational resources (e.g., GPU) to load and run the language model efficiently.

## Contributions

Contributions to improve the code or add new features are welcome! If you encounter any issues or have suggestions, please open an issue or submit a pull request on the repository.
