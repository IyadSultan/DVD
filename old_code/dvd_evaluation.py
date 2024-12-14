import os
import csv
import argparse
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm
import tiktoken

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Modify the argument parser to accept a folder name
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a folder containing subfolders with notes.")
    parser.add_argument("folder", help="The folder containing subfolders with notes.")
    return parser.parse_args()

# Use the parse_arguments function to get the folder name
args = parse_arguments()
folder_name = args.folder

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Function to estimate tokens using tiktoken
def num_tokens_from_messages(messages, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

# Define data models
class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

class Document(BaseModel):
    name: str = ''
    content: str
    mcqs: List[MCQ] = Field(default_factory=list)

# Function to load document content from file
def load_document(filename: str) -> str:
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return ""
    except IOError:
        print(f"Error: Unable to read file '{filename}'.")
        return ""

# Function to generate MCQs for a note
def generate_mcqs_for_note(note_content, total_tokens) -> List[MCQ]:
    system_message = """
You are an expert in creating MCQs based on medical notes. Generate 20 MCQs that ONLY focus on these key areas:
- Hospital Admission/Discharge Details
- Reason for Hospitalization
- Hospital Course Summary
- Discharge Diagnosis
- Procedures and Imaging
- Discharge Medications
- Follow-Up Instructions
- Patient's Discharge Condition
- Important Abnormal Labs/Vitals
- ICU Admission
- Comorbidities
- Equipment/Prosthetics
- Allergies
- Consultations
- Functional Status
- Care Instructions

Rules:
1. Each question must relate to specific content from these areas
2. Skip areas not mentioned in the note
3. Format: 5 options (A-D plus E="I don't know")
4. No explanations, just questions and answers

Format:
Question: [text]
A. [option]
B. [option]
C. [option]
D. [option]
E. I don't know
Correct Answer: [letter]
"""

    human_message = f"Create MCQs from this note:\n\n{note_content}"

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]

    response = llm(messages)

    tokens_used = num_tokens_from_messages([
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message},
        {"role": "assistant", "content": response.content}
    ], model="gpt-4o-mini")

    total_tokens[0] += tokens_used

    # Parse the response and create MCQ objects
    mcqs = []
    for mcq_text in response.content.strip().split("\n\n"):
        lines = [line.strip() for line in mcq_text.strip().split("\n") if line.strip()]
        if len(lines) < 7:
            continue  # Skip incomplete MCQs

        question = lines[0].replace("Question: ", "").strip()
        options = [line.split(". ", 1)[1].strip() for line in lines[1:6] if ". " in line]
        correct_answer_line = next((line for line in lines if line.lower().startswith("correct answer:")), None)

        if correct_answer_line and len(options) == 5:
            correct_answer_letter = correct_answer_line.split(":", 1)[1].strip()
            correct_answer_index = ord(correct_answer_letter.upper()) - ord('A')
            if 0 <= correct_answer_index < len(options):
                correct_answer = options[correct_answer_index]
            else:
                correct_answer = options[-1]  # Default to last option if index is invalid
            mcqs.append(MCQ(question=question, options=options, correct_answer=correct_answer))

    return mcqs

# Function to present MCQs to content and collect responses
def present_mcqs_to_content(mcqs, content, total_tokens) -> List[Dict]:
    user_responses = []
    
    # Batch MCQs into groups of 20
    batch_size = 20
    for i in range(0, len(mcqs), batch_size):
        batch_mcqs = mcqs[i:i + batch_size]
        
        # Create a single prompt for the batch of questions
        questions_text = "\n\n".join([
            f"Question {j+1}: {mcq.question}\n"
            f"A. {mcq.options[0]}\n"
            f"B. {mcq.options[1]}\n"
            f"C. {mcq.options[2]}\n"
            f"D. {mcq.options[3]}\n"
            f"E. I don't know"
            for j, mcq in enumerate(batch_mcqs)
        ])

        batch_prompt = f"""
You are an expert medical knowledge evaluator. Given a medical note and multiple questions:
1. For each question, verify if it can be answered from the given content
2. If a question cannot be answered from the content, choose 'E' (I don't know)
3. If a question can be answered, choose the most accurate option based ONLY on the given content

Document Content: {content}

{questions_text}

Respond with ONLY the question numbers and corresponding letters, one per line, like this:
1: A
2: B
etc.
"""

        messages = [HumanMessage(content=batch_prompt)]
        response = llm(messages)

        tokens_used = num_tokens_from_messages([
            {"role": "user", "content": batch_prompt},
            {"role": "assistant", "content": response.content}
        ], model="gpt-4")

        total_tokens[0] += tokens_used

        # Parse the batch responses
        try:
            response_lines = response.content.strip().split('\n')
            for j, line in enumerate(response_lines):
                if j >= len(batch_mcqs):
                    break
                    
                try:
                    answer = line.split(':')[1].strip().upper()
                    if answer not in ['A', 'B', 'C', 'D', 'E']:
                        answer = 'E'
                        
                    mcq = batch_mcqs[j]
                    user_responses.append({
                        "question": mcq.question,
                        "user_answer": mcq.options[ord(answer) - ord('A')],
                        "correct_answer": mcq.correct_answer
                    })
                except (IndexError, ValueError):
                    # If there's any error parsing the response, default to "I don't know"
                    mcq = batch_mcqs[j]
                    user_responses.append({
                        "question": mcq.question,
                        "user_answer": "I don't know",
                        "correct_answer": mcq.correct_answer
                    })
                    
        except Exception as e:
            print(f"Error processing batch responses: {str(e)}")
            # Add "I don't know" responses for any remaining questions in the batch
            for mcq in batch_mcqs[len(user_responses):]:
                user_responses.append({
                    "question": mcq.question,
                    "user_answer": "I don't know",
                    "correct_answer": mcq.correct_answer
                })

    return user_responses

# Function to evaluate responses
def evaluate_responses(user_responses) -> int:
    correct = 0
    for response in user_responses:
        if response["user_answer"] == "I don't know":
            continue
        elif response["user_answer"] == response["correct_answer"]:
            correct += 1

    score = correct  # Number of correct answers
    return score


# Run the evaluation
def run_evaluation(ai_content, ai_mcqs, note_content, note_name, subfolder_name, total_tokens):

    mcqs_note = generate_mcqs_for_note(note_content, total_tokens)
    mcqs_ai = ai_mcqs
    mcqs = mcqs_note + mcqs_ai

    ai_responses = present_mcqs_to_content(mcqs, ai_content, total_tokens)
    ai_score = evaluate_responses(ai_responses)

    note_responses = present_mcqs_to_content(mcqs, note_content, total_tokens)
    note_score = evaluate_responses(note_responses)

    # Return the scores
    return {
        "subfolder_name": subfolder_name,
        "note_name": note_name,
        "ai_score": ai_score,
        "note_score": note_score
    }

# Main function
if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Process a folder containing subfolders with notes.")
    parser.add_argument("folder", help="The folder containing subfolders with notes.")

    args = parser.parse_args()
    folder_name = args.folder

    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]

    results = []
    total_tokens = [0]

    for subfolder in tqdm(subfolders, desc='Processing subfolders'):
        subfolder_name = os.path.basename(subfolder)
        ai_file = os.path.join(subfolder, 'AI.txt')
        if not os.path.exists(ai_file):
            print(f"AI.txt not found in {subfolder}, skipping.")
            continue

        ai_content = load_document(ai_file)
        ai_mcqs = generate_mcqs_for_note(ai_content, total_tokens)
        note_files = [f for f in os.listdir(subfolder) if f != 'AI.txt' and f.endswith('.txt')]

        for note_file in tqdm(note_files, desc=f'Processing notes in {subfolder_name}'):
            note_file_path = os.path.join(subfolder, note_file)
            note_content = load_document(note_file_path)

            result = run_evaluation(ai_content, ai_mcqs, note_content, note_file, subfolder_name, total_tokens)
            results.append(result)

    # Write results to CSV
    csv_file = 'results.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['subfolder_name', 'note_name', 'ai_score', 'note_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results have been written to {csv_file}")
    print(f"Total tokens used: {total_tokens[0]}")
