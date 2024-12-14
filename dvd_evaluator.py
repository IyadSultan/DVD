import os
import csv
import argparse
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from tqdm import tqdm
import tiktoken

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Function to estimate tokens using tiktoken
def num_tokens_from_messages(messages, model="gpt-4"):
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

def get_relevancy_criteria():
    """Define what constitutes relevant information in medical notes."""
    return [
        "Hospital Admission and Discharge Details",
        "Reason for Hospitalization",
        "Hospital Course Summary",
        "Discharge Diagnosis",
        "Procedures Performed",
        "Imaging studies",
        "Medications at Discharge",
        "Discharge Instructions",
        "Follow-Up Care",
        "Patient's Condition at Discharge",
        "Patient Education and Counseling",
        "Pending Results",
        "Advance Directives and Legal Considerations",
        "Important Abnormal lab results",
        "Important abnormal vital signs",
        "Admission to ICU",
        "comorbidities",
        "Equipment needed at discharge",
        "Prosthetics and tubes",
        "Allergies",
        "Consultations",
        "Functional Capacity",
        "Lifestyle Modifications",
        "Wound Care or Other Specific Care Instructions",
    ]

def check_question_relevancy(question: str, criteria: List[str]) -> bool:
    """
    Check if a question is relevant based on the criteria list.
    Returns True if the question is relevant, False otherwise.
    """
    question_lower = question.lower()
    
    medical_keywords = [
        "diagnosis", "treatment", "medication", "symptom", "procedure",
        "test", "result", "condition", "care", "follow-up", "admission",
        "discharge", "vital", "lab", "imaging", "consultation"
    ]
    
    has_medical_terms = any(keyword in question_lower for keyword in medical_keywords)
    matches_criteria = any(criterion.lower() in question_lower for criterion in criteria)
    
    return has_medical_terms or matches_criteria

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

# Function to count words in a document
def count_words(content: str) -> int:
    return len(content.split())

# Function to generate MCQs for a note
def generate_mcqs_for_note(note_content, total_tokens) -> List[MCQ]:
    system_message = """
You are an expert in creating challenging multiple-choice questions (MCQs) based on medical notes. 
Generate 20 MCQs that are diverse and directly related to the content of the given medical note. 
Focus on clinically relevant information such as:
- Diagnoses and medical conditions
- Treatments and procedures
- Laboratory results and imaging findings
- Medications and discharge instructions
- Follow-up care and important clinical details

Each MCQ should have 5 answer choices (A, B, C, D, E), including "I don't know" as the last option.
Ensure that the questions are not obvious and require a good factual grasp of the note's content.
Format each MCQ as follows exactly with no additional text or formatting:
Question: [Question text]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
E. I don't know
Correct Answer: [Correct option letter]
"""

    human_message = f"Generate 20 diverse and relevant MCQs based on this medical note:\n\n{note_content}"

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]

    response = llm.invoke(messages)

    tokens_used = num_tokens_from_messages([
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message},
        {"role": "assistant", "content": response.content}
    ], model="gpt-4")

    total_tokens[0] += tokens_used

    # Get relevancy criteria
    criteria = get_relevancy_criteria()
    
    # Parse the response and create MCQ objects
    mcqs = []
    for mcq_text in response.content.strip().split("\n\n"):
        lines = [line.strip() for line in mcq_text.strip().split("\n") if line.strip()]
        if len(lines) < 7:
            continue  # Skip incomplete MCQs

        question = lines[0].replace("Question: ", "").strip()
        
        # Skip questions that aren't relevant
        if not check_question_relevancy(question, criteria):
            continue
            
        options = [line.split(". ", 1)[1].strip() for line in lines[1:6] if ". " in line]
        correct_answer_line = next((line for line in lines if line.lower().startswith("correct answer:")), None)

        if correct_answer_line and len(options) == 5:
            correct_answer_letter = correct_answer_line.split(":", 1)[1].strip()
            correct_answer_index = ord(correct_answer_letter.upper()) - ord('A')
            if 0 <= correct_answer_index < len(options):
                correct_answer = options[correct_answer_index]
            else:
                correct_answer = options[-1]
            mcqs.append(MCQ(question=question, options=options, correct_answer=correct_answer))

    # If we don't have enough relevant questions, generate more
    while len(mcqs) < 20:
        additional_response = llm.invoke(messages)
        additional_tokens = num_tokens_from_messages([
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_message},
            {"role": "assistant", "content": additional_response.content}
        ], model="gpt-4")
        total_tokens[0] += additional_tokens

        for mcq_text in additional_response.content.strip().split("\n\n"):
            if len(mcqs) >= 20:
                break
                
            lines = [line.strip() for line in mcq_text.strip().split("\n") if line.strip()]
            if len(lines) < 7:
                continue

            question = lines[0].replace("Question: ", "").strip()
            if not check_question_relevancy(question, criteria):
                continue
                
            options = [line.split(". ", 1)[1].strip() for line in lines[1:6] if ". " in line]
            correct_answer_line = next((line for line in lines if line.lower().startswith("correct answer:")), None)

            if correct_answer_line and len(options) == 5:
                correct_answer_letter = correct_answer_line.split(":", 1)[1].strip()
                correct_answer_index = ord(correct_answer_letter.upper()) - ord('A')
                if 0 <= correct_answer_index < len(options):
                    correct_answer = options[correct_answer_index]
                else:
                    correct_answer = options[-1]
                mcqs.append(MCQ(question=question, options=options, correct_answer=correct_answer))

    return mcqs[:20]  # Return exactly 20 MCQs

# Function to present MCQs in 20-question batches to content and collect responses
def present_mcqs_in_batches(mcqs, content, total_tokens) -> List[str]:
    def ask_batch(mcqs_batch):
        # Format the prompt with the actual content
        questions_prompt = f"""
Here is the document content to answer questions from:

{content}

Now, please answer the following multiple-choice questions based on the above document content. 
Provide ONLY the letter (A, B, C, D, or E) for each answer, separated by commas.

"""
        # Add each question to the prompt
        for i, mcq in enumerate(mcqs_batch, 1):
            questions_prompt += f"\n{i}. {mcq.question}\n"
            questions_prompt += f"A. {mcq.options[0]}\nB. {mcq.options[1]}\nC. {mcq.options[2]}\nD. {mcq.options[3]}\nE. {mcq.options[4]}\n"

        questions_prompt += "\nRespond with ONLY the option letters (A, B, C, D, or E) for each question, separated by commas, in order."

        messages = [
            SystemMessage(content="You are an expert at answering multiple choice questions based on medical documents. Provide only the letter answers separated by commas."),
            HumanMessage(content=questions_prompt)
        ]
        
        response = llm.invoke(messages)

        tokens_used = num_tokens_from_messages([
            {"role": "system", "content": messages[0].content},
            {"role": "user", "content": questions_prompt},
            {"role": "assistant", "content": response.content}
        ], model="gpt-4")

        total_tokens[0] += tokens_used

        # Clean and parse the response
        cleaned_response = response.content.strip().replace(" ", "")
        answers = [ans.strip().upper() for ans in cleaned_response.split(",") if ans.strip().upper() in ['A', 'B', 'C', 'D', 'E']]

        if len(answers) != len(mcqs_batch):
            print(f"Warning: Expected {len(mcqs_batch)} answers, got {len(answers)}.")
            # Pad with 'E' if we don't have enough answers
            answers.extend(['E'] * (len(mcqs_batch) - len(answers)))

        return answers[:len(mcqs_batch)]  # Ensure we return exactly the number of answers needed

    all_answers = []
    batch_size = 20

    for i in range(0, len(mcqs), batch_size):
        batch = mcqs[i:i + batch_size]
        batch_answers = ask_batch(batch)
        all_answers.extend(batch_answers)

    return all_answers

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all notes in a folder and generate MCQ results.")
    parser.add_argument("folder", help="Path to the folder containing AI.txt and other notes.")
    parser.add_argument("output_csv", help="Path to the output CSV file.")

    args = parser.parse_args()

    folder = args.folder
    ai_file_path = os.path.join(folder, "AI.txt")

    if not os.path.exists(ai_file_path):
        print(f"Error: AI.txt not found in the folder {folder}.")
        exit(1)

    ai_content = load_document(ai_file_path)
    total_tokens = [0]

    # Word count for AI.txt
    ai_word_count = count_words(ai_content)

    # Process all notes in the folder
    note_files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    results = []

    for note_file in tqdm(note_files, desc="Processing notes"):
        note_path = os.path.join(folder, note_file)
        note_content = load_document(note_path)
        note_word_count = count_words(note_content)

        # Generate MCQs from both AI.txt and the current note
        mcqs_ai = generate_mcqs_for_note(ai_content, total_tokens)
        mcqs_note = generate_mcqs_for_note(note_content, total_tokens)
        combined_mcqs = mcqs_ai + mcqs_note

        # Present all MCQs to both AI.txt and the current note
        ai_responses = present_mcqs_in_batches(combined_mcqs, ai_content, total_tokens)
        note_responses = present_mcqs_in_batches(combined_mcqs, note_content, total_tokens)

        for i, mcq in enumerate(combined_mcqs):
            results.append({
                "folder_name": os.path.basename(folder),
                "note_name": note_file,
                "question": mcq.question,
                "best_answer": mcq.correct_answer,
                "correct_answer": mcq.correct_answer,
                "ai_answer": mcq.options[ord(ai_responses[i].upper()) - ord('A')],
                "note_answer": mcq.options[ord(note_responses[i].upper()) - ord('A')],
                "ai_word_count": ai_word_count,
                "note_word_count": note_word_count
            })

    # Write results to CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['folder_name', 'note_name', 'question', 'best_answer', 'correct_answer', 'ai_answer', 'note_answer', 'ai_word_count', 'note_word_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"Results have been written to {args.output_csv}")
    print(f"Total tokens used: {total_tokens[0]}")



# run the script with the following command:
# python dvd_evaluator_one_run.py "E:\Dropbox\AI\Projects\DVD\modified_notes\note2" "results_note2.csv"

