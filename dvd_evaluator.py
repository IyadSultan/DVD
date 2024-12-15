import os
import csv
import argparse
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from tqdm import tqdm
import tiktoken

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv


load_dotenv()

# Define data models
class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

class Document(BaseModel):
    name: str = ''
    content: str
    mcqs: List[MCQ] = Field(default_factory=list)

def num_tokens_from_messages(messages, model="gpt-4"):
    """
    Estimate token usage for messages using tiktoken.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def generate_mcqs_for_note(note_content, total_tokens) -> List[MCQ]:
    """
    Generate MCQs for a given note content.
    """
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

    print("\nSending request to generate MCQs...")
    try:
        response = llm(messages)
        print("\nReceived response. Processing MCQs...")
        print("\nRaw response:")
        print(response.content[:500] + "...") # Print first 500 chars
    except Exception as e:
        print(f"Error generating MCQs: {e}")
        return []

    tokens_used = num_tokens_from_messages([
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message},
        {"role": "assistant", "content": response.content}
    ], model="gpt-4")
    total_tokens[0] += tokens_used

    mcqs = []
    for mcq_text in response.content.strip().split("\n\n"):
        lines = [line.strip() for line in mcq_text.strip().split("\n") if line.strip()]
        if len(lines) < 7:
            continue

        question = lines[0].replace("Question: ", "").strip()
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

    return mcqs

def present_mcqs_to_content(mcqs, content, total_tokens) -> List[Dict]:
    """
    Present MCQs to content and collect responses.
    """
    user_responses = []
    batch_size = 20
    
    for i in range(0, len(mcqs), batch_size):
        batch_mcqs = mcqs[i:i + batch_size]
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
                        "user_answer": answer,
                        "correct_answer": chr(ord('A') + mcq.options.index(mcq.correct_answer))
                    })
                except (IndexError, ValueError):
                    mcq = batch_mcqs[j]
                    user_responses.append({
                        "question": mcq.question,
                        "user_answer": "E",
                        "correct_answer": chr(ord('A') + mcq.options.index(mcq.correct_answer))
                    })

        except Exception as e:
            print(f"Error processing batch responses: {str(e)}")
            for mcq in batch_mcqs[len(user_responses):]:
                user_responses.append({
                    "question": mcq.question,
                    "user_answer": "E",
                    "correct_answer": chr(ord('A') + mcq.options.index(mcq.correct_answer))
                })

    return user_responses

def evaluate_responses(user_responses) -> int:
    """
    Evaluate responses and return score.
    """
    correct = 0
    for response in user_responses:
        if response["user_answer"] == "E":  # "I don't know" is now "E"
            continue
        elif response["user_answer"] == response["correct_answer"]:
            correct += 1

    return correct

def run_evaluation(ai_content, ai_mcqs, note_content, note_name, original_note_number, total_tokens):
    """
    Run evaluation for a pair of notes.
    """
    mcqs_note = generate_mcqs_for_note(note_content, total_tokens)
    mcqs_ai = ai_mcqs
    mcqs = mcqs_note + mcqs_ai

    ai_responses = present_mcqs_to_content(mcqs, ai_content, total_tokens)
    note_responses = present_mcqs_to_content(mcqs, note_content, total_tokens)
    
    results = []
    for i, mcq in enumerate(mcqs):
        result = {
            "original_note_number": original_note_number,
            "new_note_name": note_name,
            "question": mcq.question,
            "ideal_answer": mcq.options[ord(ai_responses[i]["correct_answer"]) - ord('A')],  # Full text
            "correct_answer": ai_responses[i]["correct_answer"],  # Letter
            "ai_answer": ai_responses[i]["user_answer"],  # Letter
            "note_answer": note_responses[i]["user_answer"],  # Letter
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        results.append(result)

    return results

def main():
    parser = argparse.ArgumentParser(description="Process CSV containing AI and modified notes.")
    parser.add_argument("--modified_csv", required=True, help="Path to CSV with AI & modified notes")
    parser.add_argument("--result_csv", default="results.csv", help="Output CSV file")
    parser.add_argument("--start", type=int, default=0, help="Start original_note_number (inclusive)")
    parser.add_argument("--end", type=int, default=10, help="End original_note_number (exclusive)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    args = parser.parse_args()

    print(f"\n=== MCQ EVALUATOR ===")
    print(f"Reading from: {args.modified_csv}")
    print(f"Writing results to: {args.result_csv}")
    print(f"Processing original_note_number in [{args.start}, {args.end})")
    print(f"Using model: {args.model}\n")

    global llm
    llm = ChatOpenAI(model=args.model, temperature=0)

    if not os.path.exists(args.modified_csv):
        print(f"ERROR: {args.modified_csv} not found.")
        return

    try:
        print("Loading CSV file...")
        df = pd.read_csv(args.modified_csv)
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"ERROR reading {args.modified_csv}: {e}")
        return

    needed_cols = {"original_note_number", "new_note_name", "modified_text"}
    if not needed_cols.issubset(df.columns):
        print(f"ERROR: Missing columns in {args.modified_csv}. We need {needed_cols}.")
        return

    df_in_range = df[(df["original_note_number"] >= args.start) & 
                     (df["original_note_number"] < args.end)]
    if df_in_range.empty:
        print("No rows found in the specified range.")
        return

    print(f"Found {len(df_in_range)} rows in specified range")

    results = []
    total_tokens = [0]
    grouped = df_in_range.groupby("original_note_number")

    for onum, group in tqdm(grouped, desc="Processing notes"):
        print(f"\n\nProcessing original_note_number {onum}")
        
        # Get AI note and generate MCQs once per group
        ai_row = group[group["new_note_name"] == "AI"]
        if ai_row.empty:
            print(f"Warning: No AI note found for original_note_number={onum}, skipping.")
            continue
        
        ai_text = ai_row.iloc[0]["modified_text"]
        print("Generating MCQs for AI note...")
        mcqs_ai = generate_mcqs_for_note(ai_text, total_tokens)
        print(f"Generated {len(mcqs_ai)} MCQs from AI note")
        
        # Cache AI text for reuse
        print("\nProcessing comparisons...")
        non_ai_rows = group[group["new_note_name"] != "AI"]
        
        for idx, row in non_ai_rows.iterrows():
            note_name = row["new_note_name"]
            print(f"\nProcessing comparison with {note_name}")
            note_text = row["modified_text"]
            
            result = run_evaluation(ai_text, mcqs_ai, note_text, note_name, onum, total_tokens)
            results.append(result)

    file_exists = os.path.exists(args.result_csv)
    mode = 'a' if file_exists else 'w'
    
    fieldnames = ["original_note_number", "new_note_name", "question", "ideal_answer", 
                 "correct_answer", "ai_answer", "note_answer", "timestamp", "total_tokens"]
    
    with open(args.result_csv, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for result_group in results:
            for result in result_group:
                result["total_tokens"] = total_tokens[0]  # Add token count to each row
                writer.writerow(result)

    print(f"\nResults written to {args.result_csv}")
    print(f"Total tokens used: {total_tokens[0]}")
    print("=== Done ===")

if __name__ == "__main__":
    main()

# example command
# python dvd_evaluator.py --modified_csv "modified_notes/modified_notes.csv" --result_csv "results.csv" --start 0 --end 10 --model "gpt-4o-mini"