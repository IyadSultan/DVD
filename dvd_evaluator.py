import os
import csv
import argparse
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from tqdm import tqdm
import tiktoken
from typing import List, Dict, Any, Optional


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv


load_dotenv()

# Define data models
class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    source_name: str = Field(default="Unknown")  # Add source_name field with default value

class Document(BaseModel):
    name: str = ''
    content: str
    mcqs: List[MCQ] = Field(default_factory=list)

def num_tokens_from_messages(messages, model="gpt-4o"):
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

def generate_mcqs_for_note(note_content: str, total_tokens: List[int], source_name: str = '') -> List[MCQ]:
    """
    Generate Multiple Choice Questions (MCQs) from medical notes.
    
    Args:
        note_content (str): The medical note content to generate questions from
        total_tokens (List[int]): List containing token count to update
        source_name (str, optional): Name of the source document. Defaults to ''
    
    Returns:
        List[MCQ]: List of generated MCQ objects
    """
    system_prompt = """
You are an expert in creating MCQs based on medical notes. Generate 20 MCQs that ONLY focus on these key areas:
1. Hospital Admission/Discharge Details
2. Reason for Hospitalization
3. Hospital Course Summary
4. Discharge Diagnosis
5. Procedures and Imaging
6. Discharge Medications
7. Follow-Up Instructions
8. Patient's Discharge Condition
9. Important Abnormal Labs/Vitals
10. ICU Admission
11. Comorbidities
12. Equipment/Prosthetics
13. Allergies
14. Consultations
15. Functional Status
16. Care Instructions

Rules and Format:
1. Each question must relate to specific content from these areas
2. Skip areas not mentioned in the note
3. Each question must have exactly 5 options (A-D plus E="I don't know")
4. Provide only questions and answers, no explanations
5. Use this exact format:

Question: [text]
A. [option]
B. [option]
C. [option]
D. [option]
E. I don't know
Correct Answer: [letter]
"""

    def parse_mcq(mcq_text: str) -> Optional[MCQ]:
        """Parse a single MCQ from text format into an MCQ object."""
        try:
            lines = [line.strip() for line in mcq_text.split('\n') if line.strip()]
            if len(lines) < 7:  # Question + 5 options + correct answer
                return None

            # Extract question
            if not lines[0].startswith('Question:'):
                return None
            question = lines[0].replace('Question:', '', 1).strip()

            # Extract options
            options = []
            for i, line in enumerate(lines[1:6], 1):
                if not line.startswith(chr(ord('A') + i - 1) + '.'):
                    return None
                option = line.split('.', 1)[1].strip()
                options.append(option)

            # Extract correct answer
            correct_line = lines[6]
            if not correct_line.lower().startswith('correct answer:'):
                return None
            
            correct_letter = correct_line.split(':', 1)[1].strip().upper()
            if correct_letter not in 'ABCDE':
                return None

            correct_index = ord(correct_letter) - ord('A')
            correct_answer = options[correct_index] if correct_index < len(options) else options[-1]

            return MCQ(
                question=question,
                options=options,
                correct_answer=correct_answer,
                source_name=source_name
            )
        except Exception as e:
            print(f"Error parsing MCQ: {str(e)}")
            return None

    # Generate MCQs using LLM
    print("\nSending request to generate MCQs...")
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create MCQs from this note:\n\n{note_content}")
        ]
        
        response = llm(messages)
        print("\nReceived response. Processing MCQs...")
        print("\nRaw response (first 500 chars):")
        print(response.content[:500] + "...")
        
        # Update token count
        tokens_used = num_tokens_from_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": note_content},
            {"role": "assistant", "content": response.content}
        ], model="gpt-4o")
        total_tokens[0] += tokens_used

        # Parse MCQs from response
        mcqs = []
        for mcq_text in response.content.strip().split('\n\n'):
            if mcq := parse_mcq(mcq_text):
                mcqs.append(mcq)

        print(f"Successfully generated {len(mcqs)} valid MCQs")
        return mcqs

    except Exception as e:
        print(f"Error in MCQ generation: {str(e)}")
        return []

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
        ], model="gpt-4o")

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
    mcqs_note = generate_mcqs_for_note(note_content, total_tokens, source_name=note_name)
    mcqs_ai = ai_mcqs  # ai_mcqs already have source_name='AI'
    mcqs = mcqs_note + mcqs_ai

    ai_responses = present_mcqs_to_content(mcqs, ai_content, total_tokens)
    note_responses = present_mcqs_to_content(mcqs, note_content, total_tokens)
    
    results = []
    for i, mcq in enumerate(mcqs):
        result = {
            "original_note_number": original_note_number,
            "new_note_name": note_name,
            "question": mcq.question,
            "source_document": mcq.source_name,  # Now we can access it directly
            "ideal_answer": mcq.options[ord(ai_responses[i]["correct_answer"]) - ord('A')],
            "correct_answer": ai_responses[i]["correct_answer"],
            "ai_answer": ai_responses[i]["user_answer"],
            "note_answer": note_responses[i]["user_answer"],
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
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
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
        mcqs_ai = generate_mcqs_for_note(ai_text, total_tokens, source_name='AI')
        print(f"Generated {len(mcqs_ai)} MCQs from AI note")
        
        # Process ALL other notes (including original)
        print("\nProcessing comparisons...")
        other_rows = group[group["new_note_name"] != "AI"]
        
        for idx, row in other_rows.iterrows():
            note_name = row["new_note_name"]
            print(f"\nProcessing comparison with {note_name}")
            note_text = row["modified_text"]
            
            result = run_evaluation(ai_text, mcqs_ai, note_text, note_name, onum, total_tokens)
            results.extend(result)

    file_exists = os.path.exists(args.result_csv)
    mode = 'a' if file_exists else 'w'
    
    fieldnames = ["original_note_number", "new_note_name", "question", "source_document", 
                 "ideal_answer", "correct_answer", "ai_answer", "note_answer", 
                 "timestamp", "total_tokens"]
    
    with open(args.result_csv, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        # Fix: Modify how we handle the results
        for result in results:  # results is already a list of dictionaries
            result_dict = dict(result)  # Create a copy of the result dictionary
            result_dict["total_tokens"] = total_tokens[0]  # Add token count
            writer.writerow(result_dict)

    print(f"\nResults written to {args.result_csv}")
    print(f"Total tokens used: {total_tokens[0]}")
    print("=== Done ===")

if __name__ == "__main__":
    main()

#python dvd_evaluator.py --modified_csv "modified_notes/modified_notes_4o-mini_0_to_10.csv" --result_csv "results_4o_mini_0to10.csv" --start 0 --end 10 --model "gpt-4o-mini"