import os
import csv
import argparse
import json
import pandas as pd
from typing import List
from pydantic import BaseModel
from tqdm import tqdm
import tiktoken

def num_tokens_from_messages(messages, model="gpt-4"):
    """
    Estimate the token usage for a list of messages using tiktoken.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # base overhead per message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def load_discharge_criteria(json_path: str = "note_criteria.json",
                          note_type: str = "discharge_note") -> List[str]:
    """
    Loads 'discharge_note' relevancy criteria from note_criteria.json.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        note_types = data.get("note_types", {})
        if note_type not in note_types:
            raise ValueError(f"note_type '{note_type}' not found in {json_path}.")
        return note_types[note_type]["relevancy_criteria"]
    except FileNotFoundError:
        print(f"ERROR: {json_path} not found.")
        return []
    except Exception as e:
        print(f"ERROR loading criteria from {json_path}: {e}")
        return []

from langchain_openai import ChatOpenAI

# Data model for MCQ
class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
from langchain.schema import HumanMessage, SystemMessage

def check_question_relevancy(question: str, criteria: List[str]) -> bool:
    """
    Check if a question is relevant based on the discharge_note criteria list.
    Returns True if the question is relevant, False otherwise.
    """
    print(f"Checking relevancy for question: {question[:100]}...")
    question_lower = question.lower()
    
    # Convert criteria to lowercase for matching
    criteria_lower = [c.lower() for c in criteria]
    
    # Direct match with criteria
    for criterion in criteria_lower:
        if any(term in question_lower for term in criterion.split()):
            print(f"Matched criterion: {criterion}")
            return True
    
    # Common medical terms that indicate relevance
    medical_keywords = {
        "diagnosis", "admission", "discharge", "hospital", "treatment",
        "procedure", "medication", "follow-up", "care", "patient",
        "symptoms", "condition", "complications", "results", "lab",
        "imaging", "vital signs", "instructions", "education",
        "surgical", "medical", "health", "clinical", "therapy"
    }
    
    # Check for medical keywords
    for keyword in medical_keywords:
        if keyword in question_lower:
            print(f"Found medical keyword: {keyword}")
            return True
    
    print("No relevant keywords or criteria found")
    return False

def generate_mcqs_for_note(note_content: str, 
                           total_tokens: List[int], 
                           discharge_criteria: List[str]) -> List[MCQ]:
    """
    Creates up to 20 relevant MCQs for note_content, referencing discharge_note criteria.
    """
    print(f"\nGenerating MCQs... ")
    print(f"Using {len(discharge_criteria)} criteria: {discharge_criteria}")
    
    system_message = """
You are an expert at creating multiple-choice questions (MCQs) based on discharge medical notes.
Generate 20 medical questions directly related to the given note.
Each question must focus on critical medical information like diagnoses, treatments, medications, or follow-up care.

Format EXACTLY as shown (including line breaks):
Question: [Question text]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
E. I don't know
Correct Answer: [A/B/C/D/E]

Example:
Question: What was the primary discharge diagnosis for this patient?
A. Bacterial pneumonia
B. Congestive heart failure
C. Acute kidney injury
D. Diabetic ketoacidosis
E. I don't know
Correct Answer: B
"""

    user_message = f"Create 20 medical MCQs based on this discharge note. Use EXACTLY the format shown in the example:\n\n{note_content}"

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message)
    ]
    
    print("\nSending request to generate MCQs...")
    try:
        response = llm.invoke(messages)
        print("\nReceived response. Processing MCQs...")
        print("\nRaw response:")
        print(response.content[:500] + "...") # Print first 500 chars of response
    except Exception as e:
        print(f"Error generating MCQs: {e}")
        return []

    # Count tokens
    tokens_used = num_tokens_from_messages([
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response.content}
    ], model="gpt-4")
    total_tokens[0] += tokens_used

    mcqs = []
    current_mcq = {}
    
    # Split into individual MCQs
    raw_mcqs = response.content.strip().split('\n\n')
    print(f"\nFound {len(raw_mcqs)} potential MCQ blocks")
    
    for block in raw_mcqs:
        print(f"\nProcessing block:\n{block[:200]}...")  # Print first 200 chars of block
        
        lines = [ln.strip() for ln in block.split('\n') if ln.strip()]
        if len(lines) < 7:  # Must have Question, 5 options, and Correct Answer
            print(f"Skipping block: insufficient lines ({len(lines)})")
            continue

        # Extract question
        if not lines[0].lower().startswith('question:'):
            print("Skipping block: no question found")
            continue
            
        question = lines[0].split(':', 1)[1].strip()
        print(f"Found question: {question}")

        # Check relevancy
        if not check_question_relevancy(question, discharge_criteria):
            print("Question not relevant, skipping")
            continue

        # Extract options
        options = []
        for opt_line in lines[1:6]:
            if '. ' in opt_line:
                option_text = opt_line.split('. ', 1)[1].strip()
                options.append(option_text)
        
        if len(options) != 5:
            print(f"Skipping: wrong number of options ({len(options)})")
            continue
            
        # Get correct answer
        correct_answer_line = None
        for ln in lines:
            if ln.lower().startswith('correct answer:'):
                correct_answer_line = ln
                break
                
        if not correct_answer_line:
            print("Skipping: no correct answer found")
            continue

        # Extract the letter answer and convert to text
        letter = correct_answer_line.split(':', 1)[1].strip().upper()
        if letter not in 'ABCDE':
            print(f"Invalid answer letter: {letter}")
            continue
            
        correct_index = ord(letter) - ord('A')
        correct_text = options[correct_index]
        
        print(f"Adding valid MCQ with answer: {correct_text}")
        mcqs.append(MCQ(question=question, options=options, correct_answer=correct_text))

    print(f"\nGenerated {len(mcqs)} valid MCQs")
    
    # If we didn't get enough valid MCQs, try again with simpler prompt
    attempts = 1
    while len(mcqs) < 20 and attempts < 3:
        attempts += 1
        print(f"\nAttempt {attempts}/3 to generate additional MCQs...")
        try:
            additional_response = llm.invoke(messages)
            print("Processing additional response...")
            new_raw_mcqs = additional_response.content.strip().split('\n\n')
            
            for block in new_raw_mcqs:
                lines = [ln.strip() for ln in block.split('\n') if ln.strip()]
                if len(lines) < 7:
                    continue
                    
                if not lines[0].lower().startswith('question:'):
                    continue
                    
                question = lines[0].split(':', 1)[1].strip()
                if not check_question_relevancy(question, discharge_criteria):
                    continue

                options = []
                for opt_line in lines[1:6]:
                    if '. ' in opt_line:
                        option_text = opt_line.split('. ', 1)[1].strip()
                        options.append(option_text)
                        
                if len(options) != 5:
                    continue

                correct_line = None
                for ln in lines:
                    if ln.lower().startswith('correct answer:'):
                        correct_line = ln
                        break
                        
                if not correct_line:
                    continue
                    
                letter = correct_line.split(':', 1)[1].strip().upper()
                if letter not in 'ABCDE':
                    continue
                    
                correct_index = ord(letter) - ord('A')
                correct_text = options[correct_index]
                
                mcqs.append(MCQ(question=question, options=options, correct_answer=correct_text))
                if len(mcqs) >= 5:
                    break
                    
        except Exception as e:
            print(f"Error in attempt {attempts}: {e}")
            continue

    print(f"Final MCQ count: {len(mcqs)}")
    return mcqs[:20]  # Return at most 20 MCQs

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message)
    ]
    response = llm.invoke(messages)

    # Count tokens
    tokens_used = num_tokens_from_messages([
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response.content}
    ], model="gpt-4")
    total_tokens[0] += tokens_used

    raw_mcqs = response.content.strip().split("\n\n")
    mcqs = []
    
    print(f"Processing {len(raw_mcqs)} raw MCQs...")

    for block in raw_mcqs:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if len(lines) < 7:
            continue

        if not lines[0].lower().startswith("question:"):
            continue
        question = lines[0].split("Question:",1)[1].strip()
        if not check_question_relevancy(question, discharge_criteria):
            continue

        options = []
        for opt_line in lines[1:6]:
            if ". " in opt_line:
                parts = opt_line.split(". ", 1)
                if len(parts) == 2:
                    options.append(parts[1].strip())
        if len(options) != 5:
            continue

        correct_answer_line = None
        for ln in lines:
            if ln.lower().startswith("correct answer:"):
                correct_answer_line = ln
                break
        if not correct_answer_line:
            continue
        letter = correct_answer_line.split(":",1)[1].strip().upper()
        if letter not in ["A","B","C","D","E"]:
            letter = "E"

        correct_index = ord(letter) - ord('A')
        correct_text = options[correct_index] if 0 <= correct_index < 5 else options[-1]

        mcqs.append(MCQ(question=question, options=options, correct_answer=correct_text))

    print(f"Generated {len(mcqs)} valid MCQs")

    # If fewer than 20, attempt re-generation
    attempt = 0
    while len(mcqs) < 20 and attempt < 3:
        attempt += 1
        print(f"\nAttempt {attempt}/3 to generate additional MCQs...")
        additional_response = llm.invoke(messages)
        additional_tokens = num_tokens_from_messages([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": additional_response.content}
        ], model="gpt-4")
        total_tokens[0] += additional_tokens

        new_blocks = additional_response.content.strip().split("\n\n")
        for nb in new_blocks:
            lines = [ln.strip() for ln in nb.split("\n") if ln.strip()]
            if len(lines) < 7:
                continue
            if not lines[0].lower().startswith("question:"):
                continue
            question = lines[0].split("Question:",1)[1].strip()
            if not check_question_relevancy(question, discharge_criteria):
                continue

            new_opts = []
            for oline in lines[1:6]:
                if ". " in oline:
                    parts = oline.split(". ", 1)
                    if len(parts) == 2:
                        new_opts.append(parts[1].strip())
            if len(new_opts) != 5:
                continue

            correct_line = None
            for ln in lines:
                if ln.lower().startswith("correct answer:"):
                    correct_line = ln
                    break
            if not correct_line:
                continue
            letter = correct_line.split(":",1)[1].strip().upper()
            if letter not in ["A","B","C","D","E"]:
                letter = "E"
            c_idx = ord(letter) - ord('A')
            c_text = new_opts[c_idx] if 0 <= c_idx < 5 else new_opts[-1]

            mcqs.append(MCQ(question=question, options=new_opts, correct_answer=c_text))
            if len(mcqs) >= 20:
                break

    print(f"Final MCQ count: {len(mcqs[:20])}")
    return mcqs[:20]

def present_mcqs_in_batches(mcqs: List[MCQ], note_content: str, 
                            total_tokens: List[int]) -> List[str]:
    """
    Presents the MCQs to the given note_content in batches, collecting answers (A..E).
    """
    def ask_batch(batch_mcqs):
        print(f"\nProcessing batch of {len(batch_mcqs)} MCQs...")
        
        prompt = f"""Here is the note content to answer questions from:

{note_content}

Provide ONLY the letter (A, B, C, D, or E) for each answer, separated by commas, in order.
"""
        for i, mcq in enumerate(batch_mcqs, 1):
            prompt += f"\n{i}. {mcq.question}"
            prompt += f"\nA. {mcq.options[0]}"
            prompt += f"\nB. {mcq.options[1]}"
            prompt += f"\nC. {mcq.options[2]}"
            prompt += f"\nD. {mcq.options[3]}"
            prompt += f"\nE. {mcq.options[4]}\n"

        prompt += "\nReply with the letters in order, separated by commas."

        messages = [
            SystemMessage(content="You are an expert at answering MCQs based on the note content. Return only the letters (A,B,C,D,E)."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)

        tokens_used = num_tokens_from_messages([
            {"role": "system", "content": messages[0].content},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ], model="gpt-4")
        total_tokens[0] += tokens_used

        cleaned = response.content.strip().replace(" ", "")
        raw_answers = cleaned.split(",")
        answers = [ans.strip().upper() for ans in raw_answers if ans.strip().upper() in ['A','B','C','D','E']]

        if len(answers) < len(batch_mcqs):
            answers += ['E']*(len(batch_mcqs) - len(answers))
        
        print(f"Received {len(answers)} answers")
        return answers[:len(batch_mcqs)]

    all_answers = []
    batch_size = 20
    for i in range(0, len(mcqs), batch_size):
        batch = mcqs[i:i+batch_size]
        batch_answers = ask_batch(batch)
        all_answers.extend(batch_answers)
    return all_answers

def main():
    parser = argparse.ArgumentParser(description="Generate and present MCQs for notes in a CSV, comparing AI vs. modifications.")
    parser.add_argument("--modified_csv", default="modified_notes.csv", 
                       help="Path to the CSV containing AI & modified notes.")
    parser.add_argument("--result_csv", default="results_mcq.csv", 
                       help="Results CSV (created or appended).")
    parser.add_argument("--start", type=int, default=0, 
                       help="Start original_note_number (inclusive).")
    parser.add_argument("--end", type=int, default=10, 
                       help="End original_note_number (exclusive).")
    parser.add_argument("--criteria_json", default="note_criteria.json", 
                       help="Path to note_criteria.json for discharge_note criteria.")
    parser.add_argument("--model", default="gpt-4", 
                       choices=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o-mini"],
                       help="OpenAI model to use for evaluation")
    args = parser.parse_args()

    print(f"\n=== MCQ EVALUATOR ===")
    print(f"Reading from: {args.modified_csv}")
    print(f"Writing results to: {args.result_csv}")
    print(f"Processing original_note_number in [{args.start}, {args.end})")
    print(f"Using model: {args.model}\n")

    global llm
    llm = ChatOpenAI(model=args.model, temperature=0.3)

    discharge_criteria = load_discharge_criteria(args.criteria_json, "discharge_note")
    if not discharge_criteria:
        print("Error: Could not load discharge_note criteria. Exiting.")
        return

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

    needed_cols = {"original_note_number","new_note_name","modified_text"}
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
        
        ai_row = group[group["new_note_name"] == "AI"]
        if ai_row.empty:
            print(f"Warning: No AI note found for original_note_number={onum}, skipping.")
            continue
        
        ai_text = ai_row.iloc[0]["modified_text"]
        print("Generating MCQs for AI note...")
        mcqs_ai = generate_mcqs_for_note(ai_text, total_tokens, discharge_criteria)

        for idx, row in group.iterrows():
            if row["new_note_name"] == "AI":
                continue

            note_name = row["new_note_name"]
            print(f"\nProcessing comparison with {note_name}")
            note_text = row["modified_text"]
            
            print("Generating MCQs for comparison note...")
            mcqs_note = generate_mcqs_for_note(note_text, total_tokens, discharge_criteria)
            combined_mcqs = mcqs_ai + mcqs_note

            print("\nEvaluating AI note responses...")
            ai_answers = present_mcqs_in_batches(combined_mcqs, ai_text, total_tokens)
            print("\nEvaluating comparison note responses...")
            note_answers = present_mcqs_in_batches(combined_mcqs, note_text, total_tokens)

            print("\nProcessing results...")
            for i, mcq in enumerate(combined_mcqs):
                ai_letter = ai_answers[i]
                note_letter = note_answers[i]
                ai_option_text = mcq.options[ord(ai_letter) - ord('A')]
                note_option_text = mcq.options[ord(note_letter) - ord('A')]

                results.append({
                    "original_note_number": onum,
                    "new_note_name": note_name,
                    "question": mcq.question,
                    "correct_answer": mcq.correct_answer,
                    "ai_answer": ai_option_text,
                    "note_answer": note_option_text
                })

    file_exists = os.path.exists(args.result_csv)
    mode = 'a' if file_exists else 'w'
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        print(f"\nSaving results...")
        cols = ["original_note_number","new_note_name","question","correct_answer","ai_answer","note_answer"]
        df_results = df_results[cols]
        with open(args.result_csv, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not file_exists:
                writer.writeheader()
            for _, r in df_results.iterrows():
                writer.writerow(r.to_dict())
        print(f"Wrote {len(df_results)} rows to {args.result_csv}")
    else:
        print("No evaluations performed.")

    print(f"\nTotal tokens used: {total_tokens[0]}")
    print("=== Done ===")

if __name__ == "__main__":
    main()

# python dvd_evaluator.py --modified_notes.csv/modified_notes.csv --results_dvd_evaluator.csv --start 0 --end 1 --model gpt-4o-mini