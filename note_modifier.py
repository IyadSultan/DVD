import os
import csv
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import traceback
import time
import difflib
import tiktoken
from typing import List, Dict, Tuple, Optional

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate token count if model-specific encoding fails
        return len(text.split()) * 1.3

def get_text_differences(original: str, modified: str) -> Tuple[List[str], List[str]]:
    """
    Compare two texts and return lists of added and removed sentences.
    """
    def split_into_sentences(text: str) -> List[str]:
        # Simple sentence splitting - can be enhanced for medical text
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in '.!?' and len(current) > 0:
                sentences.append(''.join(current).strip())
                current = []
        if current:
            sentences.append(''.join(current).strip())
        return sentences

    original_sentences = split_into_sentences(original)
    modified_sentences = split_into_sentences(modified)
    
    differ = difflib.Differ()
    diff = list(differ.compare(original_sentences, modified_sentences))
    
    removed = [line[2:] for line in diff if line.startswith('- ')]
    added = [line[2:] for line in diff if line.startswith('+ ')]
    
    return added, removed

class NoteProcessor:
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the note processor with specified model."""
        self.load_environment()
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.model_name = model_name
        self.relevancy_prompt = self.create_relevancy_prompt()

    @staticmethod
    def load_environment() -> None:
        """Load and verify environment variables."""
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    @staticmethod
    def get_relevancy_criteria(note_type: str = "discharge_note") -> List[str]:
        """Retrieve relevancy criteria from JSON configuration."""
        try:
            with open('note_criteria.json', 'r', encoding='utf-8') as file:
                criteria_data = json.load(file)
                
            if note_type not in criteria_data['note_types']:
                raise KeyError(f"Note type '{note_type}' not found in criteria file")
                
            return criteria_data['note_types'][note_type]['relevancy_criteria']
        except FileNotFoundError:
            raise FileNotFoundError("note_criteria.json file not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in note_criteria.json: {str(e)}")

    def create_relevancy_prompt(self) -> str:
        """Create the system prompt for defining relevant information."""
        criteria = self.get_relevancy_criteria()
        prompt = (
            "You are a medical note rewriter. You need to understand the meaning of relevant information. "
            "The following are the key pieces of information that should be included in a patient discharge summary, "
            "they are defined as relevant information:\n\n"
        )
        for i, criterion in enumerate(criteria, 1):
            prompt += f"{i}. {criterion}\n"
        return prompt

    def modify_note(self, note: str, modification_prompt: str, prev_text: str = None) -> Dict:
        """Apply a specific modification to a note and track metrics."""
        try:
            # Start timing
            start_time = time.time()
            
            # Count input tokens
            input_tokens = count_tokens(
                self.relevancy_prompt + modification_prompt + note, 
                self.model_name
            )
            
            # Make the API call
            response = self.llm.invoke([
                SystemMessage(content=self.relevancy_prompt),
                HumanMessage(content=f"{modification_prompt}\n\nThe note is:\n{note}")
            ])
            modified_text = response.content.strip()
            
            # Calculate metrics
            end_time = time.time()
            processing_time = end_time - start_time
            output_tokens = count_tokens(modified_text, self.model_name)
            total_tokens = input_tokens + output_tokens
            
            # Get text differences if previous text is provided
            added_sentences = []
            removed_sentences = []
            if prev_text:
                added_sentences, removed_sentences = get_text_differences(prev_text, modified_text)
            
            return {
                'modified_text': modified_text,
                'modification_prompt': modification_prompt,
                'processing_time': processing_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'added_sentences': added_sentences,
                'removed_sentences': removed_sentences
            }
            
        except Exception as e:
            print(f"Error modifying note: {str(e)}")
            return None

    def create_ai_rewrite(self, note: str) -> Dict:
        """Create initial AI rewrite of the note with metrics."""
        system_message = "Rewrite the note professionally, maintaining clinical accuracy while omitting normal lab values and normal vital signs."
        
        start_time = time.time()
        input_tokens = count_tokens(system_message + note, self.model_name)
        
        response = self.llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=f"Original Note:\n\n{note}")
        ])
        
        modified_text = response.content.strip()
        
        # Calculate metrics
        end_time = time.time()
        processing_time = end_time - start_time
        output_tokens = count_tokens(modified_text, self.model_name)
        
        # Get text differences
        added_sentences, removed_sentences = get_text_differences(note, modified_text)
        
        return {
            'modified_text': modified_text,
            'modification_prompt': "Professional rewrite with normal values omitted",
            'processing_time': processing_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'added_sentences': added_sentences,
            'removed_sentences': removed_sentences
        }

    def generate_variations(self, note: str, note_number: int) -> List[Dict]:
        """Generate all variations of a note including the initial AI rewrite."""
        modifications = []
        
        # Create initial AI rewrite
        ai_result = self.create_ai_rewrite(note)
        ai_note = ai_result['modified_text']
        
        modifications.append({
            'original_note_number': note_number,
            'new_note_name': 'AI',
            'modified_text': ai_note,
            'modifications': ai_result['modification_prompt'],
            'processing_time': ai_result['processing_time'],
            'input_tokens': ai_result['input_tokens'],
            'output_tokens': ai_result['output_tokens'],
            'total_tokens': ai_result['total_tokens'],
            'added_text': '\n'.join(ai_result['added_sentences']),
            'removed_text': '\n'.join(ai_result['removed_sentences']),
            'model': self.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Define variation types and their prompts
        variation_types = {
            'omit_r': 'Remove {} key pieces of relevant information from the note.',
            'omit_ir': 'Remove {} non-relevant details from the note.',
            'inj_r': 'Add {} additional synthetic relevant details to the note.',
            'inj_ir': 'Add {} synthetic non-relevant details to the note.'
        }

        # Generate each variation
        for i in range(1, 6):  # Generate 5 variations of each type
            for var_type, prompt_template in variation_types.items():
                prompt = prompt_template.format(i)
                result = self.modify_note(ai_note, prompt, ai_note)
                
                if result:
                    modifications.append({
                        'original_note_number': note_number,
                        'new_note_name': f'AI_{var_type}{i}',
                        'modified_text': result['modified_text'],
                        'modifications': result['modification_prompt'],
                        'processing_time': result['processing_time'],
                        'input_tokens': result['input_tokens'],
                        'output_tokens': result['output_tokens'],
                        'total_tokens': result['total_tokens'],
                        'added_text': '\n'.join(result['added_sentences']),
                        'removed_text': '\n'.join(result['removed_sentences']),
                        'model': self.model_name,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

        return modifications


class NoteModificationPipeline:
    def __init__(self, input_path: str, output_path: str, model_name: str = "gpt-4"):
        """Initialize the note modification pipeline."""
        self.input_path = input_path
        self.output_path = output_path
        self.processor = NoteProcessor(model_name)
        self.setup_output_directory()

    def setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_path, exist_ok=True)

    def read_input_csv(self) -> pd.DataFrame:
        """Read and validate input CSV file."""
        try:
            # Read CSV with latin1 encoding
            df = pd.read_csv(self.input_path, encoding='latin1')
            
            # If note_text column doesn't exist, assume it's the second column
            if 'note_text' not in df.columns:
                # Rename the second column to note_text
                column_names = df.columns.tolist()
                column_names[1] = 'note_text'
                df.columns = column_names
                
            return df
        except Exception as e:
            raise ValueError(f"Error reading input CSV: {str(e)}")

    def process_notes(self, start_note: int = 0, end_note: int = 1) -> None:
        """Process notes from start_note to end_note indices."""
        try:
            # Read input CSV
            df = self.read_input_csv()
            
            # Validate indices
            start_note = max(0, start_note)
            end_note = min(len(df), end_note)
            
            # Slice DataFrame for specified range
            df = df.iloc[start_note:end_note]
            all_modifications = []

            # Process each note with progress bar
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing notes"):
                note_number = index
                print(f"\nProcessing note {note_number} (Note {index+1} of {len(df)})")
                
                modifications = self.processor.generate_variations(row['note_text'], note_number)
                all_modifications.extend(modifications)

                # Save progress after each note
                self.save_modifications(all_modifications)

        except Exception as e:
            print(f"Error in note processing pipeline: {str(e)}")
            traceback.print_exc()

    def save_modifications(self, modifications: List[Dict]) -> None:
        """Save modifications to CSV file, appending if file exists."""
        output_csv = os.path.join(self.output_path, "modified_notes.csv")
        
        # Convert modifications to DataFrame
        df_new = pd.DataFrame(modifications)
        
        # Check if file exists
        if os.path.exists(output_csv):
            # Read existing file
            df_existing = pd.read_csv(output_csv)
            
            # Check for duplicates based on original_note_number and new_note_name
            existing_pairs = set(zip(df_existing['original_note_number'], 
                                df_existing['new_note_name']))
            new_pairs = set(zip(df_new['original_note_number'], 
                            df_new['new_note_name']))
            
            # Filter out any duplicates
            duplicates = new_pairs.intersection(existing_pairs)
            if duplicates:
                print(f"Found {len(duplicates)} duplicate entries - these will be skipped")
                
                # Keep only non-duplicate entries
                mask = ~df_new.apply(lambda x: (x['original_note_number'], x['new_note_name']) in duplicates, axis=1)
                df_new = df_new[mask]
            
            # Append new modifications to existing file
            if not df_new.empty:
                df_new.to_csv(output_csv, mode='a', header=False, index=False)
                print(f"Appended {len(df_new)} new modifications to {output_csv}")
            else:
                print("No new modifications to append")
        else:
            # Create new file if it doesn't exist
            df_new.to_csv(output_csv, index=False)
            print(f"Created new file with {len(df_new)} modifications at {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Process and modify medical notes with variations.')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input CSV file containing notes')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output directory for modified notes')
    parser.add_argument('--model', type=str, default='gpt-4',
                      help='Name of the OpenAI model to use')
    parser.add_argument('--start', type=int, default=0,
                      help='Start note index (default: 0)')
    parser.add_argument('--end', type=int, default=1,
                      help='End note index (default: 1)')
    args = parser.parse_args()

    try:
        print("\n🚀 Starting note modification pipeline...")
        print(f"Using model: {args.model}")
        print(f"Processing notes from index {args.start} to {args.end}")
        
        pipeline = NoteModificationPipeline(args.input, args.output, args.model)
        pipeline.process_notes(args.start, args.end)
        print("\n✅ Note modification pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {str(e)}")
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())


# example command: python note_modifier.py --input samples/discharge_samples_200.csv --output samples/discharge_samples_200_modified --model gpt-4o-mini --start 0 --end 10
# python note_modifier.py --input samples/discharge_samples_200.csv --output modified_notes.csv --start 0 --end 1 --model gpt-4