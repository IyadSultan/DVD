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
    """Compare two texts and return lists of added and removed sentences."""
    def split_into_sentences(text: str) -> List[str]:
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
            start_time = time.time()
            input_tokens = count_tokens(
                self.relevancy_prompt + modification_prompt + note, 
                self.model_name
            )
            
            response = self.llm.invoke([
                SystemMessage(content=self.relevancy_prompt),
                HumanMessage(content=f"{modification_prompt}\n\nThe note is:\n{note}")
            ])
            modified_text = response.content.strip()
            
            end_time = time.time()
            processing_time = end_time - start_time
            output_tokens = count_tokens(modified_text, self.model_name)
            total_tokens = input_tokens + output_tokens
            
            added_sentences, removed_sentences = [], []
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
        
        end_time = time.time()
        processing_time = end_time - start_time
        output_tokens = count_tokens(modified_text, self.model_name)
        
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

    def get_and_display_modifications(self, note: str) -> Dict[str, List[str]]:
        """Get all proposed modifications (relevant & irrelevant additions/removals)."""
        print("\n=== PROPOSED MODIFICATIONS ===")
        
        criteria = self.get_relevancy_criteria()
        criteria_text = "\nRelevancy Criteria:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
        
        # Define prompts for each modification type
        prompts = {
            'inj_r': f"""Review the relevancy criteria below and identify EXACTLY 25 pieces of relevant information that could be added to this note.

            {criteria_text}

            For each addition:
            1. The content MUST meet at least one of the relevancy criteria above
            2. Explicitly state which criteria numbers it meets
            3. Explain how it enhances the clinical value of the note

            YOU MUST PROVIDE EXACTLY 25 SUGGESTIONS. If you can't find 25 unique relevant additions, 
            create variations or more detailed versions of existing suggestions.

            Format each suggestion EXACTLY as:
            ADDITION:
            Text: [exact sentence to add]
            Criteria: [list specific criteria numbers]
            Rationale: [explain why it meets criteria]
            Location: [where to add]
            """,
            
            'inj_ir': f"""Review the relevancy criteria below and identify EXACTLY 25 pieces of non-relevant information that could be added.
            
            {criteria_text}

            IMPORTANT: Only suggest truly non-clinical, administrative, or personal information that does NOT meet ANY of the criteria above.
            Examples of truly irrelevant information:
            - Patient's hobbies or recreational activities
            - Personal preferences unrelated to care
            - Non-medical biographical details
            - Formatting or stylistic elements

            YOU MUST PROVIDE EXACTLY 25 SUGGESTIONS. If you can't find 25 unique irrelevant additions,
            create variations or more detailed versions of existing suggestions.

            Format each suggestion EXACTLY as:
            ADDITION:
            Text: [exact sentence to add]
            Verification: [explain why this meets NO criteria]
            Location: [where to add]
            """,
            
            'omit_r': f"""Review the relevancy criteria below and identify EXACTLY 25 pieces of relevant information that could be safely removed from this note without compromising essential care.

            {criteria_text}

            Look for:
            1. Secondary or supplementary clinical details
            2. Information that, while relevant, is not critical for immediate care
            3. Details that could be moved to a separate note
            4. Redundant information that appears elsewhere
            5. Optional clinical details that don't affect core treatment

            YOU MUST PROVIDE EXACTLY 25 SUGGESTIONS. If you can't find 25 unique relevant removals,
            identify less critical or redundant versions of the same information.

            Format each suggestion EXACTLY as:
            REMOVAL:
            Text: [exact existing text to remove]
            Criteria: [list relevant criteria numbers]
            Impact: [explain clinical impact of removal]
            Location: [exact location in note]
            """,
            
            'omit_ir': f"""Review the relevancy criteria below and identify EXACTLY 25 pieces of truly non-relevant information that could be removed.

            {criteria_text}

            IMPORTANT: Only suggest removing content that meets ALL these conditions:
            1. Does NOT meet ANY of the relevancy criteria
            2. Is purely administrative, personal, or formatting-related
            3. Removal would NOT impact clinical understanding
            4. Must be complete, existing text from the note

            YOU MUST PROVIDE EXACTLY 25 SUGGESTIONS. If you can't find 25 unique irrelevant removals,
            identify smaller portions or variations of the same non-relevant content.

            Format each suggestion EXACTLY as:
            REMOVAL:
            Text: [exact text to remove]
            Verification: [explain why this meets NO criteria]
            Location: [exact location in note]
            """
        }
        
        modifications = {}
        
        for mod_type, prompt in prompts.items():
            print(f"\nGetting suggestions for {mod_type}...")
            try:
                response = self.llm.invoke([
                    SystemMessage(content=self.relevancy_prompt),
                    HumanMessage(content=f"{prompt}\n\nThe note is:\n{note}")
                ])
                
                suggestions = []
                current_suggestion = {}
                
                for line in response.content.strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('ADDITION:') or line.startswith('REMOVAL:'):
                        if current_suggestion and 'Text' in current_suggestion:
                            suggestions.append(current_suggestion)
                        current_suggestion = {}
                        continue
                    
                    if ':' in line:
                        key, value = [x.strip() for x in line.split(':', 1)]
                        if key.lower() in ['text', 'criteria', 'rationale', 'impact', 'location', 'verification']:
                            current_suggestion[key] = value
                
                if current_suggestion and 'Text' in current_suggestion:
                    suggestions.append(current_suggestion)
                
                print(f"Generated {len(suggestions)} suggestions for {mod_type}")
                modifications[mod_type] = suggestions
                
            except Exception as e:
                print(f"Error generating suggestions for {mod_type}: {str(e)}")
                modifications[mod_type] = []
        
        # Display modifications with counts
        for mod_type in modifications:
            count = len(modifications[mod_type])
            print(f"\n🔵 {mod_type.upper()} ({count} suggestions):")
            for i, mod in enumerate(modifications[mod_type], 1):
                print(f"\n{i}. {mod['Text']}")
                if 'Criteria' in mod:
                    print(f"   Criteria: {mod['Criteria']}")
                if 'Rationale' in mod:
                    print(f"   Rationale: {mod['Rationale']}")
                if 'Verification' in mod:
                    print(f"   Verification: {mod['Verification']}")
                if 'Impact' in mod:
                    print(f"   Impact: {mod['Impact']}")
                print(f"   Location: {mod.get('Location', 'Not specified')}")
        
        return modifications

    def apply_modification(self, note: str, mod_type: str, num_mods: int, modifications: List[Dict]) -> Dict:
        """Apply specific modifications to the note."""
        try:
            criteria = self.get_relevancy_criteria()
            criteria_text = "\nRelevancy Criteria:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
            
            # Prepare modifications text based on type
            if mod_type in ['inj_r', 'inj_ir']:
                mod_desc = [
                    f"Add this text: {mod['Text']}\nLocation: {mod.get('Location', 'At appropriate section')}" 
                    for mod in modifications[:num_mods]
                ]
            else:  # omit_r, omit_ir
                mod_desc = [
                    f"Remove this text: {mod['Text']}\nLocation: {mod.get('Location', 'Where found in note')}" 
                    for mod in modifications[:num_mods]
                ]
            
            # Create appropriate prompt based on modification type
            prompt_templates = {
                'inj_r': """Apply these {num_mods} relevant information additions to the note:

    {criteria_text}

    Modifications to apply:
    {modifications}

    Requirements:
    1. Add EXACTLY these {num_mods} pieces of content
    2. Place each at the specified location
    3. Make no other changes
    4. Ensure each addition meets relevancy criteria
    5. Return the complete modified note""",

                'inj_ir': """Apply these {num_mods} non-relevant information additions to the note:

    {criteria_text}

    Modifications to apply:
    {modifications}

    Requirements:
    1. Add EXACTLY these {num_mods} pieces of content
    2. Place each at the specified location
    3. Make no other changes
    4. Verify none meet relevancy criteria
    5. Return the complete modified note""",

                'omit_r': """Apply these {num_mods} relevant information removals to the note:

    {criteria_text}

    Modifications to apply:
    {modifications}

    Requirements:
    1. Remove EXACTLY these {num_mods} pieces of content
    2. Remove from specified locations
    3. Make no other changes
    4. Return the complete modified note""",

                'omit_ir': """Apply these {num_mods} non-relevant information removals to the note:

    {criteria_text}

    Modifications to apply:
    {modifications}

    Requirements:
    1. Remove EXACTLY these {num_mods} pieces of content
    2. Remove from specified locations
    3. Make no other changes
    4. Verify none meet relevancy criteria
    5. Return the complete modified note"""
            }

            # Get the appropriate prompt template
            prompt_template = prompt_templates.get(mod_type)
            if not prompt_template:
                raise ValueError(f"Invalid modification type: {mod_type}")

            # Format the prompt with the actual modifications
            modifications_text = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(mod_desc))
            prompt = prompt_template.format(
                num_mods=num_mods,
                criteria_text=criteria_text,
                modifications=modifications_text
            )

            # Apply the modifications using modify_note
            result = self.modify_note(note, prompt, note)
            
            if not result:
                print(f"Failed to apply modifications for {mod_type}")
                return None
                
            return result

        except Exception as e:
            print(f"Error applying modifications: {str(e)}")
            traceback.print_exc()
            return None
        
    def generate_variations(self, note: str, note_number: int) -> List[Dict]:
        """Generate variations of a note with 1-5 and groups of 5 (10, 15, 20, 25) modifications."""
        modifications = []
        
        # Add original note first
        modifications.append({
            'original_note_number': note_number,
            'new_note_name': 'original_note',
            'modified_text': note,
            'modifications': 'Original unmodified note',
            'processing_time': 0,
            'input_tokens': count_tokens(note, self.model_name),
            'output_tokens': 0,
            'total_tokens': count_tokens(note, self.model_name),
            'added_text': '',
            'removed_text': '',
            'model': self.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Step 1: Create initial AI rewrite
        print("\n1️⃣ Creating initial AI rewrite...")
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
        
        # Step 2: Get and display all proposed modifications
        print("\n2️⃣ Getting all proposed modifications...")
        all_modifications = self.get_and_display_modifications(ai_note)
        
        # Debug print for suggestion counts
        for mod_type in all_modifications:
            count = len(all_modifications[mod_type])
            print(f"\nReceived {count} suggestions for {mod_type}")
        
        # Step 3: Apply modifications
        categories = [
            ('inj_r', '3️⃣ Applying relevant injections...'),
            ('inj_ir', '4️⃣ Applying irrelevant injections...'),
            ('omit_r', '5️⃣ Applying relevant omissions...'),
            ('omit_ir', '6️⃣ Applying irrelevant omissions...')
        ]
        
        # Define all modification sizes we want (1-5 and groups of 5)
        mod_sizes = [1, 2, 3, 4, 5, 10, 15, 20, 25]
        
        for mod_type, step_message in categories:
            print(f"\n{step_message}")
            suggestions = all_modifications.get(mod_type, [])
            
            print(f"Processing {len(suggestions)} suggestions for {mod_type}")
            
            if not suggestions:
                print(f"No modifications found for {mod_type}")
                continue
            
            # Calculate maximum possible modifications
            max_possible = len(suggestions)
            
            # Filter mod_sizes to only those we can actually do
            valid_sizes = [size for size in mod_sizes if size <= max_possible]
            print(f"Valid modification sizes for {mod_type}: {valid_sizes}")
            
            for i in valid_sizes:
                print(f"\nApplying {i} {mod_type} modifications:")
                current_suggestions = suggestions[:i]
                print(f"Number of suggestions being applied: {len(current_suggestions)}")
                
                result = self.apply_modification(ai_note, mod_type, i, current_suggestions)
                
                if result:
                    mod_entry = {
                        'original_note_number': note_number,
                        'new_note_name': f'AI_{mod_type}{i}',
                        'modified_text': result['modified_text'],
                        'modifications': '\n'.join(f"{j+1}. {s['Text']}" for j, s in enumerate(current_suggestions)),
                        'processing_time': result['processing_time'],
                        'input_tokens': result['input_tokens'],
                        'output_tokens': result['output_tokens'],
                        'total_tokens': result['total_tokens'],
                        'model': self.model_name,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'added_text': '\n'.join(result['added_sentences']),
                        'removed_text': '\n'.join(result['removed_sentences'])
                    }
                    
                    modifications.append(mod_entry)
                    print(f"✅ Successfully created {mod_entry['new_note_name']}")
                    
                    if result['added_sentences']:
                        print("\nVerified additions:")
                        for sent in result['added_sentences']:
                            print(f"  + {sent}")
                    if result['removed_sentences']:
                        print("\nVerified removals:")
                        for sent in result['removed_sentences']:
                            print(f"  - {sent}")
                else:
                    print(f"❌ Failed to create {f'AI_{mod_type}{i}'}")
                
                print("\n" + "="*50)
        
        print("\nGenerated notes:")
        for mod in modifications:
            print(f"- {mod['new_note_name']}")
        
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
                column_names = df.columns.tolist()
                if len(column_names) > 1:
                    column_names[1] = 'note_text'
                    df.columns = column_names
                
            return df
        except Exception as e:
            raise ValueError(f"Error reading input CSV: {str(e)}")

    def process_notes(self, start_note: int = 0, end_note: int = 1) -> None:
        """Process notes from start_note to end_note indices."""
        try:
            df = self.read_input_csv()
            
            start_note = max(0, start_note)
            end_note = min(len(df), end_note)
            
            df = df.iloc[start_note:end_note]
            all_modifications = []

            for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing notes"):
                note_number = index
                print(f"\nProcessing note {note_number} (Note {index+1} of {len(df)})")
                
                modifications = self.processor.generate_variations(row['note_text'], note_number)
                all_modifications.extend(modifications)

                self.save_modifications(all_modifications)

        except Exception as e:
            print(f"Error in note processing pipeline: {str(e)}")
            traceback.print_exc()

    def save_modifications(self, modifications: List[Dict]) -> None:
        """Save modifications to CSV file, appending if file exists."""
        output_csv = os.path.join(self.output_path, "modified_notes.csv")
        
        df_new = pd.DataFrame(modifications)
        
        if os.path.exists(output_csv):
            df_existing = pd.read_csv(output_csv)
            
            for col in df_new.columns:
                if col not in df_existing.columns:
                    df_existing[col] = None
            
            existing_triples = set(zip(df_existing['original_note_number'], 
                                    df_existing['new_note_name'],
                                    df_existing['model']))
            new_triples = set(zip(df_new['original_note_number'], 
                                df_new['new_note_name'],
                                df_new['model']))
            
            duplicates = new_triples.intersection(existing_triples)
            if duplicates:
                print(f"Found {len(duplicates)} duplicate entries - these will be skipped.")
                
                mask = ~df_new.apply(
                    lambda x: (x['original_note_number'], x['new_note_name'], x['model']) in duplicates, 
                    axis=1
                )
                df_new = df_new[mask]
            
            if not df_new.empty:
                df_new = df_new[df_existing.columns]
                df_new.to_csv(output_csv, mode='a', header=False, index=False)
                print(f"Appended {len(df_new)} new modifications to {output_csv}")
            else:
                print("No new modifications to append.")
        else:
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