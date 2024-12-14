import os
import csv
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def setup_folders(modified_folder):
    """Create necessary folders if they don't exist."""
    os.makedirs(modified_folder, exist_ok=True)

def load_environment():
    """Load environment variables."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

def initialize_llm():
    """Initialize the language model."""
    return ChatOpenAI(model="gpt-4o", temperature=0.7)

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
        "Important Abnormal (not normal)lab results, e.g. bacterial cultures, urine cultures, electrolyte disturbances, etc.",
        "Important abnormal vital signs, e.g. fever, tachycardia, hypotension, etc.",
        "Admission to ICU",
        "comorbidities, e.g. diabetes, hypertension, etc.",
        "Equipment needed at discharge, e.g. wheelchair, crutches, etc.",
        "Prosthetics and tubes, e.g. Foley catheter, etc.",
        "Allergies",
        "Consultations (e.g., specialty or ancillary services)",
        "Functional Capacity (ADLs and mobility status)",
        "Lifestyle Modifications (diet, exercise, smoking cessation, etc.)",
        "Wound Care or Other Specific Care Instructions",
        ]

def create_relevancy_prompt():
    """Create the prompt for defining relevant information."""
    criteria = get_relevancy_criteria()
    prompt = (
        "You are a medical note rewriter. You need to understand the meaning of relevant information. "
        "The following are the key pieces of information that should be included in a patient discharge summary, "
        "they are defined as relevant information:\n"
    )
    for i, criterion in enumerate(criteria, 1):
        prompt += f"{i}. {criterion},\n"
    print("Relevancy prompt: ", prompt)
    return prompt

def modify_note_with_prompt(llm, note, prompt, relevancy_prompt):
    """Modify a note using a specific prompt."""
    try:
        print(f"\n🔄 Applying prompt: {prompt[:50]}...")  # Show first 50 chars of prompt
        response = llm.invoke([
            SystemMessage(content=relevancy_prompt),
            HumanMessage(content=f"{prompt}\n\nThe note is: {note}")
        ])
        print("✅ Modification complete")
        return response.content.strip()
    except Exception as e:
        print(f"❌ Error modifying note: {str(e)}")
        return None

def generate_variations(llm, note, prefix, relevancy_prompt):
    """Generate variations of the note."""
    print("\n📝 Generating note variations...")
    variations = []
    modification_details = []
    
    variation_types = {
        'omit_r': 'Remove {} key pieces of relevant information from the note.',
        'omit_ir': 'Remove {} non-relevant details from the note.',
        'inj_r': 'Add {} additional synthetic relevant details to the note.',
        'inj_ir': 'Add {} synthetic non-relevant details to the note.'
    }
    
    for i in range(1, 6):
        for var_type, prompt_template in variation_types.items():
            print(f"\n🔄 Creating variation {prefix}{var_type}{i}")
            prompt = prompt_template.format(i)
            modified_note = modify_note_with_prompt(llm, note, prompt, relevancy_prompt)
            
            if modified_note:
                var_name = f"{prefix}{var_type}{i}"
                variations.append((var_name, modified_note))
                print(f"✅ Successfully created {var_name}")
                
                modification_details.append({
                    'variation_name': var_name,
                    'modification_type': var_type,
                    'modification_count': i,
                    'success': True
                })

                # Debugging: Print and Save each variation immediately
                print(f"Debug: Variation Content: {modified_note[:50]}...")  # Show first 50 chars
                var_path = os.path.join(os.getcwd(), f"{var_name}.txt")
                try:
                    with open(var_path, 'w', encoding='utf-8') as file:
                        file.write(modified_note)
                    print(f"✅ Successfully saved variation to {var_path}")
                except Exception as e:
                    print(f"❌ Failed to save variation {var_name}: {str(e)}")
            else:
                print(f"❌ Failed to create {prefix}{var_type}{i}")
                modification_details.append({
                    'variation_name': f"{prefix}{var_type}{i}",
                    'modification_type': var_type,
                    'modification_count': i,
                    'success': False
                })
    
    return variations, modification_details

def process_note(filename, original_folder, modified_folder, llm):
    """Process a single note file."""
    # Convert to absolute paths
    original_folder = os.path.abspath(original_folder)
    modified_folder = os.path.abspath(modified_folder)
    
    filepath = os.path.join(original_folder, filename)
    note_folder = os.path.join(modified_folder, os.path.splitext(filename)[0])
    
    print(f"\n📂 Working with directories:")
    print(f"Original folder: {original_folder}")
    print(f"Modified folder: {modified_folder}")
    print(f"Note folder: {note_folder}")
    
    try:
        # Create output directory
        print(f"Creating directory: {note_folder}")
        os.makedirs(note_folder, exist_ok=True)
        
        # Read the original note
        print(f"📖 Reading from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as file:
            original_note = file.read()
        
        # Create relevancy prompt
        relevancy_prompt = create_relevancy_prompt()
        
        # Generate the AI rewritten note
        print("🤖 Creating AI rewritten version...")
        system_message = "Rewrite the note professionally, omitting normal lab values and normal vital signs."
        
        response = llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=f"Original Note:\n\n{original_note}")
        ])
        
        ai_note = response.content.strip()
        print("✅ AI rewrite complete")
        
        # Save original and AI notes
        original_output = os.path.join(note_folder, "original.txt")
        ai_output = os.path.join(note_folder, "AI.txt")
        
        print(f"💾 Saving original note to: {original_output}")
        with open(original_output, 'w', encoding='utf-8') as file:
            file.write(original_note)
        
        print(f"💾 Saving AI note to: {ai_output}")
        with open(ai_output, 'w', encoding='utf-8') as file:
            file.write(ai_note)
        
        # Generate variations using the relevancy_prompt
        variations, mod_details = generate_variations(llm, ai_note, "AI_", relevancy_prompt)
        
        # Save variations
        for var_name, var_content in variations:
            var_path = os.path.join(note_folder, f"{var_name}.txt")
            print(f"💾 Saving variation to: {var_path}")
            with open(var_path, 'w', encoding='utf-8') as file:
                file.write(var_content)
            print(f"✅ Saved {var_name}")
        
        # List all files in the directory after saving
        print("\n📂 Files created in output directory:")
        for file in os.listdir(note_folder):
            print(f"  - {file}")
        
        return mod_details
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return [{
            'variation_name': 'ERROR',
            'modification_type': 'error',
            'modification_count': 0,
            'success': False,
            'error_message': str(e)
        }]

def save_file(filepath, content):
    """Helper function to save files with error checking"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"✅ Successfully saved: {filepath}")
    except Exception as e:
        print(f"❌ Error saving {filepath}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Modify medical notes with variations.')
    parser.add_argument('--input', type=str, help='Input file or folder path', required=True)
    parser.add_argument('--output', type=str, help='Output folder path', required=True)
    args = parser.parse_args()
    
    try:
        print("🚀 Starting note modification process...")
        
        # Setup
        print("⚙️ Loading environment...")
        load_environment()
        print("📁 Setting up folders...")
        setup_folders(args.output)
        print("🤖 Initializing AI model...")
        llm = initialize_llm()
        
        # Determine if input is file or folder
        if os.path.isfile(args.input):
            files_to_process = [os.path.basename(args.input)]
            original_folder = os.path.dirname(args.input)
        else:
            files_to_process = [f for f in os.listdir(args.input) if f.endswith('.txt')]
            original_folder = args.input
        
        print(f"📝 Found {len(files_to_process)} files to process")
        
        all_modifications = []
        
        # Process each file
        for filename in tqdm(files_to_process, desc="Processing notes"):
            mod_details = process_note(filename, original_folder, args.output, llm)
            
            for detail in mod_details:
                detail['filename'] = filename
                all_modifications.append(detail)
        
        # Save modification details
        print("\n📊 Saving modification details...")
        csv_filepath = os.path.join(args.output, "modification_details.csv")
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'variation_name', 'modification_type', 
                         'modification_count', 'success', 'error_message']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for detail in all_modifications:
                writer.writerow(detail)
        
        print("✅ Notes processing complete!")
        print(f"📄 Modification details saved to {csv_filepath}")
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
