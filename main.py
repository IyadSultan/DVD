# Choose model, start, end
# python note_modifier.py --input samples/discharge_samples_200.csv --output modified_notes --result_csv results --start 0 --end 1 --model gpt-4

import argparse
import os
import pandas as pd
from datetime import datetime
from note_modifier import NoteModificationPipeline
from dvd_evaluator import run_evaluation

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process and modify medical notes with variations.')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file containing notes')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory for modified notes (or a CSV file if you prefer)')
    parser.add_argument('--model', type=str, default='gpt-4', help='Name of the OpenAI model to use')
    parser.add_argument('--start', type=int, default=0, help='Start note index (default: 0)')
    parser.add_argument('--end', type=int, default=1, help='End note index (default: 1)')
    parser.add_argument('--result_csv', type=str, default='results', help='Output directory for evaluation results')
    
    args = parser.parse_args()

    # Construct output file names with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'{args.output}/modified_notes_{args.model}_{args.start}_to_{args.end}_{timestamp}.csv'
    result_file = f'{args.result_csv}/results_{args.model}_{args.start}_to_{args.end}_{timestamp}.csv'

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    # Initialize the note modification pipeline
    pipeline = NoteModificationPipeline(args.input, output_file, args.model)

    # Process notes from start to end
    pipeline.process_notes(args.start, args.end)

    # Run evaluation on the modified notes
    run_evaluation(
        output_file,
        args.result_csv,
        start=args.start,
        end=args.end,
        model=args.model
    )

if __name__ == "__main__":
    main()


#python main.py --input samples/discharge_samples_200.csv --output modified_notes --result_csv results --start 0 --end 1 --model gpt-4o-mini 