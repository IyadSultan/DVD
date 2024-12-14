# DVD (Discharge Variation Detection) Evaluation System

## Overview
The DVD Evaluation System is a tool designed to assess and compare medical discharge notes using Multiple Choice Questions (MCQs). It uses advanced language models to generate questions, evaluate responses, and measure the consistency and completeness of medical documentation.

## Features
- Automated MCQ generation from medical notes
- Relevancy-based question filtering
- Batch processing of questions for efficiency
- Token usage tracking
- Detailed evaluation metrics
- CSV output for analysis

## Prerequisites
- Python 3.8+
- OpenAI API key
- Required Python packages (install via `pip install -r requirements.txt`):
  - langchain
  - openai
  - pydantic
  - tiktoken
  - tqdm
  - python-dotenv

## Installation
1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage
### Basic Usage
```bash
python dvd_evaluation.py /path/to/notes/folder output.csv
```

### Folder Structure
Your input folder should be organized as follows:
```
notes_folder/
├── subfolder1/
│   ├── AI.txt
│   ├── note1.txt
│   └── note2.txt
├── subfolder2/
│   ├── AI.txt
│   └── note1.txt
```

## Key Components

### MCQ Generation
The system generates MCQs based on clinically relevant criteria including:
- Hospital Admission/Discharge Details
- Reason for Hospitalization
- Hospital Course Summary
- Discharge Diagnosis
- Procedures and Imaging
- And more...

Reference: 
```python:dvd_evaluation.py
startLine: 60
endLine: 133
```

### Question Processing
Questions are processed in batches of 20 for efficiency, with each batch containing:
- Question text
- 5 options (A-D + "I don't know")
- Correct answer
- Relevance score

Reference:
```python:dvd_evaluation.py
startLine: 201
endLine: 254
```

## Output
The system generates a CSV file containing:
- Subfolder name
- Note name
- AI score
- Note score
- Word counts
- Question details

## Token Optimization
The system includes several optimizations to reduce token usage:
1. Batch processing of MCQs
2. Focused relevancy criteria
3. Streamlined prompts
4. Efficient response parsing

## Error Handling
The system includes robust error handling for:
- File I/O operations
- API responses
- MCQ parsing
- Response validation

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License

## Acknowledgments
- OpenAI for the GPT-4 API
- LangChain for the framework
- Contributors and testers

## Contact
For support or questions, please open an issue in the repository.
