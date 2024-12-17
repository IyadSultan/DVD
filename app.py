from flask import Flask, render_template, request, jsonify
import os
import tempfile
import pandas as pd
from werkzeug.utils import secure_filename
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import tiktoken
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define data models
class MCQ(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    source_name: str = Field(default="Unknown")

class Document(BaseModel):
    name: str = ''
    content: str
    mcqs: List[MCQ] = Field(default_factory=list)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'txt'}
MODELS = ['gpt-4o', 'gpt-4o-mini']

with open('note_criteria.json', 'r') as f:
    NOTE_CRITERIA = json.load(f)['note_types']  # Note the ['note_types'] key

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def generate_mcqs_for_note(note_content: str, total_tokens: List[int], source_name: str = '', document_type: str = 'discharge_note') -> List[MCQ]:
    """
    Generate Multiple Choice Questions (MCQs) from medical notes.
    """
    # Get relevancy criteria for selected document type
    criteria = NOTE_CRITERIA[document_type]['relevancy_criteria']
    criteria_list = "\n".join(f"{i+1}. {criterion}" for i, criterion in enumerate(criteria))
    
    system_prompt = f"""
You are an expert in creating MCQs based on medical notes. Generate 20 MCQs that ONLY focus on these key areas:
{criteria_list}

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
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create MCQs from this note:\n\n{note_content}")
        ]
        
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        response = llm(messages)
        
        # Update token count
        tokens_used = num_tokens_from_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": note_content},
            {"role": "assistant", "content": response.content}
        ], model="gpt-4")
        total_tokens[0] += tokens_used

        # Parse MCQs from response
        mcqs = []
        for mcq_text in response.content.strip().split('\n\n'):
            if mcq := parse_mcq(mcq_text):
                mcqs.append(mcq)

        return mcqs

    except Exception as e:
        print(f"Error in MCQ generation: {str(e)}")
        return []

def present_mcqs_to_content(mcqs: List[MCQ], content: str, total_tokens: List[int]) -> List[Dict]:
    """
    Present MCQs to content and collect responses.
    """
    user_responses = []
    batch_size = 20
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
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
                # If there are more lines than questions, break
                if j >= len(batch_mcqs):
                    break

                mcq = batch_mcqs[j]
                try:
                    # line should look like "1: A"
                    answer_letter = line.split(':')[1].strip().upper()
                except (IndexError, ValueError):
                    answer_letter = 'E'  # Fallback to "don't know"

                # Convert letter to actual text
                if answer_letter in ['A', 'B', 'C', 'D']:
                    user_answer_text = mcq.options[ord(answer_letter) - ord('A')]
                else:
                    user_answer_text = "I don't know"

                user_responses.append({
                    "question": mcq.question,
                    "user_answer": user_answer_text,   # <-- store text
                    "correct_answer": mcq.correct_answer,  # <-- store text
                })

        except Exception as e:
            print(f"Error processing batch responses: {str(e)}")
            # If something fails, default the remainder to "I don't know"
            for mcq in batch_mcqs[len(user_responses):]:
                user_responses.append({
                    "question": mcq.question,
                    "user_answer": "I don't know",
                    "correct_answer": mcq.correct_answer,
                })

    return user_responses


def run_evaluation(ai_content: str, ai_mcqs: List[MCQ], note_content: str, note_mcqs: List[MCQ], 
                  note_name: str, original_note_number: int, total_tokens: List[int]) -> List[Dict]:
    """
    Run evaluation for a pair of notes.
    
    Parameters:
    - ai_content: Content of the first document
    - ai_mcqs: MCQs generated from the first document (20 questions)
    - note_content: Content of the second document
    - note_mcqs: MCQs generated from the second document (20 questions)
    - note_name: Name of the second document
    - original_note_number: Original note number for tracking
    - total_tokens: List to track token usage
    
    Returns:
    - List of evaluation results
    """
    # Combine MCQs from both documents
    all_mcqs = ai_mcqs + note_mcqs  # Total 40 questions

    # Present all 40 MCQs to both documents
    ai_responses = present_mcqs_to_content(all_mcqs, ai_content, total_tokens)
    note_responses = present_mcqs_to_content(all_mcqs, note_content, total_tokens)
    
    results = []
    for i, mcq in enumerate(all_mcqs):

        # We used to do some letter-based logic here for correct answer, but now they are text fields.
        results.append({
            "original_note_number": original_note_number,
            "new_note_name": note_name,
            "question": mcq.question,
            "source_document": mcq.source_name,
            "snippet_doc1": getattr(mcq, 'snippet_doc1', None),  # or some logic to retrieve snippet
            "snippet_doc2": getattr(mcq, 'snippet_doc2', None),
            # "ideal_answer" can just be mcq.correct_answer, since that's your gold standard
            "ideal_answer": mcq.correct_answer,

            # Instead of letter, store text
            "correct_answer": ai_responses[i]["correct_answer"],
            "ai_answer": ai_responses[i]["user_answer"],
            "note_answer": note_responses[i]["user_answer"],

            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    return results

import concurrent.futures

import concurrent.futures
import csv
import os
from flask import jsonify, request

@app.route('/compare', methods=['POST'])
def compare_documents():
    """Compare two documents by generating MCQs and evaluating them in parallel,
       then return JSON analysis plus the original document contents."""
    if 'doc1' not in request.files or 'doc2' not in request.files:
        return jsonify({'error': 'Both documents are required'}), 400
    
    doc1 = request.files['doc1']
    doc2 = request.files['doc2']
    model = request.form.get('model', MODELS[0])
    document_type = request.form.get('document_type', 'discharge_note')
    
    if not all([doc1.filename, doc2.filename]):
        return jsonify({'error': 'Both documents are required'}), 400
    
    if not all(allowed_file(doc.filename) for doc in [doc1, doc2]):
        return jsonify({'error': 'Invalid file type. Only .txt files are allowed'}), 400

    try:
        # Save documents to a temporary CSV if needed
        temp_csv = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_modified_notes.csv')
        
        # Read document contents
        doc1_content = doc1.read().decode('utf-8')
        doc2_content = doc2.read().decode('utf-8')
        
        # Write the two documents into a CSV (though your final logic may or may not need this)
        with open(temp_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['original_note_number', 'new_note_name', 'modified_text'])
            writer.writerow([0, 'Doc1', doc1_content])
            writer.writerow([0, 'Doc2', doc2_content])

        # Initialize a shared token counter if needed
        total_tokens = [0]

        # --------------------------------------------------------------
        # STEP 1: PARALLEL MCQ GENERATION
        # --------------------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_doc1_mcqs = executor.submit(
                generate_mcqs_for_note, 
                doc1_content, 
                total_tokens, 
                'Doc1', 
                document_type
            )
            future_doc2_mcqs = executor.submit(
                generate_mcqs_for_note, 
                doc2_content, 
                total_tokens, 
                'Doc2', 
                document_type
            )
            
            mcqs_doc1 = future_doc1_mcqs.result()
            mcqs_doc2 = future_doc2_mcqs.result()

        # --------------------------------------------------------------
        # STEP 2: PARALLEL EVALUATION (ANSWERING MCQs FOR BOTH DOCS)
        # --------------------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_doc1_results = executor.submit(
                run_evaluation,
                doc1_content, mcqs_doc1,  # doc1 as AI content + doc1 MCQs
                doc2_content, mcqs_doc2,  # doc2 as note content + doc2 MCQs
                'Doc2',                   # new_note_name for doc2
                0,                        # original_note_number
                total_tokens
            )
            
            future_doc2_results = executor.submit(
                run_evaluation,
                doc2_content, mcqs_doc2,  # doc2 as AI content + doc2 MCQs
                doc1_content, mcqs_doc1,  # doc1 as note content + doc1 MCQs
                'Doc1',
                0,
                total_tokens
            )
            
            doc1_results = future_doc1_results.result()
            doc2_results = future_doc2_results.result()

        # --------------------------------------------------------------
        # STEP 3: ANALYZE & APPEND DOCUMENT CONTENTS
        # --------------------------------------------------------------
        analysis = analyze_results(doc1_results, doc2_results)
        
        # Add raw notes to JSON output so the front-end can display them side by side
        analysis['doc1_content'] = doc1_content
        analysis['doc2_content'] = doc2_content
        
        return jsonify(analysis)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



def analyze_results(doc1_results, doc2_results):
    def process_document_results(results, doc_name):
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r['ai_answer'] == r['correct_answer'])

        self_questions = [r for r in results if r['source_document'] == doc_name]
        other_questions = [r for r in results if r['source_document'] != doc_name]

        # Replace "..." with actual logic or an empty list
        # e.g. basic logic:
        self_mistakes = [
            r for r in self_questions 
            if (r['ai_answer'] != r['correct_answer'] and r['ai_answer'] != 'I don’t know')
        ]
        other_mistakes = [
            r for r in other_questions 
            if (r['ai_answer'] != r['correct_answer'] and r['ai_answer'] != 'I don’t know')
        ]
        unknown_answers = [
            r for r in other_questions if r['ai_answer'] == 'I don’t know'
        ]

        return {
            'total_score': f"{correct_answers}/{total_questions}",
            'self_mistakes': [
                {
                    'question': r['question'],
                    'ideal_answer': r['ideal_answer'],
                    'model_answer': r['ai_answer'],
                    'snippet_doc1': r.get('snippet_doc1', ''),
                    'snippet_doc2': r.get('snippet_doc2', '')
                }
                for r in self_mistakes
            ],
            'other_mistakes': [
                {
                    'question': r['question'],
                    'ideal_answer': r['ideal_answer'],
                    'model_answer': r['ai_answer'],
                    'snippet_doc1': r.get('snippet_doc1', ''),
                    'snippet_doc2': r.get('snippet_doc2', '')
                }
                for r in other_mistakes
            ],
            'unknown_answers': [
                {
                    'question': r['question'],
                    'ideal_answer': r['ideal_answer'],
                    'snippet_doc1': r.get('snippet_doc1', ''),
                    'snippet_doc2': r.get('snippet_doc2', '')
                }
                for r in unknown_answers
            ]
        }
    
    return {
        'doc1_analysis': process_document_results(doc1_results, 'Doc1'),
        'doc2_analysis': process_document_results(doc2_results, 'Doc2')
    }


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', 
                         models=MODELS,
                         document_types=NOTE_CRITERIA)

if __name__ == '__main__':
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create index.html in templates directory if it doesn't exist
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Comparison Tool</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 p-8">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-3xl font-bold mb-8">Document Comparison Tool</h1>
        
        <!-- Upload Form -->
        <form id="uploadForm" class="bg-white p-6 rounded-lg shadow-md mb-8">
            <div class="grid grid-cols-2 gap-6 mb-6">
                <div>
                    <label class="block text-sm font-medium mb-2">Document 1</label>
                    <input type="file" name="doc1" accept=".txt" required
                           class="w-full border rounded p-2">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-2">Document 2</label>
                    <input type="file" name="doc2" accept=".txt" required
                           class="w-full border rounded p-2">
                </div>
            </div>
            
            <div class="mb-6">
                <label class="block text-sm font-medium mb-2">Model</label>
                <select name="model" class="w-full border rounded p-2">
                    {% for model in models %}
                    <option value="{{ model }}">{{ model }}</option>
                    {% endfor %}
                </select>
            </div>
            
            <div class="mb-6">
                <label class="block text-sm font-medium mb-2">Document Type</label>
                <select name="document_type" class="w-full border rounded p-2">
                    {% for type_id, type_info in document_types.items() %}
                    <option value="{{ type_id }}">{{ type_info.name }}</option>
                    {% endfor %}
                </select>
            </div>
            
            <button type="submit" 
                    class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                Compare Documents
            </button>
        </form>

        <!-- Loading indicator -->
        <div id="loading" class="hidden">
            <div class="text-center py-4">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
                <p class="mt-2">Processing documents... May take few minutes.</p>
            </div>
        </div>
        
        <!-- Results Section -->
        <div id="results" class="hidden">
            <div class="grid grid-cols-2 gap-6">
                <!-- Document 1 Results -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-bold mb-4">Document 1 Results</h2>
                    <div id="doc1Results"></div>
                </div>
                
                <!-- Document 2 Results -->
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-bold mb-4">Document 2 Results</h2>
                    <div id="doc2Results"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            loading.classList.remove('hidden');
            results.classList.add('hidden');
            
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/compare', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults('doc1Results', data.doc1_analysis);
                    displayResults('doc2Results', data.doc2_analysis);
                    results.classList.remove('hidden');
                } else {
                    alert(data.error || 'An error occurred');
                }
            } catch (error) {
                alert('An error occurred while processing the documents');
            } finally {
                loading.classList.add('hidden');
            }
        });

        function displayResults(elementId, analysis) {
            const container = document.getElementById(elementId);
            
            container.innerHTML = `
                <div class="mb-4">
                    <h3 class="font-bold">Total Score:</h3>
                    <p>${analysis.total_score}</p>
                </div>
                
                <div class="mb-4">
                    <h3 class="font-bold">Self Questions Mistakes:</h3>
                    ${renderQuestionList(analysis.self_mistakes)}
                </div>
                
                <div class="mb-4">
                    <h3 class="font-bold">Other Document Mistakes:</h3>
                    ${renderQuestionList(analysis.other_mistakes)}
                </div>
                
                <div class="mb-4">
                    <h3 class="font-bold">Unknown Answers:</h3>
                    ${renderQuestionList(analysis.unknown_answers, true)}
                </div>
            `;
        }

        function renderQuestionList(questions, isUnknown = false) {
    if (!questions.length) {
        return '<p class="text-gray-500">None</p>';
    }
    
    return questions.map((q, idx) => {
        // We'll store the snippet in a hidden div and toggle it on click
        const questionId = `question-${Math.random().toString(36).slice(2)}`;
        
        return `
            <div class="mb-2 p-2 bg-gray-50 rounded" id="${questionId}">
                <button 
                    class="font-medium text-left w-full"
                    onclick="toggleSnippet('${questionId}')"
                >
                    ${q.question}
                </button>
                        <p class="text-sm">Ideal Answer: ${q.ideal_answer}</p>
                        ${!isUnknown ? `<p class="text-sm">Model Answer: ${q.model_answer}</p>` : ''}

                        <!-- Hidden snippet container -->
                        <div class="hidden mt-2 p-2 border-l-4 border-blue-300" id="${questionId}-snippet">
                            <h4 class="font-bold mb-1">Relevant Snippet (Doc1):</h4>
                            <p class="text-sm mb-2">${q.snippet_doc1 || 'No snippet found'}</p>
                            
                            <h4 class="font-bold mb-1">Relevant Snippet (Doc2):</h4>
                            <p class="text-sm">${q.snippet_doc2 || 'No snippet found'}</p>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // JavaScript function to toggle snippet visibility
        function toggleSnippet(questionId) {
            const snippetDiv = document.getElementById(`${questionId}-snippet`);
            if (snippetDiv.classList.contains('hidden')) {
                snippetDiv.classList.remove('hidden');
            } else {
                snippetDiv.classList.add('hidden');
            }
        }
    </script>
</body>
</html>
""")
    
    app.run(debug=True)
    