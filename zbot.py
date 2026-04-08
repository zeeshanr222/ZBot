import openai
import datetime
import os
import textwrap
import re
import glob
import json
import shutil
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI

# === Configuration ===
# --- IMPORTANT ---
# Replace with your actual API key if needed, otherwise this is a placeholder.
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "meta-llama/llama-3-8b-instruct" # Using Llama 3 8B
FAISS_INDEX_PATH = "faiss_index" # Path to save/load the vector store

# --- Session Memory and Bot Output Helper ---
session_memory = {"name": None, "symptom_log": []}

def bot_print(message: str) -> None:
    """Prints a message from the bot, addressing the user by name if known."""
    if session_memory["name"]:
        print(f"Bot: {session_memory['name']}, {message}")
    else:
        print(f"Bot: {message}")

# === Document Loading ===
def load_documents_from_folder(folder_path):
    """
    Loads all supported documents (.txt, .pdf, .docx, .csv, .doc, .xlsx, .json) from a specified folder.
    """
    supported_loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredFileLoader,
        ".doc": UnstructuredFileLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredFileLoader,
        ".json": UnstructuredFileLoader, # Re-added JSON loader
    }
    docs = []
    print(f"--- Document Loading ---")
    print(f"Searching for documents in: {folder_path}")
    
    all_files = glob.glob(os.path.join(folder_path, '*.*'))
    if not all_files:
        print("DEBUG: No files found in the specified directory. Check the path and that files exist.")
        return []
    
    print(f"DEBUG: Found {len(all_files)} files: {[os.path.basename(f) for f in all_files]}")

    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in supported_loaders:
            try:
                print(f"→ Loading {os.path.basename(file_path)}")
                loader = supported_loaders[ext](file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️ Error loading {os.path.basename(file_path)}: {e}")
    
    print(f"--- End of Document Loading ---")
    return docs

# === Quiz Functionality ===
def load_quiz(file_path):
    """Loads a quiz from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            quiz_data = json.load(f)
        return quiz_data
    except FileNotFoundError:
        bot_print("I couldn't find the `quiz.json` file. Please make sure it's in the same folder as the script.")
        return None
    except json.JSONDecodeError:
        bot_print("The `quiz.json` file seems to be formatted incorrectly. Please check the file.")
        return None

def start_quiz(quiz_data):
    """Administers the quiz to the user."""
    score = 0
    total_questions = len(quiz_data)
    print("", end="\n")
    bot_print("Great! Let's start the quiz. Please choose the number for the correct answer.")
    print("-" * 50)

    for i, (question, details) in enumerate(quiz_data.items()):
        print(f"\nQuestion {i+1}: {question}")
        options = details["options"]
        correct_answer = details["correct"]
        
        for j, option in enumerate(options):
            print(f"  {j+1}. {option}")

        while True:
            try:
                user_answer_num = int(input("Your answer (number): "))
                if 1 <= user_answer_num <= len(options):
                    break
                else:
                    bot_print("Please enter a number from the options provided.")
            except ValueError:
                bot_print("That's not a valid number. Please try again.")

        user_answer_text = options[user_answer_num - 1]

        if user_answer_text == correct_answer:
            bot_print("That's correct! Well done.")
            score += 1
        else:
            bot_print(f"Not quite. The correct answer was: '{correct_answer}'")
        print("-" * 20)

    print("\n--- Quiz Complete ---")
    bot_print(f"You scored {score} out of {total_questions}. Great job!")
    print("-" * 50 + "\n")

# === RAG Chain Initialization ===
def create_rag_chain(force_reload=False):
    """
    Creates or loads the RAG chain. If force_reload is True, it rebuilds the vector store.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if we need to force a reload
    if force_reload and os.path.exists(FAISS_INDEX_PATH):
        print("Force reload requested. Deleting existing vector store...")
        shutil.rmtree(FAISS_INDEX_PATH)

    # Load from disk if it exists, otherwise create it
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing vector store from {FAISS_INDEX_PATH}...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")
    else:
        print("No existing vector store found. Creating a new one...")
        # Assuming the script and documents are in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw_documents = load_documents_from_folder(script_dir)
        
        if not raw_documents:
            print("\n❌ No documents were successfully loaded. Cannot create RAG chain.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        documents = text_splitter.split_documents(raw_documents)

        print("\nCreating embeddings and vector store... (This may take a moment)")
        vectorstore = FAISS.from_documents(documents, embedding_model)
        
        print(f"Saving vector store to {FAISS_INDEX_PATH}...")
        vectorstore.save_local(FAISS_INDEX_PATH)
        print("Vector store created and saved.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=0.7,
        max_tokens=1024
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    prompt_template = """
    You are a helpful and friendly NHS nurse supporting a patient doing peritoneal dialysis at home. Your role has two parts:
    1.  **Answering Questions:** When the patient asks a direct question, use the provided context from NHS documents to give a clear and accurate answer.
    2.  **Exploring Well-being:** Your goal is to understand the patient's experience with their training and the therapy itself. Be empathetic and focus on their confidence, understanding, and overall quality of life.

    **Rules for Answering Questions:**
    - Use ONLY the provided CONTEXT to answer the patient's question.
    - Provide a comprehensive and detailed answer, covering all relevant points from the context.
    - Do NOT guess or make up anything.
    - If the answer is not in the documents, say: "I'm sorry, I couldn't find that information in the documents I have."
    - Use bullet points or numbered steps for clarity.
    - Always be friendly and reassuring.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    return chain

# === Main Script Logic ===
rag_chain = create_rag_chain()

# === Chatbot Interactive Loop ===
if not rag_chain:
    print("\n--- Chatbot Failed to Initialize ---")
    print("Please check the errors above and restart the script.")
    exit()

training_review_questions = [
    "To start, how confident are you feeling about doing the therapy on your own?",
    "How would you describe your experience with the training you received?",
    "Do you feel you understand the key aspects of the treatment well?",
    "If a problem comes up, how confident do you feel about dealing with it?",
    "And finally, do you know how and when you should get advice if you need it?"
]

knowledge_check_questions = [
    "First, could you tell me what you think is the most important thing to remember to conduct the treatment safely?",
    "Do you know of any situations where you might need to vary the treatment?",
    "If there was a problem, like a machine alarm or a drain issue, what would you do?",
    "And when would be the right time to call your care team for help?"
]

print("\n--- Chatbot Initialized ---")
print("Hi, I'm your PD care assistant! I can answer questions about your therapy.")
print("You can also say 'review my training', 'check my knowledge', or 'take a quiz'.")
print("If you add new documents, type 'reload' to update my knowledge.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        bot_print("Goodbye! Take care!")
        break

    elif user_input.lower() == 'reload':
        print("\n", end="")
        bot_print("Reloading knowledge base... This may take a moment.")
        try:
            new_chain = create_rag_chain(force_reload=True)
        except Exception as e:
            bot_print(f"Failed to reload knowledge base ({e}). I'll keep using the previous version.")
        else:
            if new_chain:
                rag_chain = new_chain
                bot_print("Knowledge base reloaded successfully!")
            else:
                bot_print("Failed to reload knowledge base. I'll keep using the previous version.")
        print("-" * 50 + "\n")
        continue

    elif any(phrase in user_input.lower() for phrase in ["review my training", "training review"]):
        print("\n", end="")
        bot_print("Of course. Let's talk a bit about your training and how you're feeling...")
        for question in training_review_questions:
            bot_print(question)
            patient_answer = input("You: ")
            bot_print("Thank you for sharing that with me.")
        print("-" * 50)
        bot_print("That's really helpful, thank you. Is there anything else I can help with today?")
        print()
        continue

    elif any(phrase in user_input.lower() for phrase in ["check my knowledge", "knowledge check"]):
        print("\n", end="")
        bot_print("Great idea. Let's quickly go over a few key points...")
        for question in knowledge_check_questions:
            bot_print(question)
            patient_answer = input("You: ")
            bot_print("Got it, thank you.")
        print("-" * 50)
        bot_print("Excellent, thank you for confirming those points. Do you have any other questions?")
        print()
        continue
    
    elif "quiz" in user_input.lower():
        # Assuming the script and quiz.json are in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        quiz_file = os.path.join(script_dir, 'quiz.json')
        quiz_data = load_quiz(quiz_file)
        if quiz_data:
            start_quiz(quiz_data)
        continue

    # --- Simple Memory Features (Re-integrated) ---
    # Remember the user's name
    if "my name is" in user_input.lower():
        session_memory["name"] = user_input.split("is")[-1].strip().capitalize()
        bot_print(f"Nice to meet you, {session_memory['name']}. I'll remember your name.")
        continue

    # Log potential symptoms
    if any(kw in user_input.lower() for kw in ["pain", "fever", "cloudy", "red", "sore"]):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        session_memory["symptom_log"].append(f"{timestamp} - {user_input}")
        bot_print("Symptom logged. Remember to contact your care team if you are concerned.")


    # --- Main RAG Chain Invocation ---
    try:
        rag_response = rag_chain({"question": user_input})
        bot_reply = rag_response['answer'].strip()
        source_docs = rag_response.get('source_documents', [])

        cleaned_reply = re.sub(r' {2,}', ' ', bot_reply)
        print("\n", end="")
        bot_print(cleaned_reply)

        if source_docs:
            unique_sources = set()
            for doc in source_docs:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page_num = doc.metadata.get('page')
                if page_num is not None:
                    unique_sources.add(f"{source_name} (page {page_num + 1})")
                else:
                    unique_sources.add(source_name)
            print(f"\n[Source(s): {', '.join(sorted(list(unique_sources)))}]")

    except Exception as e:
        print("\n", end="")
        bot_print(textwrap.fill(f"Sorry, I encountered an error: {e}", width=120))

    print("\n" + "-" * 50 + "\n")