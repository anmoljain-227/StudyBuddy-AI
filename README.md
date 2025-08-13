# StudyBuddy-AI
StudyBuddy AI is a Streamlit app that lets you upload PDF/DOCX/TXT files, index them into a local vector store (ChromaDB) using Sentence Transformers, and ask questions with Google Gemini. It also generates concise summaries and auto-creates MCQ quizzes straight from your documents.

StudyBuddy-AI is an AI-powered study assistant that helps users interact with their uploaded study materials in a conversational way. Designed for students, educators, and professionals, it leverages Google Gemini AI for generating accurate, context-aware answers, summaries, and quizzes directly from the uploaded documents.

# Key Features
📂 Upload & Process Files - Upload PDF study materials to get instant AI-driven insights.

💬 Q&A Chat Interface – Ask any question from your uploaded content and get precise, context-rich answers.

📝 Automatic Summarization – Generate concise summaries without needing manual input.

🎯 Auto Quiz Generation – Create AI-generated quizzes from the uploaded material to test understanding.

📱 Responsive UI – A clean, two-column layout with a dedicated chat area and document utilities for easy navigation.

⚡ Powered by Gemini AI – Ensures natural, accurate, and relevant responses.

# Workflow of StudyBuddy-AI
## User Uploads a Document

<img width="1906" height="1003" alt="Screenshot 2025-08-12 164004" src="https://github.com/user-attachments/assets/50b08059-7a46-42be-8cc7-1641197cab7d" />

The user uploads a PDF study file.

The file is processed using pdfplumber to extract clean text.

## Text Processing & Storage
<img width="1883" height="764" alt="Screenshot 2025-08-12 164145" src="https://github.com/user-attachments/assets/b6042246-bd25-4521-b8fc-1e178021c9b6" />
<img width="1912" height="800" alt="Screenshot 2025-08-12 163757" src="https://github.com/user-attachments/assets/218a594b-f840-4504-a265-24db032f26b1" />


Extracted text is split into chunks for better AI understanding.

The chunks are stored temporarily in memory for instant retrieval.

## User Interaction (Chat)
<img width="1903" height="887" alt="Screenshot 2025-08-12 163718" src="https://github.com/user-attachments/assets/65d6edb2-c230-4da4-959b-221b5d3aa28b" />
<img width="1897" height="803" alt="Screenshot 2025-08-13 125740" src="https://github.com/user-attachments/assets/cfbffbc9-0503-4047-a4a7-37302dbf5056" />

Users ask questions in the right-side chat interface.

Gemini AI searches relevant chunks and generates an answer.

## Summarization
<img width="1912" height="800" alt="Screenshot 2025-08-12 163757" src="https://github.com/user-attachments/assets/7ba06214-445c-45ba-9ae2-50dce4f609dc" />


Users click "Summarize" to instantly get a condensed summary of the uploaded content.

No input required — AI generates the summary automatically.

## Quiz Generation
<img width="1853" height="753" alt="Screenshot 2025-08-13 125846" src="https://github.com/user-attachments/assets/603d87b1-26f3-4772-a49a-db4203d9f5a1" />
<img width="1865" height="774" alt="Screenshot 2025-08-13 130010" src="https://github.com/user-attachments/assets/f21ed930-e25c-40ce-b43e-f97df72fa864" />
<img width="1852" height="750" alt="Screenshot 2025-08-13 130041" src="https://github.com/user-attachments/assets/e7c6c959-c75f-467e-b3a1-b41e9d688b86" />



The "Generate Quiz" button creates multiple-choice or short-answer questions from the uploaded document.

Ideal for self-assessment.

## Display Results
<img width="1907" height="743" alt="Screenshot 2025-08-13 233312" src="https://github.com/user-attachments/assets/b7bb3f5b-e562-4ed8-8a2a-7edc757fee53" />

All responses (Q&A, summary, quiz) are shown in the chat panel with a conversational flow.
