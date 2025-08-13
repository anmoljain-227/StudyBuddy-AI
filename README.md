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

The user uploads a PDF study file.

The file is processed using pdfplumber to extract clean text.

## Text Processing & Storage

Extracted text is split into chunks for better AI understanding.

The chunks are stored temporarily in memory for instant retrieval.

## User Interaction (Chat)

Users ask questions in the right-side chat interface.

Gemini AI searches relevant chunks and generates an answer.

## Summarization

Users click "Summarize" to instantly get a condensed summary of the uploaded content.

No input required — AI generates the summary automatically.

## Quiz Generation

The "Generate Quiz" button creates multiple-choice or short-answer questions from the uploaded document.

Ideal for self-assessment.

## Display Results

All responses (Q&A, summary, quiz) are shown in the chat panel with a conversational flow.
