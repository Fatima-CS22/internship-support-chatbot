# 💠 Internship Support Chatbot

An AI-powered support assistant that answers intern queries instantly using semantic search built as part of the Internee.pk ML internship.

## 📌 Objective

Develop an AI chatbot to automate real-time responses to common intern queries task submission, deadlines, certificates, technical issues, etc. reducing the need for manual back-and-forth with mentors and support staff.

## 🚀 Live Demo

(https://internship-support-chatbot-lfbk8hn4z9lzu7zmawyej9.streamlit.app/)

## 🧠 How It Works

This is a **retrieval-based chatbot**, not a generative one every response comes directly from a verified answer set, so the bot never "makes up" an answer.

1. A user's message is converted into a numerical embedding using a Sentence-BERT model
2. That embedding is compared against a pool of pre-embedded FAQ questions and support ticket messages using **cosine similarity**
3. The closest matching entry's linked answer is retrieved and returned instantly
4. If no match is confident enough below a similarity threshold, the bot responds with a fallback message instead of guessing

This approach was chosen over a purely generative one after testing showed generative rephrasing (Flan-T5) occasionally altered or fabricated details for a support chatbot, factual accuracy matters more than varied phrasing.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embedding Model | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Similarity Matching | Cosine Similarity (scikit-learn) |
| Interface | Streamlit |
| Data Handling | Pandas, NumPy |
| Development | Google Colab (training) + VS Code (app) |

## 📂 Project Structure

```
internship-support-chatbot/
├── app.py                    # Streamlit chatbot application
├── requirements.txt          # Python dependencies
├── chatbot_artifacts.pkl     # Pre-computed embeddings + lookup tables
├── answers.csv               # Unique verified answers
├── questions.csv             # FAQ-style question variations
├── support_tickets.csv       # Informal/realistic support ticket messages
├── internship_support_chatbot.ipynb   # Training notebook (Colab)
└── README.md
```

## 📊 Dataset

The dataset was custom-built to cover 20 common intern support topics: task submission, deadlines, mentor contact, certificates, attendance, profile updates, dashboard issues, login problems, password reset, internship duration, task evaluation, resubmission, feedback, completion status, technical issues, team projects, GitHub submission, Streamlit deployment, project requirements and AI usage policy.

It is organized as a normalized structure:
- **`answers.csv`**  one row per unique answer (`answer_id`, `category`, `answer`)
- **`questions.csv`**  multiple phrasings of common FAQ questions each linked to an `answer_id`
- **`support_tickets.csv`**  realistic, informally worded user messages each linked to an `answer_id`

This design keeps the dataset scalable — updating an answer only requires editing one row, regardless of how many question variations point to it.

## ⚙️ Setup & Installation

1. Clone the repository
   ```bash
   git clone https://github.com/Fatima-CS22/internship-support-chatbot.git
   cd internship-support-chatbot
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app
   ```bash
   streamlit run app.py
   ```

## 🧪 Model Training

The full training pipeline (data loading, EDA, preprocessing, embedding generation and artifact export) is documented in `internship_support_chatbot.ipynb` which was run on Google Colab. It produces `chatbot_artifacts.pkl` which the Streamlit app loads directly.

## 👩‍💻 Author
Fatima Waseem
BS Computer Science, MUST | ML Intern Internee.pk
[GitHub](https://github.com/Fatima-CS22) · [LinkedIn](https://linkedin.com/in/fatima-waseem-608604335)