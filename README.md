# BookMind 🧠📚

> **Your AI-Powered Intelligent Reading Companion.**  
> *Discover, Summarize, and Track your favorite books with the power of Machine Learning.*

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Django](https://img.shields.io/badge/django-5.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📖 Overview

**BookMind** is a sophisticated web application designed to revolutionize how you interact with books. By leveraging advanced **Natural Language Processing (NLP)** and **Machine Learning** algorithms, BookMind provides personalized book recommendations, generates instant summaries for PDFs, and offers a premium, modern user interface for managing your reading journey.

Whether you're looking for your next great read or need a quick summary of a dense textbook, BookMind has you covered.

## ✨ Key Features

### 🤖 AI & Machine Learning
-   **Smart Recommendations**: A hybrid recommendation engine that combines **Content-Based Filtering** (using Semantic Embeddings) with **User Behavior Analysis** to suggest books you'll actually love.
-   **Instant Summarization**: Upload any PDF book and get an AI-generated summary using state-of-the-art parameters (T5/Transformers).
-   **Semantic Search**: Search for books not just by keywords, but by meaning.

### 🌐 Rich Content Integration
-   **OpenLibrary API**: Seamlessly integrates with OpenLibrary to fetch metadata, covers, and availability for millions of books.
-   **Hybrid Search**: Instantly searches both your local library and the global OpenLibrary database.

### 🎨 Modern UI/UX
-   **Premium Dark Theme**: A sleek "Dark Slate & Emerald" aesthetic designed for comfortable reading at any time.
-   **Responsive Design**: Built with **Tailwind CSS** for a flawless experience on desktop, tablet, and mobile.
-   **Glassmorphism**: Modern UI elements with blur effects and smooth animations.

### 👤 User Features
-   **Dashboard**: Track your reading progress, recently viewed books, and activity history.
-   **Social**: Like, bookmark, and review books to refine your personal recommendation profile.
-   **Profile Management**: Customize your profile with a bio, avatar, and improved, secure authentication.

## 🛠️ Tech Stack

-   **Backend Framework**: [Django](https://www.djangoproject.com/) (Python)
-   **Frontend**: Django Templates, [Tailwind CSS](https://tailwindcss.com/), JavaScript
-   **Database**: SQLite (Development)
-   **ML/AI Libraries**: 
    -   `input-transformers` (Sentence Embeddings)
    -   `scikit-learn` (Cosine Similarity)
    -   `transformers` (HuggingFace)
    -   `spacy` (NLP)
-   **External APIs**: OpenLibrary API

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.9 or higher
-   Pip (Python Package Manager)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/intensealchemist/Book-Summarizer-and-recommendation.git
    cd Book-Summarizer-and-recommendation
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Apply Migrations**
    ```bash
    python manage.py migrate
    ```

5.  **Run the Server**
    ```bash
    python manage.py runserver
    ```

6.  **Access the App**
    Open your browser and navigate to `http://127.0.0.1:8000/`.

## 📂 Project Structure

```
BookMind/
├── book_app/           # Project Configuration
├── summaries/          # Main Application Logic
│   ├── models.py       # Database & AI Models
│   ├── views.py        # Controllers & API Logic
│   ├── recommendation.py # Recommendation Engine
│   └── templates/      # HTML Templates (Tailwind)
├── media/              # User Uploads (Covers, PDFs)
├── static/             # CSS, JS, Images
├── manage.py           # Django Management Script
└── README.md           # Documentation
```

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Built by <a href="https://github.com/intensealchemist">Atul Sharma</a>
</p>
