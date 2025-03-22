# LLM-Powered-Booking-Analytics-QA-System

# README

## Project Overview
This project integrates a Large Language Model (LLM) with analytics and an API to process and analyze user queries. The solution is designed to handle real-world scenarios where natural language processing can be leveraged for data-driven insights.

## Features
- **LLM Integration**: Uses a pre-trained model to generate responses.
- **Analytics**: Collects and visualizes query data for insights.
- **API**: RESTful API for interaction with the LLM.
- **Test Queries**: Includes sample test cases and expected outputs.

## Setup Instructions
### Prerequisites
- Python 3.13
- Required libraries: Install using `requirements.txt`
  ```bash
  pip install -r requirements.txt
  ```

### Running the Application
1. Clone the repository:
   ```bash
   git clone <https://github.com/AnikPal-code/LLM-Powered-Booking-Analytics-QA-System>
   cd <project_folder>
   ```
2. Start the API server:
   ```bash
   python app.py
   ```
3. Run analytics module:
   ```bash
   python analytics.py
   ```

### Sample API Usage
Send a query to the LLM using:
```bash
curl -X POST "http://127.0.0.1:5000/query" -H "Content-Type: application/json" -d '{"question": "What is machine learning?"}'
```

---

# Short Report: Implementation & Challenges

## Implementation Choices
### 1. **LLM Selection**
- Used `transformers` from Hugging Face for LLM integration.
- Fine-tuned model for domain-specific responses.

### 2. **API Design**
- Built with Flask to allow easy interaction.
- Supports JSON-based queries.

### 3. **Analytics Integration**
- Logs user queries and response times.
- Uses `matplotlib` and `pandas` for visualization.

## Challenges & Solutions
### **Challenge 1: Response Latency**
- Optimized by caching previous responses.

### **Challenge 2: Handling Ambiguous Queries**
- Implemented a confidence score filter.

### **Challenge 3: Deployment**
- Dockerized for easy deployment.
- Used `nginx` as a reverse proxy.

---

# Codebase Structure
```
/llm_project
│── app.py               # API server
│── analytics.py         # Query analytics module
│── model.py             # LLM integration
│── test_queries.json    # Sample test cases
│── requirements.txt     # Dependencies
│── README.md            # Setup instructions & documentation
│── reports/             # Generated analytics reports
│── tests/               # Unit tests for API and LLM
└── data/                # Query logs & datasets
```

---

## Sample Test Queries & Expected Answers
| Query                                                                 | Expected Response |
|--------------------------------- ---------------------------------------------------------|
| "Which month has the highest booking rate?"                           | "August"          |
| "What's the average length of stay in city hotels vs. resort hotels?" | "3 night"         |
| "Is there a correlation between lead time and cancellation rate?"     | "Yes"             |      


This documentation ensures a smooth setup and usage experience. 🚀

