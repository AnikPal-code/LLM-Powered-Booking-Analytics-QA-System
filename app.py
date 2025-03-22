from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from time import time
from io import StringIO
from typing import Dict, Any

# Load model and tokenizer
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Load FAISS index
FAISS_INDEX_PATH = "faiss_bookings.index"
try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"✅ FAISS index loaded with {index.ntotal} vectors.")
except Exception as e:
    print(f"❌ Error loading FAISS index: {str(e)}")
    index = None

# Load cleaned dataset
DATASET_PATH = "hotel_bookings_cleaned.csv"
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Cleaned dataset loaded with {len(df)} records.")
except Exception as e:
    print(f"❌ Error loading dataset: {str(e)}")
    df = None

# Feature scaling for FAISS retrieval
NUMERIC_FEATURES = ['lead_time', 'adr', 'stays_in_week_nights', 'stays_in_weekend_nights']
scaler = StandardScaler()
if df is not None:
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)
    vectors = scaler.fit_transform(df[NUMERIC_FEATURES].values)
else:
    vectors = None

# Initialize FastAPI
app = FastAPI()

# Request models
class AskRequest(BaseModel):
    question: str

class AnalyticsRequest(BaseModel):
    data: str

# ---------- HELPER FUNCTIONS ----------

def retrieve_similar_bookings(query_vector, top_k=5):
    if index is None or df is None:
        return "No FAISS index or dataset loaded."
    
    if query_vector.shape[1] != index.d:
        return "Query vector dimension mismatch. Cannot search in FAISS index."

    distances, indices = index.search(query_vector, top_k)
    valid_indices = [idx for idx in indices[0] if 0 <= idx < len(df)]
    
    if not valid_indices:
        return "No matching bookings found in FAISS index. Try adjusting search criteria."

    return df.iloc[valid_indices]

def analyze_csv_data(csv_data: str) -> Dict[str, Any]:
    try:
        df = pd.read_csv(StringIO(csv_data))

        if df.empty:
            raise ValueError("Parsed CSV data contains no records.")

        required_columns = {"arrival_date_year", "arrival_date_month", "hotel", "is_canceled", "reservation_status_date"}
        if not required_columns.issubset(df.columns):
            raise ValueError("Missing required booking columns in data.")

        df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors='coerce')
        df = df[df["is_canceled"] == 0]

        if df.empty:
            return {"report": "No valid (non-canceled) bookings found in the dataset."}

        valid_bookings = len(df)
        most_common_hotel = df["hotel"].mode()[0] if not df["hotel"].empty else "Unknown"
        monthly_counts = df["arrival_date_month"].value_counts()
        top_months = monthly_counts.head(3).to_dict()
        top_months_str = ", ".join(f"{month} ({count} bookings)" for month, count in top_months.items())

        report = f"""
        📊 **Hotel Booking Analysis**
        - **Total Valid Bookings:** {valid_bookings}
        - **Most Booked Hotel Type:** {most_common_hotel}
        - **Top 3 Booking Months:** {top_months_str}
        """

        return {"report": report.strip()}

    except Exception as e:
        raise ValueError(f"Analysis failed: {str(e)}")

# ---------- API ENDPOINTS ----------

@app.post("/ask")
async def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time()

    if df is None or index is None:
        return {"answer": "Dataset not loaded. Please ensure FAISS and dataset are available.", "response_time": 0.1}

    query_vector = np.random.rand(1, index.d).astype("float32")

    if query_vector.shape[1] != index.d:
        return {"answer": "Query vector dimension mismatch. Cannot search in FAISS index.", "response_time": 0.1}

    relevant_data = retrieve_similar_bookings(query_vector)

    if isinstance(relevant_data, str):
        return {"answer": relevant_data, "response_time": 0.1}

    try:
        context = relevant_data.to_string(index=False)
        prompt = f"Using the following hotel booking data:\n{context}\n\nAnswer the question: {request.question}\n\n"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        output = model.generate(**inputs, max_length=150, temperature=0.7, do_sample=True, top_k=50)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        response_time = time() - start_time
        return {"answer": decoded_output, "response_time": round(response_time, 3)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/analytics")
async def analytics(request: AnalyticsRequest):
    if not request.data.strip():
        raise HTTPException(status_code=400, detail="CSV data cannot be empty.")
    
    try:
        analysis = analyze_csv_data(request.data)
        return analysis
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")
