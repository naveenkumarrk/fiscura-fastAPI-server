import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
import csv
import requests
from groq import Groq
from together import Together
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import joblib 
from pydantic import BaseModel

from clean_csv import extract_tables, cleanCsv, process_chunk



load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
SAMPLE_DIR = BASE_DIR / "samples"
SAMPLE_CSV = SAMPLE_DIR / "sample_statement.csv"
WORKING_CSV = BASE_DIR / "working_statement.csv"
PKL_MODEL = BASE_DIR / "model.pkl"

UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

#  Groq API
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Together API
together_client = Together()

SAMPLE_STATEMENTS = {
    "hdfc_2023": "sample_statements/hdfc_2023.csv",
    # "hdfc_2022": "sample_statements/hdfc_2022.csv",
    # "sbi_2023": "sample_statements/sbi_2023.csv"    
}

@app.get("/sample-statements")
async def get_sample_statements():
    return [
        {
            "id": key,
            "name": key.replace("_", " ").title(),
            "description": f"Sample {key.replace('_', ' ').title()} Statement"
        }
        for key in SAMPLE_STATEMENTS.keys()
    ]





@app.post("/use-sample")
async def use_sample_statement():
    try:
        if not SAMPLE_CSV.exists():
            raise HTTPException(status_code=404, detail="Sample statement not found")
        process_data_for_analysis(SAMPLE_CSV)
        return {"message": "Sample statement processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_pdf = None
    
    try:
        # Ensure directories exist
        UPLOAD_DIR.mkdir(exist_ok=True)
        WORKING_CSV.parent.mkdir(exist_ok=True)
        
        # Save uploaded PDF
        temp_pdf = UPLOAD_DIR / "temp.pdf"
        with temp_pdf.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Extract tables from PDF
        raw_df = extract_tables(str(temp_pdf))
        
        if raw_df.empty:
            raise HTTPException(status_code=400, detail="No valid tables found in PDF")

        # Process with LLM and save result
        cleaned_df = cleanCsv(str("extracted_tables.csv"), use_together=True)
        cleaned_df.to_csv(str(WORKING_CSV), index=False)
        
        # Process data for analysis
        process_data_for_analysis(WORKING_CSV)
        
        return {
            "message": "File processed successfully",
            "rows_processed": len(cleaned_df)
        }
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files
        if temp_pdf and temp_pdf.exists():
            temp_pdf.unlink()
        if Path("extracted_tables.csv").exists():
            Path("extracted_tables.csv").unlink()

# df = cleanCsv("extracted_tables.csv", use_together=True)
# df = pd.read_csv("cleaned_bank_statement.csv")





@app.get("/summary")
def get_summary():
    # Load Data
    df = pd.read_csv(WORKING_CSV)
    df = df.replace({np.nan: None})
    
    # (ISO format: YYYY-MM-DD)
    df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d', errors="coerce")

    # Drop invalid dates
    df = df.dropna(subset=["Date"])
    if df.empty:
        return {"error": "No valid date entries found in the dataset."}

    # Extract for display
    start_date = df['Date'].iloc[0].strftime("%d %B %Y")  # Example: "01 November 2022"
    end_date = df['Date'].iloc[-1].strftime("%d %B %Y")  # Example: "04 November 2022"

    # Calculate duration
    days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days + 1  # Ensure at least 1 day

    # Convert necessary columns to numeric
    df["Withdrawal"] = pd.to_numeric(df["Withdrawal"], errors="coerce").fillna(0)
    df["Deposited"] = pd.to_numeric(df["Deposited"], errors="coerce").fillna(0)
    df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce").fillna(0)

    total_withdrawal = df["Withdrawal"].sum()
    total_deposit = df["Deposited"].sum()

    # Return Summary
    return {
        "period": {
            "start": start_date,
            "end": end_date,
            "days": days
        },
        "metrics": {
            "totalTransactions": len(df),
            "totalWithdrawal": float(total_withdrawal),
            "totalDeposit": float(total_deposit),
            "closingBalance": float(df["Balance"].iloc[-1]),
            "openingBalance": float(df["Balance"].iloc[0]),
            "avgWithdrawalPerDay": float(total_withdrawal / days) if days else 0,
            "avgWithdrawalPerMonth": float(total_withdrawal / (days / 30)) if days else 0
        }
    }



@app.get("/trends")
async def get_trends():
    try:
        df = pd.read_csv(WORKING_CSV)
        df = df.replace({np.nan: None})
        
        # Parse dates flexibly
        df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True)
        
        # Calculate daily totals
        daily_totals = df.groupby("Date").agg({
            "Withdrawal": "sum",
            "Deposited": "sum",
            "Balance": "last"
        }).fillna(0)

        return {
            "withdrawal": [
                {"date": date.strftime("%d-%b"), "value": float(row["Withdrawal"])}
                for date, row in daily_totals.iterrows()
            ],
            "deposit": [
                {"date": date.strftime("%d-%b"), "value": float(row["Deposited"])}
                for date, row in daily_totals.iterrows()
            ],
            "balance": [
                {"date": date.strftime("%d-%b"), "value": float(row["Balance"])}
                for date, row in daily_totals.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions")
def get_transactions():
    df = pd.read_csv(WORKING_CSV)
    df = df.replace({np.nan: None})  # Replace NaN with None
    return df[['UPI_Name', 'UPI_Description', 'Date_Formated', 
               'Withdrawal', 'Deposited', 'Balance', 
               'Cumulative_Withdrawal', 'Cumulative_Deposited']].to_dict(orient='records')



@app.get("/upi-analysis")
def get_upi_analysis():
    df = pd.read_csv(WORKING_CSV)
    df = df.replace({np.nan: None})  # Replace NaN with None

    # Convert "Date" column to datetime, handle errors
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True, errors='coerce')

    # Drop rows where Date conversion failed
    df = df.dropna(subset=["Date"])

    # Fill NaN with 0 for calculations
    df["Withdrawal"] = pd.to_numeric(df["Withdrawal"], errors='coerce').fillna(0)

    # Ensure "UPI_Name" column is a string
    df["UPI_Name"] = df["UPI_Name"].astype(str)

    # Group by UPI Name and sum withdrawals
    upi_summary = df.groupby('UPI_Name')['Withdrawal'].sum().sort_values(ascending=False)

    # Get highest daily spend
    daily_spend = df.groupby("Date")["Withdrawal"].sum()

    if not daily_spend.empty:
        highest_spend_date = daily_spend.idxmax()
        highest_spend_amount = daily_spend.max()
    else:
        highest_spend_date = None
        highest_spend_amount = 0.0

    # Find highest individual transaction
    # Check if the DataFrame is empty or if all values are zero
    if not df.empty and df["Withdrawal"].max() > 0:
        max_withdrawal_idx = df["Withdrawal"].idxmax()
        highest_transaction = {
            "amount": float(df.loc[max_withdrawal_idx, "Withdrawal"]),
            "date": df.loc[max_withdrawal_idx, "Date"].strftime("%d %B %Y"),
            "description": df.loc[max_withdrawal_idx, "UPI_Description"]
        }
    else:
        highest_transaction = {"amount": 0.0, "date": None, "description": None}

    return {
        "upiWise": [
            {"name": name, "amount": float(amount)}
            for name, amount in upi_summary.items()
            if name.strip() and name != "nan"  # Skip empty names and 'nan' strings
        ],
        "highestTransaction": highest_transaction,
        "highestDailySpend": {
            "date": highest_spend_date.strftime("%d %B %Y") if highest_spend_date else None,
            "amount": float(highest_spend_amount)
        }
    }






model = joblib.load('loan_status_predictor.pkl')
scaler = joblib.load('vector.pkl')
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']




class LoanApproval(BaseModel):
    Gender: float 
    Married: float
    Dependents:float
    Education: float
    Self_Employed: float
    ApplicantIncome:float
    CoapplicantIncome:float
    LoanAmount:float 
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: float



@app.post("/predict")
async def predict_loan_status(application: LoanApproval):
    input_data = pd.DataFrame([application.dict()])
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    result = model.predict(input_data)

    if result[0] == 1:
        return {'Loan Status': "Approved"}
    else:
        return {'Loan Status': "Not Approved"}


# @app.get("/install-ghostscript")
# def install_ghostscript():
#     try:
#         os.system("apt-get update && apt-get install -y ghostscript")
#         return {"message": "✅ Ghostscript installed successfully!"}
#     except Exception as e:
#         return {"error": f"❌ Failed to install Ghostscript: {str(e)}"}


def process_data_for_analysis(csv_path: Path) -> None:
    """Process CSV data and prepare it for analysis"""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True)
    df["Date_Formated"] = df["Date"].dt.strftime("%d-%b-%Y")
    df.to_csv(WORKING_CSV, index=False)


model = joblib.load('credit_card_model.pkl')  # Load your model

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict-fraud")
def predict_fraud(transaction: Transaction):
    # Dynamically exclude 'Time'
    features = [getattr(transaction, f) for f in transaction.__fields__ if f != 'Time']
    data = np.array([features])
    
    prediction = model.predict(data)
    return {"fraud": bool(prediction[0])}


TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY","894f20aae5d74def34790efed3a6d2f3883fc6a7b5716bfcdea47a8c6207ba19")
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"

def generate_together_response(prompt, temperature=0.7, max_tokens=500):
    """Generate a response using the Together API"""
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1", # You can change this to your preferred model
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40
    }
    
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error calling Together API: {str(e)}")
        return f"I'm sorry, I encountered an error processing your request. Please try again later."
    

@app.post("/ask-financial-advisor")
async def ask_financial_advisor(question: str = Form(...), financial_status: Optional[str] = Form(None)):
    """
    Process user questions using the Together API to provide dynamic, personalized financial advice
    
    Parameters:
    - question: User's question about financial planning
    - financial_status: Optional financial status detected from previous analysis
    """
    try:
        # Format prompt based on whether a financial status was provided
        status_context = ""
        financial_categories = {
            "retirement": ("RET", "Retirement Planning"),
            "investment": ("INV", "Investment Strategy"),
            "debt": ("DBT", "Debt Management"),
            "budgeting": ("BDG", "Personal Budgeting"),
            "tax": ("TAX", "Tax Planning"),
            "insurance": ("INS", "Insurance Planning")
        }
        
        if financial_status and financial_status in financial_categories:
            status_name = financial_categories.get(financial_status, ("UNK", "Unknown"))[1]
            status_context = f"The user has previously indicated interest in {status_name} ({financial_status})."
        
        # Create the prompt for the Together API
        prompt = f"""You are an expert financial advisor specializing in personal finance and investment strategies. 
Provide a helpful, concise response to the user's question. Keep your answer short (3-5 sentences maximum), 
factually accurate, and focused on evidence-based financial principles. If appropriate, organize key points in a bullet list format.

{status_context}

User question: {question}

Remember:
- Always recommend consulting a licensed financial professional for personalized advice
- Be clear, concise, and educational
- Provide actionable advice where appropriate
- Focus on generally accepted financial principles and cite sources when possible

Your response:"""

        # Call the Together API
        response_text = generate_together_response(prompt, temperature=0.7, max_tokens=500)
        
        # Extract key points (if any) from the response
        lines = response_text.split('\n')
        main_response = lines[0] if lines else response_text
        additional_info = []
        
        for line in lines[1:]:
            # Look for bullet points or numbered lists
            clean_line = line.strip()
            if clean_line and (clean_line.startswith('-') or clean_line.startswith('•') or 
                              (len(clean_line) > 1 and clean_line[0].isdigit() and clean_line[1] == '.')):
                additional_info.append(clean_line.lstrip('- •0123456789. '))
        
        # If no bullet points were found but we have multiple paragraphs, use those
        if not additional_info and len(lines) > 1:
            main_response = lines[0]
            additional_info = [line.strip() for line in lines[1:] if line.strip()]
        
        # Add references when available
        references = []
        reference_keywords = ["according to", "based on", "as reported by", "source:", "reference:"]
        for info in additional_info:
            for keyword in reference_keywords:
                if keyword in info.lower():
                    references.append(info)
                    break
                    
        if not references:
            references = ["Financial information based on generally accepted principles in personal finance.",
                         "For personalized advice, please consult with a certified financial planner."]
        
        return {
            "response": main_response,
            "additional_info": additional_info[:3],  # Limit to 3 additional points
            "references": references[:2]  # Include up to 2 references
        }
    
    except Exception as e:
        return {
            "response": "I'm sorry, I encountered an error while processing your financial question.",
            "additional_info": [
                "This may be due to a temporary issue with our service.",
                "Please try again later or rephrase your question."
            ],
            "error": str(e)
        }