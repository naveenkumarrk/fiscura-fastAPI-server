import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
import csv
from groq import Groq
from together import Together
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import joblib 
from pydantic import BaseModel

from clean_csv import extract_tables, cleanCsv, process_chunk, process_data_for_analysis



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
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = joblib.load('vector.pkl')


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
