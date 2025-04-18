import pandas as pd 
import camelot
import os 

from groq import Groq
from together import Together


groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Initialize Together client
together_client = Together()

def extract_tables(file_path):
    """Extract tables from PDF using Camelot and save raw data to CSV"""
    all_data = []
    tables = camelot.read_pdf(file_path, pages='all')
    print(f"Total tables found: {tables.n}")
    for table in tables:
        df = table.df
        all_data.append(df)
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("extracted_tables.csv", index=False)
    return final_df

def cleanCsv(file_path, chunk_index=0, total_chunks=1, use_together=False):
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = file.read()
    
    system_prompt = """You are a financial data assistant. Your task is to clean, extract, and structure raw bank statement data into a well-organized CSV format.
    Return ONLY the CSV content with no additional explanations or markdown.
    The CSV must have the following columns:
    Index,Narration,Date,Date_Formated,Withdrawal,Deposited,Balance,UPI_Name,UPI_Bank,UPI_Description,Cumulative_Withdrawal,Cumulative_Deposited"""
    
    user_prompt = f"""Format the following bank statement data into a structured CSV.
    This is chunk {chunk_index+1} of {total_chunks}.
    
    ### **Raw Data to Process**:
    {raw_data}
    
    ### **Guidelines**:
    - Convert 'Date' to '%d/%m/%y' format and create a new column 'Date_Formated' as '%d-%b-%Y'.
    - Format 'Withdrawal' and 'Deposited' to one decimal place.
    - Convert 'Balance' to float.
    - Extract 'UPI_Name' and 'UPI_Bank' from 'Narration'.
    - Extract 'UPI_Description' using a function.
    - Calculate 'Cumulative_Withdrawal' and 'Cumulative_Deposited'.
    - Do not include any markdown or explanations, just return the CSV content with headers.
    - If this is NOT the first chunk, do not include the header row.
    """
    
    if use_together:
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000
        )
    else:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2000
        )
    
    cleaned_csv = response.choices[0].message.content.strip()
    output_file = "cleaned_bank_statement.csv"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(cleaned_csv)
    
    print(f"Cleaned CSV saved to {output_file}")
    return pd.read_csv(output_file)
