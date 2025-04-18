import os
import math
import camelot
import pandas as pd
from groq import Groq
from together import Together
from dotenv import load_dotenv
from pathlib import Path



load_dotenv()

BASE_DIR = Path(__file__).parent
WORKING_CSV = BASE_DIR / "working_statement.csv"

#  Groq API
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Together API
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



def process_chunk(chunk_data, chunk_index, total_chunks, starting_index=0, use_together=False):
    """Processes a single chunk of data using LLM"""
    
    system_prompt = """You are a financial data assistant. Your task is to clean, extract, and structure raw bank statement data into a well-organized CSV format.
    Return ONLY the CSV content with no additional explanations or markdown.
    The CSV must have the following columns:
    Index,Narration,Date,Date_Formated,Withdrawal,Deposited,Balance,UPI_Name,UPI_Bank,UPI_Description,Cumulative_Withdrawal,Cumulative_Deposited"""
    
    user_prompt = f"""Format the following bank statement data into a structured CSV.
    This is chunk {chunk_index+1} of {total_chunks}.
    Start your Index at {starting_index} to ensure continuity between chunks.
    
    ### **Raw Data to Process**:
    {chunk_data}
    
    ### **Guidelines**:
    - Convert 'Date' to '%d/%m/%y' format and create a new column 'Date_Formated' as '%d-%b-%Y'.
    - Format 'Withdrawal' and 'Deposited' to one decimal place.
    - Convert 'Balance' to float.
    - Extract 'UPI_Name' and 'UPI_Bank' from 'Narration'.
    - Extract 'UPI_Description' using a function.
    - Calculate 'Cumulative_Withdrawal' and 'Cumulative_Deposited'.
    - Do not include any markdown or explanations, just return the CSV content with headers.
    - If this is NOT the first chunk, do not include the header row.
    - Ensure each row has a numeric index and all required fields.
    - Do not include rows that don't contain valid transaction data.
    """
    
    if use_together:
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=2000
        )
    else:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=2000
        )
    
    return response.choices[0].message.content.strip()



def cleanCsv(file_path, chunk_size=200, use_together=False):
    """Reads the extracted CSV in chunks, processes each chunk, and merges results"""
    
    # Read CSV
    raw_df = pd.read_csv(file_path)
    
    # Convert df to str
    raw_data_str = raw_df.to_csv(index=False)
    
    # Split into chunks
    lines = raw_data_str.split("\n")
    total_chunks = math.ceil(len(lines) / chunk_size)
    
    all_chunks_combined = []
    header = None
    
    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_data = "\n".join(lines[start:end])
        
        print(f"Processing chunk {i+1}/{total_chunks}...")
        # current count of rows as start
        starting_index = len(all_chunks_combined)
        cleaned_chunk = process_chunk(chunk_data, i, total_chunks, starting_index, use_together)
        
        # Process chunk into rows
        chunk_rows = cleaned_chunk.strip().split('\n')
        
        # Extract header
        if i == 0 and len(chunk_rows) > 0:
            header = chunk_rows[0]
            # Add data rows from first chunk
            if len(chunk_rows) > 1:
                all_chunks_combined.extend(chunk_rows[1:])
        else:
            # For subsequent chunks, skip header if present
            if chunk_rows and chunk_rows[0].startswith("Index,Narration"):
                if len(chunk_rows) > 1:
                    all_chunks_combined.extend(chunk_rows[1:])
            else:
                all_chunks_combined.extend(chunk_rows)
    
    # Make sure
    if not header:
        header = "Index,Narration,Date,Date_Formated,Withdrawal,Deposited,Balance,UPI_Name,UPI_Bank,UPI_Description,Cumulative_Withdrawal,Cumulative_Deposited"
    
    # Combine all chunks with header
    final_csv_content = header + '\n' + '\n'.join(all_chunks_combined)
    
    # Write the raw combined data first
    raw_output_file = "raw_cleaned_statement.csv"
    with open(raw_output_file, "w", encoding="utf-8") as file:
        file.write(final_csv_content)
    
    print(f"Raw combined CSV saved to {raw_output_file}")
    
  
    try:
        # Read the raw combined data
        # Updated parameter from error_bad_lines to on_bad_lines
        df = pd.read_csv(raw_output_file, on_bad_lines='warn')
        
        # Basic cleaning
        # Fill missing values appropriately
        df = df.fillna({
            'Narration': '',
            'UPI_Name': '',
            'UPI_Bank': '',
            'UPI_Description': ''
        })
        
        # Ensure date fields are properly formatted
        # First, check if Date column exists
        if 'Date' in df.columns:
            # Try to convert dates, keeping original if fails
            try:
                # Handle date conversion errors gracefully
                df['Date_Temp'] = pd.to_datetime(df['Date'], 
                                                format='mixed',
                                                dayfirst=True, 
                                                errors='coerce')
                # Keep only rows with valid dates
                df = df.dropna(subset=['Date_Temp'])
                # Format the dates correctly
                df['Date'] = df['Date_Temp'].dt.strftime('%d/%m/%y')
                df['Date_Formated'] = df['Date_Temp'].dt.strftime('%d-%b-%Y')
                # Remove temporary column
                df = df.drop('Date_Temp', axis=1)
            except Exception as e:
                print(f"Date conversion warning: {e}")
        
        # Convert numeric columns
        numeric_cols = ['Withdrawal', 'Deposited', 'Balance', 
                        'Cumulative_Withdrawal', 'Cumulative_Deposited']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove invalid rows
        df = df.dropna(subset=['Date'], how='all')
        
        # Ensure no duplicate index values
        df = df.reset_index(drop=True)
        df['Index'] = df.index
        
        # Save the clean data
        output_file = "cleaned_bank_statement.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Successfully processed and validated {len(df)} rows of data")
        return df
        
    except Exception as e:
        print(f"Error processing combined data: {e}")
        # Emergency fallback - try to recover data manually
        try:
            with open(raw_output_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                raise ValueError("No data found in combined CSV")
            
            header_line = lines[0].strip()
            data_lines = [line for line in lines[1:] if line.strip() and ',' in line]
            
            if not data_lines:
                raise ValueError("No valid data rows found")
            
            # Write clean data
            output_file = "cleaned_bank_statement.csv"
            with open(output_file, 'w') as f:
                f.write(header_line + '\n')
                f.writelines(data_lines)
            
            print(f"Recovered {len(data_lines)} rows using manual recovery")
            
            # Try to read the manually fixed file
            # Updated parameter from error_bad_lines to on_bad_lines
            df = pd.read_csv(output_file, on_bad_lines='warn')
            return df
            
        except Exception as recovery_error:
            print(f"Recovery attempt failed: {recovery_error}")
            # Create minimal valid DataFrame to prevent further errors
            columns = header.split(',')
            df = pd.DataFrame(columns=columns)
            df.to_csv("cleaned_bank_statement.csv", index=False)
            return df
 



