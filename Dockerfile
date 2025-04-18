FROM python:3.10

RUN apt-get update && apt-get install -y ghostscript && apt-get clean && rm -rf /var/lib/apt/lists/*

# create dir
WORKDIR /app

# .venv
COPY .venv .venv

# activate env
ENV PATH="/app/.venv/bin:$PATH"

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# cpy files
COPY . .

# PORT 
EXPOSE 8000

# command to start FASTAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
