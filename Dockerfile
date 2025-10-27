# Use official Python as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements to install first (for caching)
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of application code
COPY . .

# Ensure needed writable folders exist (for volumes too)
RUN mkdir -p uploads translations logs static && \
    chmod -R a+rw uploads translations logs static

# Expose port Flask will use
EXPOSE 5001

# Set default command (use gunicorn for production, python for dev)
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "translator:app"]
# Or for development, use:
# CMD ["python", "translator.py"]
