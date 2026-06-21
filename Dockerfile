# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables to optimize Python execution and set timezone
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Kolkata

# Set the working directory inside the container
WORKDIR /app

# Install glib system library (required by OpenCV) and tzdata (for timezone mapping)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency list
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend and frontend folders into the container
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose the application port
EXPOSE 8000

# Change working directory to backend folder to prevent Python import path errors
WORKDIR /app/backend

# Start the FastAPI server using shell format to resolve Railway/Render's dynamic port variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
