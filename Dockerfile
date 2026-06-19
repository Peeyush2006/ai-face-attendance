# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables to optimize Python execution
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the dependency list
COPY requirements.txt .

# Install Python packages (uses headless OpenCV to avoid building native X11/libGL dependencies)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend and frontend folders into the container
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose the application port
EXPOSE 8000

# Start the FastAPI server using shell format to resolve Railway/Render's dynamic port variable
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
