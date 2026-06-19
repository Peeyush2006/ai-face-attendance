# Cloud Deployment Guide - AI Attendance System

This guide outlines how to deploy your AI Attendance System online so that it can be accessed from any device on the internet.

---

## 🚀 Recommendation: Deploy using Docker on Render (Free Tier)

Since our application relies on **OpenCV** (which requires specific system libraries like `libgl1` and `glib` to process images), deploying it as a **Docker Container** is the most reliable method. This ensures it will build and run identically in the cloud as it does on your local machine.

### Step 1: Push your project to GitHub
1. Create a new repository on [GitHub](https://github.com).
2. Initialize git in your project directory and push the code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of AI Attendance System"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

---

### Step 2: Deploy on Render.com (Web Service)
Render is a cloud hosting provider that offers a generous free tier.

1. Create a free account at [Render](https://render.com).
2. Click the **New +** button in the dashboard and select **Web Service**.
3. Connect your GitHub account and select your project repository.
4. Configure the Web Service:
   * **Name**: `igntu-ai-attendance`
   * **Region**: Select a region close to your users (e.g., Singapore or US Oregon).
   * **Branch**: `main`
   * **Runtime**: **`Docker`** *(Render will automatically detect your `Dockerfile` and build the container!)*
   * **Instance Type**: **`Free`**
5. Click **Deploy Web Service**. Render will build and deploy your container. Once completed, they will give you a public URL (e.g., `https://igntu-ai-attendance.onrender.com`).

---

### Step 3: Persist Registered Faces & Database
> [!IMPORTANT]
> Render's free tier has an **ephemeral disk**. This means when your web service restarts (which happens periodically on the free tier), any registered students' photo files and database logs will be deleted.
> 
> To prevent this, you should set up a **Persistent Disk** on Render:
1. In your Render web service dashboard, click on **Disk** in the left menu.
2. Click **Add Disk**:
   * **Name**: `attendance-data`
   * **Mount Path**: `/app/data` *(This matches the `/app/data` folder inside our Docker container where SQLite database and face crops are saved)*
   * **Size**: `1 GB` (More than enough for thousands of student entries)
3. Click **Save**. Render will restart the server and mount the persistent disk. Now, all face models and database entries are completely safe and persistent!

---

## 🚆 Alternative: Deploy on Railway.app

Railway is an extremely fast and easy-to-use hosting platform.

1. Go to [Railway](https://railway.app) and create an account.
2. Click **New Project** -> **Deploy from GitHub repo** and select your repository.
3. Railway will detect the `Dockerfile` and start the deploy.
4. To add persistence:
   * Click on the service -> **Settings** -> **Volumes**.
   * Add a Volume mounted at `/app/data`.

---

## 🐳 Running locally with Docker

If you want to run the containerized application on your own local server or network:
1. Build the image:
   ```bash
   docker build -t ai-attendance-system .
   ```
2. Run the container with a volume to persist data:
   ```bash
   docker run -d -p 8000:8000 -v attendance_data:/app/data --name ai-attendance ai-attendance-system
   ```
