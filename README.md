# Python Audio-Video Matcher

## Overview

The Python Audio-Video Matcher project is designed to process audio and video data, extract features, and train a machine learning model. The trained model can then be served via an API for further use.

This README provides a step-by-step guide to set up the project, run the ingestion pipeline, train the model, and serve it via an API.

---

## Prerequisites

Before starting, ensure you have the following installed on your system:

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tools (e.g., `venv` or `virtualenv`)

---

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/vaibhavd2103/python-audio-video-matcher
   cd python-audio-video-matcher
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   - Create a `.env` file in the project root directory.
   - Add the required environment variables. For example:
     ```env
     API_KEY=your_api_key_here
     DATA_PATH=./data
     ```

---

## Running the Project

### Step 1: Run the Ingestion Pipeline

The ingestion pipeline fetches metadata, downloads videos, processes them, and extracts audio and video features.

Run the following command to execute the entire ingestion pipeline:

```bash
python run_project.py
```

This script will sequentially execute the following steps:

1. Fetch metadata
2. Download videos
3. Process videos
4. Extract audio features
5. Extract video features

### Step 2: Train the Model

Once the ingestion pipeline is complete, the next step is to train the model. The training step is included in the `run_project.py` script and will automatically execute after the ingestion steps.

### Step 3: Serve the Model via API

To serve the trained model via an API, you can run the backend server manually or `run_project.py` will automatically run the backend for serving it locally. To run manually:

```bash
uvicorn api.app:app --reload
```

This will start the API server, which can be accessed at `http://127.0.0.1:8000`.

### Step 4: Start the Frontend Server

To start the frontend server, run the following command:

```bash
streamlit run frontend/app.py
```

---

## Project Structure

```
python-audio-video-matcher/
├── .env                # Environment variables
├── .gitignore          # Git ignore file
├── run_project.py      # Main script to run ingestion and training
├── requirements.txt    # Python dependencies
├── app/                # API server code
├── ingestion/          # Data ingestion scripts
├── training/           # Model training scripts
└── data/               # Directory for storing data
```

---

## Notes

- Ensure all dependencies are installed before running the project.
- The `.env` file must be properly configured with the required variables.
- For production deployment, consider using a process manager like `gunicorn` or `supervisord` to serve the API.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For any questions or issues, please contact **Tanya Thakur** at [dtanyathakur@gmail.com](mailto:dtanyathakur@gmail.com) or **Tanisha Thakur** at [tanisha.vanneri@example.com](mailto:tanisha.vanneri@example.com).
