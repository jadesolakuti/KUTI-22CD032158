You are an AI engineer that uses the Python language for your projects.
You are to build a machine learning model that can detect the emotion of a person’s face (e.g. a picture of a person frowning will return:

“You are frowning. Why are you sad?”

This project will include a website (no external CSS) where a student fills in their information and uploads a picture of themselves.
Upon submission, the website should:

Detect and return the person’s emotion from the uploaded picture.

Save the user’s inputted information and their image to a .db file.

📁 Project Structure
FACE_DETECTION/
│
├── app.py
├── model_training.py
├── requirements.txt
├── database.db              # stores user data and image info
├── face_emotionModel.h5     # trained model
├── link_web_app.txt          # contains the Render deployment link
│
└── templates/
     └── index.html           # front-end interface (HTML only)
🧩 Project Execution Plan
We’ll take this project step-by-step.git 
At each step, you will:

Explain what’s happening in simple terms (assuming I'm a novice).

Provide complete code for each file when necessary. It's advised you view project as a whole and code's provided should encompass all requirements specified.

Ask me whether to proceed to the next step before moving on.

Use the GIGO mentality: everything you tell me to do, I’ll execute as-is.

The workflow will be:

Dataset selection and download:
You’ll use a dataset I (the AI engineer) choose — most likely FER2013, a popular open dataset for facial emotion recognition.
Guide me on how to download and prepare it for training.

Project setup:
Creating the project folder structure (FACE_DETECTION) and explaining where each file goes.

Model training:
Writing model_training.py to build and train a CNN model using TensorFlow/Keras.
The model will be saved as face_emotionModel.h5.
Initially, the model just needs to work — accuracy improvement will come later if desired.

Flask web app:
Writing app.py to serve the HTML form, accept image uploads, perform predictions, and store the data in database.db.

Database setup:
Creating an SQLite database (database.db) and defining its schema.

Front-end:
Writing templates/index.html with a simple HTML form (no external CSS).

Dependencies:
Creating requirements.txt listing all necessary Python packages.

Testing locally:
Running the app locally on my computer and testing it with images.

Hosting:
Deploying the app using Render (since it supports live Python backends).
GitHub Pages will be used only to store the code, not for hosting the live web app.
The deployed app link will be saved in link_web_app.txt.

Model improvement (optional):
After the full project is working, we can explore improving model accuracy, adding face detection with OpenCV, or optimizing model performance.