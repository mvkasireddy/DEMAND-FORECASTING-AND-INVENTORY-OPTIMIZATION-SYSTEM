

# Demand Forecasting and Inventory Optimization System

**An End-to-End Machine Learning and Generative AI Application using Python, XGBoost, Streamlit, RAG, FAISS and Ollama.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![AI](https://img.shields.io/badge/Generative%20AI-RAG-blueviolet)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![Vector Database](https://img.shields.io/badge/Vector%20Database-FAISS-orange)

## Project Overview

The Demand Forecasting and Inventory Optimization System is an end-to-end Machine Learning and Generative AI application designed to predict product demand and support inventory-related business decisions.

The system uses an **XGBoost Regression model** to estimate demand based on business inputs such as product price, discount, inventory level, promotional activity, competitor pricing and product category.

A Streamlit application provides an interactive interface where users can enter business parameters, generate demand predictions, visualize demand comparisons and create downloadable prediction reports.

Additionally, a Retrieval-Augmented Generation (RAG) assistant integrates FAISS, Hugging Face embeddings and a locally running Mistral model through Ollama to provide context-based retail and inventory guidance using business strategy documentation.

## Business Problem

Businesses need to estimate product demand to make better inventory and pricing decisions.

Incorrect demand estimation can contribute to:
- Excess inventory and storage costs.
- Product shortages and missed sales.
- Inefficient discount and promotional decisions.
- Difficulty responding to competitor pricing.

This project addresses these challenges by combining machine learning-based demand prediction with an AI-powered assistant for contextual business guidance.

## Project Objectives

- Analyze and preprocess demand forecasting data.
- Develop an XGBoost regression model for demand prediction.
- Identify relevant business factors affecting demand.
- Build an interactive Streamlit prediction application.
- Compare predicted demand with available inventory and competitor pricing.
- Generate downloadable demand forecasting reports.
- Integrate a RAG-based AI assistant for business-related questions.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | XGBoost, Scikit-learn |
| Model Optimization | RandomizedSearchCV |
| Application Development | Streamlit |
| Generative AI | Mistral via Ollama |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | Hugging Face, all-MiniLM-L6-v2 |
| Model Persistence | Pickle |
| API Integration | Python Requests |

## System Architecture

```text
              DEMAND FORECASTING DATA
                         |
                         v
              DATA ANALYSIS & EDA
                 Pandas / NumPy
                         |
                         v
              DATA PREPROCESSING
             Cleaning & Feature Setup
                         |
                         v
              MACHINE LEARNING MODEL
                  XGBoost Regressor
                         |
                         v
                TRAINED MODEL
                  Pickle (.pkl)
                         |
                         v
               STREAMLIT APPLICATION
                         |
              +----------+----------+
              |                     |
              v                     v
        DEMAND PREDICTION      RAG AI ASSISTANT
              |                     |
              v                     v
       Demand Comparison      Business Strategy PDF
              |                     |
              v                     v
       Report Generation      Text Chunking
              |                     |
              v                     v
       Download / WhatsApp     Hugging Face Embeddings
                                    |
                                    v
                               FAISS VECTOR DB
                                    |
                                    v
                               Context Retrieval
                                    |
                                    v
                              OLLAMA - MISTRAL
                                    |
                                    v
                              AI RESPONSE
```

## 1. Data Analysis and Preprocessing

**Files:**
- `analysis.ipynb`
- `demand_forecasting.csv`
- `preprocessed_demand_forecasting_data.csv`

The initial stage focuses on understanding and preparing the demand forecasting dataset.

Implementation:
- Loaded the dataset using Pandas.
- Examined dataset structure, data types and statistical information.
- Checked missing values and duplicate records.
- Performed data cleaning and preprocessing.
- Prepared relevant features for machine learning.
- Explored data patterns using Matplotlib and Seaborn.

### Feature Engineering

Additional features were prepared to support analysis, including:

- Year
- Month
- Day
- Weekday
- Discounted Price
- Sell Through Rate

The processed dataset is saved for subsequent machine learning development.

## 2. Machine Learning Model Development

**File:** `machine_learning.ipynb`

The system uses **XGBoost Regression** to predict product demand.

### Model Input Features

| Feature | Description |
|---|---|
| Price | Product selling price |
| Discount | Discount percentage |
| Inventory Level | Available inventory |
| Promotion | Promotional activity indicator |
| Competitor Pricing | Competitor product price |
| Category | Product category |

**Target Variable:** Demand

### Model Development Process

1. Selected relevant input features and the demand target.
2. Encoded categorical data using Scikit-learn LabelEncoder.
3. Divided the dataset into training and testing sets.
4. Trained an XGBoost regression model.
5. Applied RandomizedSearchCV for hyperparameter tuning.
6. Evaluated the model using regression evaluation techniques.
7. Saved the trained model and label encoders using Pickle.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Algorithm | XGBoost Regressor |
| Objective | reg:squarederror |
| Train-Test Split | 80:20 |
| Hyperparameter Tuning | RandomizedSearchCV |
| Iterations | 25 |
| Cross Validation | 3 |
| Scoring | Negative Mean Absolute Error |

### Saved Model Artifacts

- `xgb_demand_forecasting_model.pkl` – Trained XGBoost model.
- `label_encoders.pkl` – Saved categorical encoders.

These artifacts allow the Streamlit application to load the trained model without retraining it whenever the application starts.

## 3. Streamlit Application

**File:** `mca.py`

The Streamlit application provides an interactive interface for demand prediction and business analysis.

### User Inputs

Users can provide:

- Product price.
- Discount percentage.
- Inventory level.
- Promotion status.
- Competitor pricing.
- Product category.

### Prediction Process

1. User enters the business parameters.
2. The application prepares the input data.
3. Saved label encoders transform categorical inputs.
4. The trained XGBoost model generates the demand prediction.
5. The predicted demand is displayed in units.

### Demand Visualization

The application displays a bar chart comparing:

- Predicted demand.
- Available inventory.
- Competitor pricing.

### Report Generation

After prediction, users can generate a demand forecasting report containing:

- Product category.
- Predicted demand.
- Product price.
- Competitor price.
- Discount percentage.
- Inventory level.
- Promotion status.
- Demand level and pricing status.
- Basic business recommendations.

Reports can be downloaded as `.txt` files or prepared as WhatsApp messages.

## 4. Generative AI and RAG Integration

**File:** `rag_pdf_pipeline.py`

The project integrates a Retrieval-Augmented Generation (RAG) system to provide additional business context to the demand forecasting application.

### RAG Implementation

**Knowledge Source:** `business_strategy.pdf`

The implementation follows these steps:

1. Loads the business strategy PDF using LangChain's PyPDFLoader.
2. Splits the extracted document text into smaller chunks.
3. Generates vector embeddings using Hugging Face.
4. Stores the document embeddings in FAISS.
5. Retrieves the three most relevant document chunks for a user query.
6. Passes the retrieved context and prediction details to the local Mistral model.
7. Generates a context-based response through Ollama.

### RAG Configuration

| Component | Technology |
|---|---|
| PDF Loading | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Chunk Size | 500 |
| Chunk Overlap | 50 |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Retrieval | Similarity Search (Top 3) |
| LLM | Mistral |
| Local Model Runtime | Ollama |

### AI Assistant Capabilities

The integrated assistant uses:
- Current demand prediction.
- Product price.
- Inventory level.
- Retrieved business strategy document context.

Users can ask business-related questions and receive AI-generated explanations and suggested actions.

The system runs the Mistral model locally through the Ollama API.

## Repository Structure

```text
DEMAND-FORECASTING-AND-INVENTORY-OPTIMIZATION-SYSTEM/
│
├── analysis.ipynb
│
├── machine_learning.ipynb
│
├── demand_forecasting.csv
│
├── preprocessed_demand_forecasting_data.csv
│
├── mca.py
│
├── xgb_demand_forecasting_model.pkl
│
├── label_encoders.pkl
│
├── rag_pdf_pipeline.py
│
├── business_strategy.pdf
│
└── README.md
```

## How to Run the Project

### Prerequisites

- Python 3.9 or later.
- pip package manager.
- Ollama installed.
- Mistral model downloaded through Ollama.

### Step 1: Clone the Repository

```bash
git clone https://github.com/mvkasireddy/DEMAND-FORECASTING-AND-INVENTORY-OPTIMIZATION-SYSTEM.git

cd DEMAND-FORECASTING-AND-INVENTORY-OPTIMIZATION-SYSTEM
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit requests langchain langchain-community langchain-text-splitters sentence-transformers faiss-cpu pypdf
```

### Step 3: Set Up Ollama

Install Ollama and download the Mistral model:

```bash
ollama pull mistral
```

Start the local Ollama service before using the AI assistant.

### Step 4: Run the Application

```bash
streamlit run mca.py
```

### Step 5: Use the Application

1. Enter the product and business parameters.
2. Click Predict Demand.
3. View the predicted demand and comparison chart.
4. Generate and download the demand forecasting report.
5. Ask business-related questions using the AI assistant.

**Note:** Keep the model files and `business_strategy.pdf` in the application directory. The RAG vector database is created from the PDF when the application initializes.

## Key Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Data Cleaning and Preprocessing
- Feature Engineering
- Regression Modeling
- XGBoost Algorithm
- Hyperparameter Optimization
- Categorical Encoding
- Model Serialization and Deployment
- Streamlit Application Development
- Retrieval-Augmented Generation
- Semantic Search and Vector Embeddings
- FAISS Vector Database
- Local LLM Integration
- Business-Oriented Predictive Analytics

## Future Enhancements

Potential improvements include:

- SKU-level demand forecasting.
- Time-series forecasting and seasonal demand modeling.
- Advanced inventory optimization algorithms.
- Model performance monitoring and retraining.
- Cloud deployment and API integration.
- Persistent vector database storage.
- Automated model evaluation and experiment tracking.

These are future development opportunities, not features claimed as currently implemented.

## Project Outcome

Developed an integrated Machine Learning and Generative AI application that predicts product demand using XGBoost and supports business decision-making through an interactive Streamlit dashboard.

Integrated a document-based RAG assistant using FAISS, Hugging Face embeddings and a locally running Mistral model to provide contextual business guidance alongside demand predictions.

## Author

**Venkata Kasi Reddy Mandadi**

GitHub: [mvkasireddy](https://github.com/mvkasireddy)

LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/mandadi-venkata-kasi-67a4643a8/)

---

*Developed as an MCA project focused on Machine Learning, predictive analytics and Generative AI integration.*


