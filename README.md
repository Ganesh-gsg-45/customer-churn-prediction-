# Customer Churn Prediction Web App 🚀

Telco customer churn prediction (80.6% RF accuracy). Flask + HTML/CSS + Docker.

## 📁 Structure
```
.
├── app
├── train_model.py         # Model training & pickle generation
├── requirements.txt       # Dependencies
├── models/                # Trained models
│   ├── churn_prediction_model.pkl
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
├── templates/             # HTML templates
│   ├── index.html
│   └── result.html
├── static/                # CSS
│   └── style.css
├── TODO.md               # Progress tracker
└── note.ipynb            # EDA notebook
├── README.md             # This file!
```

##  Setup & Run
```bash
pip install -r requirements.txt
python train_model.py    # Train & save model (~1min)
python app.py            # Start server
```
**Open:** http://127.0.0.1:5000/

##  Features
- **Input form:** Key Telco features (gender, tenure, charges, contract, etc.)
- **Prediction:** Churn Yes/No + probability %
- **Responsive UI:** Modern gradient design
- **Model:** Tuned RandomForest + ColumnTransformer (OHE + scaling)

##  Model Details
- **Dataset:** WA_Fn-UseC_-Telco-Customer-Churn.csv (../datasets/)
- **Preprocessing:** OneHotEncoder (cat) + StandardScaler (num)
- **Accuracy:** 80.6% on test set
- **Features:** 21 total (10 key in form, others default 'No')


