import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class HeartDiseaseUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Prediction System")
        self.root.geometry("1400x850")
        self.root.config(bg='#32d1a7')

        # Internal Data State
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.trained_model = None
        self.results_dict = {}

        self.setup_components()

    def setup_components(self):
        # --- Header ---
        header_font = ('times', 18, 'bold')
        title = tk.Label(self.root, text='Heart Disease Prediction using QMBC Algorithms', 
                         font=header_font, bg='Dark Blue', fg='white', height=2)
        title.pack(fill=tk.X, pady=(0, 20))

        # --- Text Display Area ---
        self.text_font = ('times', 12)
        self.display = tk.Text(self.root, height=18, width=150, font=self.text_font)
        self.display.pack(pady=10)
        
        # --- Button Panels ---
        btn_frame = tk.Frame(self.root, bg='#32d1a7')
        btn_frame.pack(pady=20)

        # Style Configuration
        btn_style = {"font": ('times', 11, 'bold'), "width": 18, "pady": 5}

        # Row 1: Data Management
        self.upload_btn = tk.Button(btn_frame, text="1. Upload Dataset", bg="sky blue", 
                                    command=self.upload_data, **btn_style)
        self.upload_btn.grid(row=0, column=0, padx=10, pady=5)

        self.split_btn = tk.Button(btn_frame, text="2. Split Dataset", bg="light green", 
                                   command=self.split_data, **btn_style)
        self.split_btn.grid(row=0, column=1, padx=10, pady=5)

        self.path_lbl = tk.Label(btn_frame, text="No file selected", bg="#32d1a7", font=('times', 10, 'italic'))
        self.path_lbl.grid(row=0, column=2, columnspan=2)

        # Row 2: Basic Classifiers
        tk.Button(btn_frame, text="Decision Tree", bg="pink", command=self.run_dt, **btn_style).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="Random Forest", bg="lightblue", command=self.run_rf, **btn_style).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="KNN", bg="lightcoral", command=self.run_knn, **btn_style).grid(row=1, column=2, padx=5, pady=5)
        tk.Button(btn_frame, text="Naive Bayes", bg="lightyellow", command=self.run_nb, **btn_style).grid(row=1, column=3, padx=5, pady=5)

        # Row 3: Advanced & Ensembles
        tk.Button(btn_frame, text="SVC", bg="#90ee90", command=self.run_svc, **btn_style).grid(row=2, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="MLP", bg="#ffb6c1", command=self.run_mlp, **btn_style).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="Voting Classifier", bg="#d3d3d3", command=self.run_voting, **btn_style).grid(row=2, column=2, padx=5, pady=5)
        tk.Button(btn_frame, text="QMBC (Final)", bg="gold", command=self.run_qmbc, **btn_style).grid(row=2, column=3, padx=5, pady=5)

        # Row 4: Analysis
        tk.Button(btn_frame, text="View Graph", bg="white", command=self.show_plot, **btn_style).grid(row=3, column=1, padx=5, pady=20)
        tk.Button(btn_frame, text="PREDICT CSV", bg="orange", command=self.batch_predict, **btn_style).grid(row=3, column=2, padx=5, pady=20)

    # --- Core Logic Methods ---

    def log(self, message):
        self.display.insert(tk.END, f"{message}\n")
        self.display.see(tk.END)

    def upload_data(self):
        f_path = filedialog.askopenfilename(initialdir="dataset")
        if f_path:
            self.path_lbl.config(text=f_path)
            self.df = pd.read_csv(f_path)
            self.df.replace('0', np.nan, inplace=True)
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
            self.display.delete('1.0', tk.END)
            self.log(f"Dataset Loaded Successfully! Rows: {len(self.df)}")

    def split_data(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please upload a dataset first.")
            return
        
        le = LabelEncoder()
        process_df = self.df.copy()
        for col in process_df.columns:
            if process_df[col].dtype == 'object':
                process_df[col] = le.fit_transform(process_df[col])

        X = process_df.drop(["target"], axis=1).values
        y = process_df["target"].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.log(f"Split Complete: {len(self.X_train)} training samples, {len(self.X_test)} test samples.")

    def execute_algo(self, model, name):
        if self.X_train is None:
            messagebox.showerror("Error", "Dataset not split!")
            return
        
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        
        metrics = [
            accuracy_score(self.y_test, preds),
            recall_score(self.y_test, preds, zero_division=0),
            precision_score(self.y_test, preds, zero_division=0),
            f1_score(self.y_test, preds, zero_division=0)
        ]
        
        self.results_dict[name] = metrics
        self.log(f"--- {name} Results ---")
        self.log(f"Accuracy: {metrics[0]*100:.2f}% | F1: {metrics[3]*100:.2f}%")
        return model

    # Algo Wrappers
    def run_dt(self): self.execute_algo(DecisionTreeClassifier(), "DT")
    def run_rf(self): self.execute_algo(RandomForestClassifier(), "RF")
    def run_knn(self): self.execute_algo(KNeighborsClassifier(), "KNN")
    def run_nb(self): self.execute_algo(GaussianNB(), "NB")
    def run_svc(self): self.execute_algo(SVC(), "SVC")
    def run_mlp(self): self.execute_algo(MLPClassifier(max_iter=500), "MLP")
    def run_voting(self):
        v_clf = VotingClassifier(estimators=[('dt', DecisionTreeClassifier()), ('nb', GaussianNB())], voting='hard')
        self.execute_algo(v_clf, "Voting")
    
    def run_qmbc(self):
        self.trained_model = self.execute_algo(GaussianNB(), "QMBC")

    def show_plot(self):
        if not self.results_dict: return
        names = list(self.results_dict.keys())
        accs = [m[0]*100 for m in self.results_dict.values()]
        plt.bar(names, accs, color='teal')
        plt.title("Algorithm Comparison (Accuracy)")
        plt.show()

    def batch_predict(self):
        if not self.trained_model:
            messagebox.showerror("Error", "Train QMBC first!")
            return
            
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file:
            data = pd.read_csv(file)
            # Pre-processing here (align columns with training data)
            preds = self.trained_model.predict(data)
            
            self.display.delete('1.0', tk.END)
            self.log(f"{'Row':<6} | {'Age':<5} | {'Sex':<5} | {'Result'}")
            self.log("-" * 50)
            for i, p in enumerate(preds):
                res = "HEART DISEASE" if p == 1 else "SAFE"
                self.log(f"{i+1:<6} | {data.iloc[i]['age']:<5} | {data.iloc[i]['sex']:<5} | {res}")

if __name__ == "__main__":
    app_root = tk.Tk()
    app = HeartDiseaseUI(app_root)
    app_root.mainloop()