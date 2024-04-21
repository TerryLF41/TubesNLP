from flask import Flask, render_template, request, jsonify
from Model import nlp_model as model
import os
import pdfplumber
from docx import Document

app = Flask(__name__, static_url_path='/static')

received_data = ""  # Variable to store received data

@app.route('/api/data', methods=['POST'])
def receive_data():
    global received_data
    data = request.get_json()
    received_string = data.get("string_data", "")
    received_data = received_string  # Assign the received data to the variable
    response_data = {"message": "Data received successfully"}
    return jsonify(response_data)

@app.route("/", methods=["GET", "POST"])
def form():
    return render_template("main.html") 
    
@app.route("/output", methods=["POST"])
def output():
    if request.method == 'POST':
        text_masuk = request.form.get("paragraf")
        if text_masuk == "":
            file = request.files['upload']  # Get the uploaded file
            file_path = os.path.join(app.root_path, 'uploads', file.filename)  # Define the path to save the file
            file.save(file_path)  # Save the uploaded file
            if file.filename.endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    file_txt = ""
                    for page in pdf.pages:
                        file_txt += page.extract_text()
            elif file.filename.endswith('.docx'):
                doc = Document(file_path)
                file_txt = ""
                for paragraph in doc.paragraphs:
                    file_txt += paragraph.text
            else:
                temp = open(file_path,"r")
                file_txt = temp.read()
                
            abstractive = model.abstractive_summary(file_txt)
            extractive = model.extractive_summary(file_txt)
            return jsonify({
                "output_abstractive": abstractive,
                "output_extractive": extractive,
                "input": file_txt
            })
        else:
            abstractive_summary = model.abstractive_summary(text_masuk)
            extractive_summary = model.extractive_summary(text_masuk)
            return jsonify({
                "output_abstractive": abstractive_summary,
                "output_extractive": extractive_summary,
                "input": text_masuk
            })

if __name__ == '__main__':
    app.run(debug=True)