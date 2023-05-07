from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import clip
import torch
import pandas as pd


# Logic

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#Encode text to CLIP vector
def encode_text(search_query):
  with torch.no_grad():
    # Encode and normalize the search query using CLIP
    text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

  # Retrieve the feature vector
  return text_encoded

def encode_image(image_path):
  from PIL import Image
  image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
  image_features = model.encode_image(image)
  image_features /= image_features.norm(dim=-1, keepdim=True)
  return image_features

def get_similarity(search_query, image_path):
    photo_features = encode_image(image_path)
    text_features = encode_text(search_query)
    return (photo_features @ text_features.T).squeeze(1).item()

df = pd.read_csv('Laptions.csv')
vector_db = torch.load('vector_db.pt')
df['Vec'] = [i for i in vector_db]
#df['Vec'] = df['CAPTION'].apply(lambda x: encode_text(x))

def get_single_vector(image_path,text = ''):
    features = encode_image(image_path)
    if(text != ''):
        features += encode_text(text)
    return features

def get_most_similar_caption_from_text_image(image_path,df,text = ''):
    features = get_single_vector(image_path,text)
    df['Similarity'] = df['Vec'].apply(lambda x: (features @ x.T).squeeze(1).item())
    return df.sort_values(by=['Similarity'],ascending=False)

#a = df[df['TOPIC'] != 'Nature: Post pictures of beautiful landscapes, animals, and plants.']


 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    data="my name is ..."
    if 'file' not in request.files:
        print('1')
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    text = request.form['text']
    print(text)
    if file.filename == '':
        print('2')
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        data = get_most_similar_caption_from_text_image(os.path.join(app.config['UPLOAD_FOLDER'], filename),df,text)['CAPTION'].iloc[0:5].to_json(orient='records') 
        print(data)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename,data=data)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(debug=True)