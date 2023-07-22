from flask import Flask,render_template,request
from flask_wtf import FlaskForm
from wtforms import FileField,SubmitField
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename



upload_folder=r'C:\Users\Ryan\Documents\learnpython\main\flaskweb\static\upload\\'
model=load_model(r'C:\Users\Ryan\Documents\learnpython\main\flaskweb\model.h5')


app = Flask(__name__)
app.config['SECRET_KEY']='supersecretkey'

    
def predict_label(pathh):
     labels=['Trimeresurus albolabris','Naja Sputatrix','Reticulatus Python']
     img = image.load_img(pathh, target_size=(256,256))
     img_array = image.img_to_array(img)
     img_array = np.expand_dims(img_array, axis=0)
     img_array /= 255.0  # Normalize the image
     prediction = model.predict(img_array)
     argmaks=np.argmax(prediction)
     maxx=np.max(prediction)
     percentage=str(int(maxx*100))
     label=labels[argmaks]
    
     
     return label,percentage,int(maxx*100)
     
     
     


@app.route("/",methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route("/predict",methods=['GET','POST'])
def hello_world():
    if  request.method== 'POST':
        img=request.files['my_image']
        
        img_path=upload_folder+img.filename
        
        img.save(img_path)
        filename=f'upload/{img.filename}'
        
        label,percentage,maxx=predict_label(img_path)
        
       
        
         
        
    return render_template('home.html',label=label,percentage=percentage,maxx=maxx,img_path=filename)

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)