from flask import Flask, request, render_template, send_from_directory, url_for, flash
import os
from PIL import Image
import cv2
from werkzeug.utils import secure_filename, redirect
import process as process

app = Flask(__name__)
app.secret_key = 'thisisasecretkey'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['UPLOAD_PATH'] = os.path.join(APP_ROOT, 'static/images/')

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def main():     
   return render_template('base.html')

@app.route('/', methods=['POST'])
def predict():
   uploaded_file = request.files['file']
   filename = secure_filename(uploaded_file.filename)
   if filename != '':
      file_ext = os.path.splitext(filename)[1]
      if file_ext not in app.config['UPLOAD_EXTENSIONS']:
         flash("Invalid input.")
         return render_template('base.html')
         # return "Invalid input", 400

      file_path = os.path.join(app.config['UPLOAD_PATH'], 'query.jpg')
      uploaded_file.stream.seek(0)
      uploaded_file.save(file_path)

      image_path = os.path.join(app.config['UPLOAD_PATH'], 'query.jpg')
      result, faces_count = process.predict_gender(image_path)
      cv2.imwrite(os.path.join(app.config['UPLOAD_PATH'], 'result.jpg'), result)
      if faces_count == 0:
         flash("No face detected.")
      return render_template('result.html')
   else:
      flash("Invalid input.")
      return render_template('base.html')

   # return render_template('result.html')

@app.route('/', methods=['POST'])
def tryagain():
   if request.method == 'POST':
        return redirect(url_for('index'))
   return render_template('base.html')

@app.route('/static/images/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename, as_attachment=True)

if __name__ == '__main__':
   app.run(debug=True)
