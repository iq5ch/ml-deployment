import cv2
import numpy as np
from keras.models import load_model

def predict_gender(path):
  model = load_model('saved/model_weights_v7.h5')
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  labels = ['female', 'male']
  faces_count = 0
  img_upload = cv2.imread(path)
  gray = cv2.cvtColor(img_upload, cv2.COLOR_BGR2GRAY)
  img_height = img_upload.shape[0]
  faces = face_cascade.detectMultiScale(gray, 1.31, 4)
  for (x, y, w, h) in faces:
    faces_count = faces_count + 1
    img = gray[y-10:y+h+10, x-10:x+w+10]
    img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, model.layers[0].output_shape[1:3])
    img = img.astype('float32')/255.0 
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    index_label = int(pred.argmax(axis=1))
    label_pred = labels[index_label]
    pred_prob = pred[0][index_label]
    col = (0, 0, 0)
    result = str(round(pred_prob*100, 2))+"% "+str(label_pred)
    cv2.rectangle(img_upload, (x-10, y-10), (x+w+10, y+h+10), col, 3)
    (tw, th), _ = cv2.getTextSize(
      result, cv2.FONT_HERSHEY_SIMPLEX, img_height*0.00125, 3)
    cv2.rectangle(img_upload, (x-10, y+h+10), (x+tw, y+h+15+th), col, -1)
    cv2.putText(
      img=img_upload, 
      text=result, 
      org=(x-5, y+h+10+th), 
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
      fontScale=img_height*0.00125, 
      color=(255, 255, 255), 
      thickness=2)

  # new_image = cv2.cvtColor(img_upload, cv2.COLOR_BGR2RGB)
  new_image = img_upload
  return new_image, faces_count

def img_blur(img):
  img = cv2.imread(img)
  return cv2.blur(img, (15, 15))