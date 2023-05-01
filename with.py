# import the opencv library
import cv2
import numpy as np
import tensorflow as tf
model=tf.ceras.models.load_model("keras_model.h5")  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    img=cv2.resize(frame,(224,224))
    test_img=np.array(img,dtype=np.float32)
    test_img=np.expand_dim(test_img,axis=0)
    normalized_img=test_img/255.0
    prediction=model.predict(normalized_img)
    print("prediction: ",prediction)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()