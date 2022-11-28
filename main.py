import cv2 

faceCascade = cv2.CascadeClassifier('trainings/haarcascade_frontalface_alt.xml')
facecount = 0

# grab the reference to the webcam
vs = cv2.VideoCapture(0)

# keep looping
while True:
	# grab the current frame
	ret, frame = vs.read()
  
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
		
	faces = faceCascade.detectMultiScale(frame)
	
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
		facecount = facecount + 1
	
	cv2.putText(frame, 'Faces: '+str(facecount), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0 ,0))
	
	# show the frame to our screen
	cv2.imshow("Video", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' or ESC key is pressed, stop the loop
	if key == ord("q") or key == 27:
		break
	
	facecount = 0
	
# close all windows
cv2.destroyAllWindows()
