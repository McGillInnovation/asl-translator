import cv2
#vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
vidcap = cv2.VideoCapture(0)


success,frame = vidcap.read()
count = 0
success = True
while success:
  success,frame = vidcap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Display the resulting frame
  cv2.imshow('frame',gray)
  print 'Read a new frame: ', success
  
  cv2.imwrite("frame%d.jpg" % count, frame)     # save frame as JPEG file
  count += 1
  print(count)

  if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()