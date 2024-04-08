# Import necessary libraries
import cv2
import numpy as np
import speech_recognition as sr

# Function to recognize voice commands
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice commands...")
        audio = r.listen(source)
        try:
            command = r.recognize_google(audio)
            print("You said:", command)
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry, I could not understand the command.")
            return ""
        except sr.RequestError:
            print("Could not request results. Check your internet connection.")
            return ""

# Function for object recognition and path planning
def navigate_with_vision():
    # Load pre-trained object detection model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Capture video from camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        height, width, channels = frame.shape
        
        # Object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Process detected objects
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected, perform path planning or obstacle avoidance
                    
                    # Example: Print class name of detected object
                    print(classes[class_id])
                    
                    # Add your path planning or obstacle avoidance logic here
        
        # Show frame
        cv2.imshow("SmartNav", frame)
        
        # Check for voice commands
        command = recognize_speech()
        if command == "stop":
            break
        
        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    navigate_with_vision()
