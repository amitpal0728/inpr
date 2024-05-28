import streamlit.components.v1 as components
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
import easyocr
import tempfile

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

st.set_page_config(page_title="INPR", page_icon="🤖")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after{content : "Vinit, Nikunj, Parag, Murtaza";
                         display : block;
                         position: relative;
                         color: #fff4e9;
                         font: san serif;
                         padding: 10px;
                         top:3px;
                         visibility: visible;}
            header {visibility: hidden;}
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Intelligent Number Plate Recognition System")

# Function to capture video from webcam
def capture_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break

# Function to process image and perform OCR
def process_image(image):
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detection
    lplate_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    found = lplate_data.detectMultiScale(image_gray)

    # Process each detected plate
    for (x, y, w, h) in found:
        # Draw a rectangle around the detected plate
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the plate region from the grayscale image
        plate_image = image_gray[y:y+h, x:x+w]

        # Perform OCR on the plate image
        result = reader.readtext(plate_image)
        text = ' '.join([box[1] for box in result])

        # Display the recognized text on the image
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image

def process_video(video_bytes):
    # Create a temporary file to save the video bytes
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    # Create a VideoWriter object to save the processed video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Load the plate detection model
    lplate_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    # Create a Streamlit placeholder for the video frames
    stframe = st.empty()

    # Loop through each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection
        found = lplate_data.detectMultiScale(frame_gray)

        # Process each detected plate
        for (x, y, w, h) in found:
            # Draw a rectangle around the detected plate
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract the plate region from the grayscale image
            plate_image = frame_gray[y:y+h, x:x+w]

            # Perform OCR on the plate image
            result = reader.readtext(plate_image)
            text = ' '.join([box[1] for box in result])

            # Display the recognized text on the image
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the processed frame in the Streamlit app
        stframe.image(frame, channels="BGR", use_column_width=True)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
    return processed_video_path


def process_frame(frame):
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the plate detection model
    lplate_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    # Detection
    found = lplate_data.detectMultiScale(frame_gray)

    # Process each detected plate
    for (x, y, w, h) in found:
        # Draw a rectangle around the detected plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the plate region from the grayscale image
        plate_image = frame_gray[y:y + h, x:x + w]

        # Perform OCR on the plate image
        result = reader.readtext(plate_image)
        text = ' '.join([box[1] for box in result])

        # Display the recognized text on the image
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

def embed_chatbot():
    chatbot_url = "https://cdn.botpress.cloud/webchat/v2/shareable.html?botId=e0b44ea1-6b0e-4390-884a-d91af3a771df"


    st.markdown(f'<iframe src="{chatbot_url}" width="100%" height="500" style="border:none;"></iframe>', unsafe_allow_html=True)


# Sidebar options
option = st.sidebar.selectbox("Choose Option", ["Webcam", "Upload Photo", "Upload Video"])

if option == "Webcam":
    if st.sidebar.button("Enable Webcam"):
        # Start the webcam
        cap = cv2.VideoCapture(0)

        # Create a Streamlit placeholder for the video frames
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture image from webcam. Please ensure the webcam is connected and working.")
                break

            # Process the frame
            processed_frame = process_frame(frame)

            # Encode the frame as JPEG
            _, jpeg = cv2.imencode('.jpg', processed_frame)
            frame_bytes = jpeg.tobytes()

            # Display the processed frame in the Streamlit app
            stframe.image(frame_bytes, channels="BGR", use_column_width=True)

        cap.release()
elif option == "Upload Photo":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        processed_image = process_image(image)
        st.image(processed_image, channels="BGR", use_column_width=True)
elif option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        video_bytes = uploaded_file.read()
        processed_video_path = process_video(video_bytes)
        st.video(processed_video_path)

st.sidebar.markdown("## Chatbot")
if st.sidebar.button("Launch Chatbot"):
    embed_chatbot()

