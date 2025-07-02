import streamlit as st
from face_train import faces_extraction
from face_model import train_model
from face_test import test_model

st.title("OpenCV Face Recognition Application")

if 'people_data' not in st.session_state:
    st.session_state.people_data = []
if 'label_map' not in st.session_state:
    st.session_state.label_map = {}
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'minNeighbor' not in st.session_state:
    st.session_state.minNeighbor = 5

person_name = st.text_input("Enter Person's Name:")
person_images = st.file_uploader("Upload Images for This Person", type=['jpg', 'png','jpeg'], accept_multiple_files=True)

if st.button("Add Person"):
    if person_name and person_images:
        st.session_state.people_data.append({
            'name': person_name,
            'images': person_images
        })
        st.success(f"Added {person_name} with {len(person_images)} image(s)")
    else:
        st.warning("Please provide a name and atleast one image.")

st.subheader("People Added So Far:")
if st.session_state.people_data:
    for idx, person in enumerate(st.session_state.people_data):
        st.write(f"**{idx+1}. {person['name']}** - {len(person['images'])} image(s)")
else:
    st.write("None")

st.markdown("---")

st.session_state.minNeighbor = st.slider("Min Neighbors (Increase if faces found > total images. Decrease for vice versa)", min_value=5, max_value=20, value=5)

if st.button("Train Model"):
    if len(st.session_state.people_data) < 1:
        st.warning("Add atleast 1 person for recognition.")
    else:
        with st.spinner("Training model... Please wait."):
            features, labels, label_map,total_images,faces_found = faces_extraction(st.session_state.people_data,st.session_state.minNeighbor)
            if faces_found == 0:
                st.warning("0 Faces Found! Please try uploading clearer images")
            else:
                st.session_state.label_map = label_map
                train_model(features, labels)
                st.success(f"""Model has been trained successfully!
                Total images: {total_images}
                Faces Found: {faces_found}""")
                st.session_state.trained = True

st.info("ℹ️ If the number of **faces found is less than total images at minNeighbor = 5**, it likely means some images were unclear or the face wasn't detected properly.")

st.markdown("---")

st.subheader("Test Model")
test_image = st.file_uploader("Upload a new image to test model with",type=['jpg', 'png','jpeg'])

if st.button("Test Model"):
    if test_image and st.session_state.trained:
        with st.spinner("Analyzing image... Please wait."):
            label, confidence, image, name = test_model(test_image, st.session_state.label_map)

            if label or confidence or name:
                st.image(image,f"{name} detected with {confidence:.2f} confidence")
            else:
                st.write("No face was found by the model!")
    else:
        if not test_image:
            st.warning("No image was uploaded.")
        else:
            st.warning("Train the model first")