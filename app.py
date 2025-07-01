import streamlit as st

st.title("OpenCV Face Recognition Application")

if 'people_data' not in st.session_state:
    st.session_state.people_data = []

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

if st.button("Train Model"):
    if len(st.session_state.people_data) < 1:
        st.warning("Add atleast 1 person for recognition.")
    else:
        st.success("You can now trigger model training here.")
