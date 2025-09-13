import streamlit as st









# --------------- streamlit ui ---------------

st.title("Predictive Text")
st.write("Enter a seed phrase to get next word suggestions or generate text.")


seed = st.text_input("Seed text", value="There is a")


if st.button("Suggest next words"):
	
	st.write("### Top Suggestions:")
	

if st.button("Generate text"):
	
	st.write("### Generated Text:")
	