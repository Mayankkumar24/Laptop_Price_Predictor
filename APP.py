import streamlit as st
import numpy as np
import pandas as pd
import pickle

def main():
    df = pickle.load(open("df.pkl", "rb"))
    pipeline1 = pickle.load(open("pipeline1.pkl", "rb"))
    pipeline2 = pickle.load(open("pipeline2.pkl","rb"))
    pipeline3 = pickle.load(open("pipeline3.pkl","rb"))
    st.title("Laptop Price Predictor")
    flag = True
    brand = st.selectbox("Brand",df['brand'].unique())
    ram = st.number_input("Ram in (GB)")
    ram_type = st.selectbox("Ram Type",df['Ram_type'].unique())
    rom_type = st.selectbox("Rom Type",df['ROM_type'].unique())
    display_size = st.number_input("Display Size")
    res_width = st.selectbox("Resolution Width",df['resolution_width'].unique())
    res_height = st.selectbox("Resolution Height",df['resolution_height'].unique())
    os = st.selectbox("Operating System",['Windows 10 OS','Windows 11 OS','Windows OS','Mac High Sierra OS','Mac 10.15.3\t OS','Mac OS','DOS OS','DOS 3.0 OS','Chrome OS','Ubuntu OS','Android 11 OS'])
    warr = st.selectbox("Warranty",df['warranty'].unique())
    gen = st.selectbox("Generation",['3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th','13th','Intel','Apple','AMD','MediaTek'])
    pro = st.selectbox("Processor",['Intel i3','Intel i5','Intel i7','Intel i9','Other Intel Processor','AMD Ryzen 5','AMD Ryzen 7','Other AMD Processor','Apple Processor','Orher Processor'])
    rom = st.number_input("Rom in (GB)")
    gpu = st.selectbox("GPU",df['NewGPU'].unique())

    if st.button("Predict Price"):
        input_data = np.array(
            [brand, ram, ram_type, rom_type, display_size, res_width, res_height, os, warr, gen, pro, rom, gpu],
            dtype=object
        )
        input_data_df = pd.DataFrame([input_data], columns=df.columns)
        if (display_size==0.00):
            flag = False
            st.error(f"Please Enter a Valid Display Size")
        if (ram==0.00):
            flag = False
            st.error(f"Please Enter a Valid Ram Size")
        if (rom==0.00):
            flag = False
            st.error(f"Please Enter a Valid Rom Size")
        else:
            try:
                if (flag):
                    if (brand=="Apple"):
                        prediction = pipeline1.predict(input_data_df)
                        predicted_price = np.exp(prediction[0])
                        st.title(f"The Predicted Price is ₹ {predicted_price:.2f}")
                    else:
                        prediction = pipeline2.predict(input_data_df)
                        predicted_price = np.exp(prediction[0])
                        st.title(f"The Predicted Price is ₹ {predicted_price:.2f}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
if __name__ == "__main__":
    main()