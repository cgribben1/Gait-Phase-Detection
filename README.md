# Using Deep Learning on Sensor Time Series Data to Estimate Gait Phase

### Project Overview
This project aims to recreate the embedded machine learning model used in an experimental therapeutic device developed by researchers at Imperial College London. The device, known as BioStim (see figure below), leverages Functional Electrical Stimulation (FES) to reduce stress on the knee joint in osteoarthritis patients and recovering athletes. The embedded model **estimates the gait phase in real-time, enabling the device to administer FES at the optimal point within the gait**.

<br>

![schema](https://github.com/cgribben1/Deep-Learning-Gait-Phase-Detection/assets/143657285/fd9a9680-d3e3-4beb-9ff1-f290f4bfa0d5)

### The Dataset
- The dataset comprises 6 trials of **sensor time series data** from two motion sensors (left and right), including **linear accelerations** and **angular velocities**.
- The dataset includes **walking ('walk'), jogging ('jog'), and various activities ('var')**.
- Data is sampled at 500Hz, resulting in a time interval of 2ms between successive rows.
- The **gait cycle is split up into 12 different phases**.
- The modeling task involves **predicting the scalar integer corresponding to the gait phase at a given time 't'** using the sensor data.

### Modeling Approach

#### Movement "Mode" Classification
- Build and train a movement mode classifier to **distinguish between 'walk' and 'jog' samples**.
- Use this classifier to separate 'walk' and 'jog' for subsequent phase classification.

#### Phase Classification
- Train **separate models** for 'walk' and 'jog' to optimize specificity.
- During prediction, **pass each sample through the mode classifier before feeding it into the relevant phase classifier**.

#### Model Type Used
**Convolutional Neural Networks** (CNNs) are chosen for their effectiveness in extracting features from time series data. Therefore, I have chosen to leverage their predictive power here.

### Future Improvements
The focus of this project was to write code which is able to **scale to a larger dataset**; therefore, the **overall fidelity of the models created here is not overly important here**. Bearing this in mind, any improvement in model performance is intended to come from an expansion in the dataset; **obtaining data from more participants to enhance classifier performance on diverse samples**.

---

The **full project report** is provided here in both **.html** and **.ipynb** format.

N.B. you will have to open the .ipynb file in a Jupyter notebook to make use of the section links within the ToC...
