# ParkinsonVoiceApp
Parkinson app that uses Deep Learning for Early detection of the disese through just 2 voice recordings. 

conda create -n parkinson_app python=3.9 -y
conda activate parkinson_app
pip install tensorflow==2.10.0 keras==2.10.0 protobuf==3.19.6 numpy==1.23.5 librosa scikit-learn tqdm streamlit==1.22.0 streamlit-mic-recorder pydub imageio-ffmpeg

conda install python=3.10 pip -y
conda install matplotlib
conda install seaborn


python train_crnn.py

streamlit run app_crnn.py

