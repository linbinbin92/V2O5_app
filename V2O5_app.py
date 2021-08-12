import streamlit as st
# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
from PIL import Image
# import some common libraries
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
import numpy as np
import cocomask
import seaborn as sns
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot as plt
import pandas as pd
import altair as alt

# import some common detectron2 utilities

setup_logger()
##########Config file###################################
cfg = get_cfg()
##### uncomment when no GPU avail our test purposes ###########

cfg.MODEL.DEVICE = 'cpu'
cfg.OUTPUT_DIR = os.path.join(os.getcwd(), "V2O5_app\\model_results")
######################## setting####################################

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("Fiber_Train_new_360",)
cfg.DATASETS.TEST = ("Fiber_Val_new_40",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.02 / 16 * cfg.SOLVER.IMS_PER_BATCH  # pick a good LR
cfg.SOLVER.MAX_ITER = 16000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  #

cfg.INPUT.MASK_FORMAT = "bitmask"
# cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS =  False
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)


col1, mid, col2 = st.beta_columns([25,1,2])
with col1:
    #st.markdown('<h1 style="color: black;">V2O5 Detector: a Data-driven model</h1>',
    #                        unsafe_allow_html=True)
    st.write("""# V2O5 Detector: a synthetic data-driven deep learning model """)

with col2:
    st.image('logo_mfm.png', width=120)

    #st.write('V2O5 Detector, a Data-driven model by MFM')


#st.image('./logo_mfm.png')
#st.write("""# Data-driven model by MFM """)
st.markdown('---')
st.write("""
## This web-based interative tool provide a deep-learning-based *image* analysis on the fly.
- To use the model, upload the image and select the functionality.
- For further information how the model works and was trained see:  [arXiv Doi](https://github.com/linbinbin92)
- If you use this tool for your research, please cite:  [arXiv/xxx](https://github.com/linbinbin92)

""")
st.markdown('---')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input image file", type=["png", "jfif", "tif"])

if uploaded_file is not None:
    #    img = resizeimg(uploaded_file)

    st.write("""This is how your image looks like: """)
    st.image(uploaded_file)

    im = Image.open(uploaded_file)
    im = np.array(im.convert("RGB"))  # pil to cv
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
else:
    st.warning('Please upload a image.')

run_ = st.sidebar.button('Start image analysis')

if run_:

    #demo = VisualizationDemo(cfg)
    #predictions, visualized_output = demo.run_on_image(im)
    st.markdown('---')
    st.write("""Programm running...... """)

    vis_output = cocomask.visualize_prediction(predictor,im)
    st.write("""This is your prediction: """)

    st.image(vis_output.get_image()[:, :, ::-1])

    #rund_statstics_ = st.sidebar.button('Show statistical results')


rund_statstics_ = st.sidebar.button('Show statistics')

if rund_statstics_:

    st.write("""This are the statistics we could provide:""")
    st.markdown('---')

    number, width, height, area, orientation = cocomask.stats(predictor, im)
    number = list(number)
    st.write("""In this figure of image size {}x{} we have found __{}__ particles :""".format(number[1],number[2], number[0]))

    width = pd.Series(width, name="width")
    height = pd.Series(height, name="height")
    area = pd.Series(area, name="area")
    orientation = pd.Series(orientation, name="orientation")

    f, axes = plt.subplots(4, 1, figsize=(8, 15))
    sns.distplot(width,  bins = 20, label='Width', kde=True, ax=axes[0])
    sns.distplot(height, bins = 20, label='Height', kde=True,ax=axes[1])
    sns.distplot(area, bins = 20, label='Area', kde=True, ax=axes[2])
    sns.distplot(orientation, bins = 20,label='Orientation',  kde=True, ax=axes[3])

    st.pyplot(f)
else:
    pass

with st.sidebar:

    st.text("")
    st.text("")
    st.text("")

    """
    #### :desktop_computer: [Source code in Github](https://github.com/linbinbin92)
    """
