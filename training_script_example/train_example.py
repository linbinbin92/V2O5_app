# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import torch

# import some common libraries
import numpy as np
import os,sys, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
# import some common libraries

import numpy as np
import os, json, cv2, random

from detectron2.structures import BoxMode
from pycocotools.mask import encode
from matplotlib import image
import matplotlib.pyplot as plt
import pycocotools as pc


outputdir = "output"

root_path = os.getcwd()
print("root_path is ", root_path)
print("Let's use", torch.cuda.device_count(), "GPUs!")

depth = 2
def get_fiber_dicts(img_dir):
	#print("inside get_fiber_dicts and the folder is " , img_dir)
	#os.chdir(os.path.abspath(img_dir))
	dataset_dicts = []
	root, dirs, files = next(os.walk(img_dir, topdown = True))
		#print("root is ", root)
		#if root[len("."):].count(os.sep) < 2: #depth = 2
		#	for dir in dirs:
		#		print("in get fiber loop the dirname is ", dir)

	for i, name in enumerate(files):
		record = {}
		idx = name.split(".")[0].split("_")[1]
		#print("number of image is",i)
		filename = os.path.join(root, name)
		#print("inside get fiber_dict with", filename)
		height, width = cv2.imread(filename).shape[:2]
#			print(height,width)
		record["file_name"] = filename
		record["image_id"] = idx
		record["height"] = height
		record["width"] = width


		objs =[]
		for i in range(20):

			try:

				path=os.path.join(img_dir, idx, 'mask', "image_{}_mask_{}.png".format(idx,float(i+1)))
				#print(path)
				image = plt.imread(path)
				mask_dir = os.path.join(os.path.dirname(path))
				mask = np.asarray(image,dtype=np.bool, order="F")
				#print(mask)
				rle=pc.mask.encode(mask)
				#print(rle)
				bbox=pc.mask.toBbox(rle)
				#print(bbox)
				obj = { "bbox": list(bbox[0]),
					"bbox_mode": BoxMode.XYWH_ABS,
					"segmentation": rle[0],
					"category_id": 0,
					}
					#print(obj)


				objs.append(obj)

			except:
				pass

		record["annotations"] = objs

		dataset_dicts.append(record)


	return dataset_dicts

################################Register Data################################

train_val_data_path = "/work/scratch/ac01asac/TorchGPU/Fiber/"

for d in ["Train_new","Val_new"]:
	DatasetCatalog.register("Fiber_" + d, lambda d = d: get_fiber_dicts(os.path.join(train_val_data_path, d)))
	print("dataset size is", np.size(DatasetCatalog.get("Fiber_"+ d)))
	data_path = os.path.join(train_val_data_path, d)
	print("registration finisched at datapath {} ".format(data_path))
	MetadataCatalog.get("Fiber_"+ d).set(thing_classes="fiber")

Fiber_meta_train = MetadataCatalog.get("Fiber_Train_new")
Fiber_meta_val = MetadataCatalog.get("Fiber_Val_new")
print("Train meta {} ".format(Fiber_meta_train))
print(Fiber_meta_train.thing_classes)
print("Val meta {} ".format(Fiber_meta_val))
print(Fiber_meta_val.thing_classes)



#######################################Check registration################

num_img = 0  ## how many image to check


for d in random.sample(dataset_dicts, num_img):
	img = cv2.imread(d["file_name"])
	visualizer = Visualizer(img[:, :, ::-1], metadata=Fiber_meta_train, scale=0.5)
	out = visualizer.draw_dataset_dict(d)
	out.save("img_check_reg_{}.png".format(d["image_id"]))
	print("img_check_reg saved as ", d["file_name"] )

################Train#################################################

from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("Fiber_Train_new",)
print("dataset size is:", np.size((dataset_dicts)))

cfg.DATASETS.TEST = ("Fiber_Val_new",)
print(cfg.DATASETS)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.02/16 * cfg.SOLVER.IMS_PER_BATCH    # pick a good LR
cfg.SOLVER.MAX_ITER = 8000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
## NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.INPUT.MASK_FORMAT= "bitmask"
#cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS =  False
cfg.OUTPUT_DIR = outputdir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print("output folder is ", os.path.abspath(cfg.OUTPUT_DIR))
print('load cfg succesussfull')

##### uncomment when no GPU avail our test purposes ###########

#cfg.MODEL.DEVICE='cpu'

############################## Train part ############

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

######################## Inference

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

test_path = os.path.join(root_path,"Fiber","Test")
for root, dirs, files in os.walk(test_path, topdown = True):
	for i, name in enumerate(files):
		#print(name)
		imfile = os.path.join(test_path,name)
		print(imfile)
		im = cv2.imread(imfile)
		outputs=predictor(im)
		v = Visualizer(im[:, :, ::-1], metadata=Fiber_meta_train, scale=3, instance_mode=ColorMode.IMAGE_BW)
		out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		out.save("./{}/img_pred_{}.png".format(outputdir, name.split(".")[0]))
		print("output image saved in ", os.path.abspath(outputdir))


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("Fiber_Val_new", ("bbox", "segm"), False, output_dir="./{}/".format(outputdir))
val_loader = build_detection_test_loader(cfg, "Fiber_Val_new")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
