import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage.measure import label, regionprops
from skimage.transform import rotate
import math
import pandas as pd
import seaborn as sns
from PIL import Image
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog


def rotate_image(img):
    """
    for any image calculate the orientation of the object in the image and rotate the image in a way that long axis of object is parallel to x axis

    Input:
        img = np.array (m*n) ## For a 2d mask with just one object

    Return:
        rotated_image: np.array
        orientation : orientation of thvisualize_predictione object in degree
    """

    label_image = label(img)
    regions = regionprops(label_image)
    for props in regions:
        orientation = math.degrees(props.orientation)
    rotated_image = rotate(img, 90 - orientation, resize=True)
    #if orientation < 0:
    #    orientation += 180
    return rotated_image, orientation+90


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([1, 4], dtype=np.int32)

    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]  # -->np.array (512,) [250,251,]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes[0] = np.array([x1, x2, y1, y2])
    return boxes.astype(np.int32)


def stats(predictor,image):
    """
    for any given image calculate the properties of object in the images prediction based on the given model

    Inputs:
        model : Dtectron2 model
        image : image for model prediction (np.array)

    Returns:
        width,height,area,orientation for all the objects in the predicted masks

    """
    predictions = predictor(image)
    masks = predictions['instances'].pred_masks
    masks = masks.cpu().numpy()
    number = masks.shape
    width = []
    height = []
    orientation = []
    area = []
    perimeters = []
    is_convexs = []
    area_cvs = []
    perimeters = []
    is_convexs = []
    aspect_ratios = []
    angle_rotated_boundingboxs = []
    angle_ecllipses = []
    circularities = []
    convexities  = []
    solidities = []
    eccentricities = []
    for i in range(masks.shape[0]):
        mask = masks[i, :, :]
        mask = np.array(mask, np.uint8)
        contours, hierarchy = cv.findContours(image=mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        cnt=contours[0]
        perimeter = cv.arcLength(cnt, True)
        is_convex = cv.isContourConvex(cnt)
        area_cv =  cv.contourArea(cnt)
        rect = cv.minAreaRect(cnt)
        rect = list(rect)
        aspect_ratio = np.min(rect[1])/np.max(rect[1])
        angle_rotated_bounding_box = rect[2]
        circularity = (4*np.pi*area_cv)/(perimeter**2)
        hull = cv.convexHull(cnt)
        convex_perimeter = cv.arcLength(hull,True)
        hull_area = cv.contourArea(hull)
        solidity =  float(area_cv)/hull_area
        (x,y), (minor_ax,major_ax),angle_ecllipse = cv.fitEllipse(cnt)
        eccentrcity = np.sqrt(1-(minor_ax**2/major_ax**2))
        rot, orien = rotate_image(mask)
        bbox = extract_bboxes(rot)
        bbox = bbox[0]
        wdth = abs(bbox[0] - bbox[1])
        hght = abs(bbox[2] - bbox[3])
        are=np.sum(mask != 0)
        width.append(min(wdth, hght))
        height.append(max(wdth, hght))
        orientation.append(orien)
        area.append(are)
        area_cvs.append(area_cv)
        perimeters.append(perimeter)
        is_convexs.append(is_convex)
        aspect_ratios.append(aspect_ratio)
        angle_rotated_boundingboxs.append(angle_rotated_boundingbox)
        angle_ecllipses.append(angle_ecllipse)
        circularities.append(circularity)
        convexities.append(convexity)
        solidities.append(soliditiy)
        eccentricities.append(eccentricity)

    return number, width, height, area, orientation, area_csvs,perimeters, is_convexs, aspect_ratios, angle_rotated_boundingboxs, angle_ecllipses,circularities, convexities, solidities, eccentricities,  masks

def visualize_prediction(predictor,image):

    visualizer = Visualizer(image, instance_mode=ColorMode.IMAGE_BW)
    predictions = predictor(image)
    instances = predictions["instances"]
    vis_output = visualizer.draw_instance_predictions(predictions=instances)

    return vis_output

class LoadCOCO:
    '''
    handeling and modifying image masks in COCO fromat from JSON file.

    #### apparently due to a bug in JSON library in the JSON module the JSON file must be in the
    same directory as the running the model if directly implemented in the model so as the GT images,
    #### sometimes if there is any image in the and subdirectories it may gives an error ( coco looks
    for all images in the current directory and subdirectories.

    attributes :
                images: returning all images names in the JSON file
                mask: return a binary mask for each image ground truth --> List
                instances: return a list of masks for all instances in the ground truth mask

    methods:
                difference(model) -->   difference between ground truth mask and predicted mask
                                        by the given model
                visual_diff()     -->   visualizing the difference for each image in JSON file.
                                        **** color mapping ****
                                        return an image where blue regions are where prediction is
                                        correct red is the regions where model does not predict a mask
                                        for a fiber and green areas where there was no fiber but model
                                        predicted a mask.


    Example:
            ... loading the pre-trained model ...
            coco = LoadCOCO('avg_stack.json')
            diff = coco.difference(predictor)
            coco.visual_diff()

    '''

    def __init__(self, json) -> None:
        """
        Input: JSON file
        OUtput: None
        """
        self.path = path = os.path.join(os.getcwd(), json)
        self.coco = COCO(self.path)
        self.images = self.coco.dataset['images']
        self.annotations = self.coco.dataset['annotations']
        self.dataset = []
        self._anns = {}
        self.msk = None

        for image in self.images:
            img = {'id': image['id'],
                   'height': image['height'],
                   'width': image['width'],
                   'name': image['file_name']
                   }
            self.dataset.append(img)
            self._anns[image['id']] = []

        for annotation in self.annotations:
            id = annotation['image_id']
            self._anns[id].append(annotation['segmentation'])

        self.load_anns()
        self.load_mask()
        self.ground_prop()

    def load_anns(self):
        for image in self.dataset:
            anns = self._anns[image['id']]
            image['annotations'] = anns

    def load_mask(self):

        for image in self.dataset:
            instance_masks = []
            for ann in image['annotations']:
                m = self.annToMask(ann, image["height"],
                                   image["width"])
                instance_masks.append(m)

            masks = np.stack(instance_masks, axis=2).astype(np.bool)
            image['instance'] = masks
            msk = np.sum(masks, axis=2)
            msk = msk != 0
            image['mask'] = msk.astype(np.int8)

        del image['annotations']

    @property
    def mask(self):

        return [image['mask'] for image in self.dataset]

    @property
    def instances(self):
        return [image['instance'] for image in self.dataset]

    def annToRLE(self, segm, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        # segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)

        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def difference(self, model):
        """
        For a given model calculate the difference between ground truth and the model Predictions

        Input:
            Detectron2 model

        Output:
            List of np.array for all images in self.dataset

        """
        for image in self.dataset:
            diffs = []
            print(image['name'])
            img = plt.imread(image['name'])
            outputs = model(img)
            masks = outputs['instances'].pred_masks
            pred_masks = masks.cpu().numpy()
            pred_mask = pred_masks.sum(axis=0).astype(np.int8)
            pred_mask = pred_mask != 0
            out_diff = np.subtract(image['mask'], pred_mask)
            diff = pred_mask != image['mask']
            in_diff = np.where(diff == out_diff, 0, 1)
            similar = image['mask'] == pred_mask
            similar = np.where(image['mask'] == 0, 0, similar)
            diff = np.stack([out_diff, in_diff, similar], axis=2)
            image['diff'] = diff
            image['pred'] = pred_mask
            diffs.append(diff)
        return diffs

    def visual_diff(self, visualize=True):
        for image in self.dataset:
            final = np.zeros((image['height'], image['width'], 3))
            final[:, :, 0] = image['diff'][:, :, 0]
            final[:, :, 1] = image['diff'][:, :, 1]
            final[:, :, 2] = image['diff'][:, :, 2]
            if visualize:
                plt.imshow(final)
                plt.show()
            return final

    def prps(self, model, image):
        """
        for any given image calculate the properties of object in the images prediction based on the given model

        Inputs:
            model : Dtectron2 model
            image : image for model prediction (np.array)

        Returns:
            width,height,area,orientation for all the objects in the predicted masks

        """

        img = plt.imread(image)
        outputs = model(img)
        masks = outputs['instances'].pred_masks
        masks = masks.cpu().numpy()
        print(masks.shape)
        width = []
        height = []
        orientation = []
        area = []
        for i in range(masks.shape[0]):
            mask = masks[i, :, :]
            rot, orien = rotate_image(mask)
            bbox = extract_bboxes(rot)
            bbox = bbox[0]
            wdth = abs(bbox[0] - bbox[1])
            hght = abs(bbox[2] - bbox[3])
            are = np.sum(mask != 0)
            width.append(min(wdth, hght))
            height.append(max(wdth, hght))
            orientation.append(orien)
            area.append(are)
        return width, height, area, orientation

    def ground_prop(self):
        """
        calculate the properties of objects in the ground truth image and add them to each image in self.dataset
        """
        for image in self.dataset:
            masks = image['instance']
            width = []
            height = []
            orientation = []
            area = []
            # print(masks.shape)
            for i in range(masks.shape[-1]):
                mask = masks[:, :, i]
                rot, orien = rotate_image(mask)
                bbox = extract_bboxes(rot)
                bbox = bbox[0]
                wdth = abs(bbox[0] - bbox[1])
                hght = abs(bbox[2] - bbox[3])
                are = np.sum(mask != 0)
                width.append(min(wdth, hght))
                height.append(max(wdth, hght))
                orientation.append(orien)
                area.append(are)
                # print(wdth,hght,orien,are)
            image['widths'] = width
            image['heights'] = height
            image['area'] = area
            image['orientation'] = orientation

    def plot_distribution(self, model, visualize=True):
        """
        Create a DataFrame all heights widths,Orientations and areas for all object in a ground truth and the predicted maskUtils
        Input:
            model : Detectron2 model
            visualize: whether to plot distribution plots or not

        Return:
            df : pd.DataFrame
        """
        for image in self.dataset:
            width, height, area, orientation = self.prps(model, image['name'])
            len_p = len(width)
            pred = np.array((width, height, area, orientation))
            gt = np.array((image['widths'], image['heights'], image['area'], image['orientation']))
            len_g = len(image['widths'])
            concat = np.concatenate((pred, gt), axis=1)
            p = np.ones((1, len_p))
            g = np.ones((1, len_g)) * 2
            t = np.concatenate((p, g), axis=1)
            final = np.concatenate((concat, t))
            final = np.transpose(final)
            df = pd.DataFrame(final, columns=['Width', 'Height', 'Area', 'Orientation', 'Type'])
            df.replace({'Type': {1.0: 'Predictions', 2.0: 'Ground Truth'}}, inplace=True)
            if visualize:
                sns.color_palette("mako", as_cmap=True)
                sns.displot(df, x='Height', hue='Type', multiple="dodge", stat="density", palette='mako', kde=True)
                sns.displot(df, x='Width', hue='Type', multiple="dodge", stat="density", palette='mako', kde=True)
                sns.displot(df, x='Area', hue='Type', multiple="dodge",
                            stat="density", palette='mako', kde=True)
                sns.displot(df, x='Orientation', hue='Type', multiple="dodge",
                            stat="density", palette='mako', kde=True)
                plt.show()
            return df
