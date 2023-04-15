# https://www.youtube.com/watch?v=dO0AajkNOH0

# https://pytorch.org/: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip3 install pycocotools
# pip3 install pixellib opencv-python
# pip3 install pixellib --upgrade
 
# Download the weights for a pre-trained model that has been trained on the COCO DS
# Goto https://github.com/matterport/Mask_RCNN/releases / Mask R-CNN 2.0 / mask_rcnn_coco.h5

import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2

cap = cv2.VideoCapture(0)

# Setup model
segmentation_video = instanceSegmentation()
# https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend
segmentation_video.load_model('../../PreTrainedModelWeights/PointRend_R50-FPN-Sched3x.pkl',
                              confidence=.7, detection_speed="fast")
segmentation_video.process_camera(cam=cap, show_bboxes=True, frames_per_second=15, 
                                  check_fps=True, show_frames=True, frame_name="frame",
                                  output_video_name="output.mp4v")
