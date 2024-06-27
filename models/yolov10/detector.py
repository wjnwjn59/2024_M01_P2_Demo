import cv2
import time
import yaml
import onnxruntime
import numpy as np
from typing import Tuple, List

class YOLOv10:
    def __init__(self,
                 model_path: str,
                 class_mapping_path: str,
                 original_size: Tuple[int, int] = (1280, 720),
                 score_threshold: float = 0.1,
                 conf_thresold: float = 0.4,
                 iou_threshold: float = 0.4,
                 device: str = "CPU") -> None:
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path

        self.device = device
        self.score_threshold = score_threshold
        self.conf_thresold = conf_thresold
        self.iou_threshold = iou_threshold
        self.image_width, self.image_height = original_size
        self.create_session()

    def create_session(self) -> None:
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = ['CPUExecutionProvider']
        if self.device.casefold() != "cpu":
            providers.append("CUDAExecutionProvider")
        session = onnxruntime.InferenceSession(self.model_path, providers=providers)
        self.session = session
        self.model_inputs = self.session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
        self.input_shape = self.model_inputs[0].shape
        self.model_output = self.session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        self.input_height, self.input_width = self.input_shape[2:]

        if self.class_mapping_path is not None:
            with open(self.class_mapping_path, 'r') as file:
                yaml_file = yaml.safe_load(file)
                self.classes = yaml_file['names']
                self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))

        # Scale input pixel value to 0 to 1
        input_image = resized / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor
    
    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y 
    
    def postprocess(self, outputs):
        predictions = np.squeeze(outputs)
        scores = predictions[:, 4]
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = predictions[:, 5].astype(int)

        # Rescale box
        boxes = predictions[:, :4]
        
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.int32)

        detections = []
        for bbox, score, label in zip(boxes, scores, class_ids):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,
                "class_name": self.get_label_name(label)
            })
        return detections
    
    def get_label_name(self, class_id: int) -> str:
        return self.classes[class_id]
        
    def detect(self, img: np.ndarray) -> List:
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        return self.postprocess(outputs)
    
    def draw_detections(self, img, detections: List):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            detections: List of detection result which consists box, score, and class_ids
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        for detection in detections:
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = detection['box'].astype(int)
            class_id = detection['class_index']
            confidence = detection['confidence']

            # Retrieve the color for the class ID
            color = self.color_palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create the label text with class name and score
            label = f"{self.classes[class_id]}: {confidence:.2f}"

            # Set the font scale and thickness for larger text
            font_scale = 1  # Increase this value for larger text
            thickness = 1   # Increase this value for thicker text

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

def inference(img_path: str, 
              weight_path: str = 'weights/helmet_safety_best.onnx', 
              yaml_path: str = 'helmet_safety.yaml') -> np.ndarray:
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    detector = YOLOv10(model_path=weight_path,
                      class_mapping_path=yaml_path,
                      original_size=(w, h))
    detections = detector.detect(img)
    detector.draw_detections(img, detections=detections)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

