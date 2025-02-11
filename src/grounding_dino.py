import supervisely as sly
import os
from supervisely.nn.inference import CheckpointInfo
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from supervisely.nn.prediction_dto import PredictionBBox


class GroundingDino(sly.nn.inference.PromptBasedObjectDetection):
    FRAMEWORK_NAME = "GroundingDino"
    MODELS = "src/models.json"
    APP_OPTIONS = "src/app_options.yaml"
    INFERENCE_SETTINGS = "src/inference_settings.yaml"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # disable GUI widgets
        self.gui.set_project_meta = self.set_project_meta

    def load_model(
        self,
        model_files: dict,
        model_info: dict,
        model_source: str,
        device: str,
        runtime: str,
    ):
        checkpoint_path = model_files["checkpoint"]
        if sly.is_development():
            checkpoint_path = "." + checkpoint_path
        self.classes = []
        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=os.path.basename(checkpoint_path),
            model_name=model_info["meta"]["model_name"],
            architecture=self.FRAMEWORK_NAME,
            checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
            model_source=model_source,
        )
        self.device = torch.device(device)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            checkpoint_path
        ).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.model = self.model.to(self.device)

    def predict(self, image_path, settings):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        text_prompt = settings.get("text_prompt", "all objects .")
        box_theshold = float(settings.get("box_threshold", 0.3))
        text_theshold = float(settings.get("text_threshold", 0.3))
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_theshold,
            text_threshold=text_theshold,
            target_sizes=[image.size[::-1]],
        )
        prediction = self.postprocess_prediction(prediction[0])
        return prediction

    def predict_batch(self, images_np, settings):
        text_prompt = settings.get("text_prompt", "all objects .")
        box_threshold = float(settings.get("box_threshold", 0.3))
        text_threshold = float(settings.get("text_threshold", 0.3))

        images = [Image.fromarray(img) for img in images_np]
        text_prompt = [text_prompt] * len(images)

        inputs = self.processor(
            text=text_prompt, images=images, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        batch_predictions = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1] for image in images],
        )
        batch_predictions = [
            self.postprocess_prediction(prediction) for prediction in batch_predictions
        ]
        return batch_predictions

    def postprocess_prediction(self, prediction):
        postprocessed_pred = []
        bboxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"]
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox
            bbox_yxyx = [int(y1), int(x1), int(y2), int(x2)]
            score = round(float(score), 2)
            sly_bbox = PredictionBBox(label, bbox_yxyx, score)
            postprocessed_pred.append(sly_bbox)
        return postprocessed_pred

    def set_project_meta(self, inference):
        """The model does not have predefined classes.
        In case of prompt-based models, the classes are defined by the user."""
        self.gui._model_classes_widget_container.hide()
        return

    def _load_model_headless(
        self,
        model_files: dict,
        model_source: str,
        model_info: dict,
        device: str,
        runtime: str,
        **kwargs,
    ):
        """
        Diff to :class:`Inference`:
           - _set_model_meta_from_classes() removed due to lack of classes
        """
        deploy_params = {
            "model_files": model_files,
            "model_source": model_source,
            "model_info": model_info,
            "device": device,
            "runtime": runtime,
            **kwargs,
        }
        self._load_model(deploy_params)

    def _create_label(self, dto):
        """
        Create a label from the prediction DTO.
        """
        class_name = dto.class_name
        obj_class = self.model_meta.get_obj_class(class_name)
        if obj_class is None:
            self._model_meta = self.model_meta.add_obj_class(
                sly.ObjClass(class_name, sly.Rectangle)
            )
            obj_class = self.model_meta.get_obj_class(class_name)
        geometry = sly.Rectangle(*dto.bbox_tlbr)
        tags = []
        if dto.score is not None:
            tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
        label = sly.Label(geometry, obj_class, tags)
        return label
