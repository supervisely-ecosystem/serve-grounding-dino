import supervisely as sly
from dotenv import load_dotenv
from src.grounding_dino import GroundingDino


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv("supervisely.env")

model = GroundingDino(
    use_gui=True,
    use_serving_gui_template=True,
    sliding_window_mode="none",
)
model.gui.pretrained_models_table.set_active_row(1)
model.serve()