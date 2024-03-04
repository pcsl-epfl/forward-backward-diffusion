from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch as th


def load_classifier(classifier_name="convnext_base"):
    if classifier_name == "convnext_base":
        weights = ConvNeXt_Base_Weights.DEFAULT
        model = convnext_base(weights=weights)
        model.eval()
        module_names = [
            "features.0",
            "features.1.0",
            "features.1.1",
            "features.1.2",
            "features.2",
            "features.3.0",
            "features.3.1",
            "features.3.2",
            "features.4",
            "features.5.0",
            "features.5.1",
            "features.5.2",
            "features.5.3",
            "features.5.4",
            "features.5.5",
            "features.5.6",
            "features.5.7",
            "features.5.8",
            "features.5.9",
            "features.5.10",
            "features.5.11",
            "features.5.12",
            "features.5.13",
            "features.5.14",
            "features.5.15",
            "features.5.16",
            "features.5.17",
            "features.5.18",
            "features.5.19",
            "features.5.20",
            "features.5.21",
            "features.5.22",
            "features.5.23",
            "features.5.24",
            "features.5.25",
            "features.5.26",
            "features.6",
            "features.7.0",
            "features.7.1",
            "features.7.2",
            "classifier",
        ]
    else:
        raise f"Classifier {classifier_name} not implemented"

    preprocess = weights.transforms()

    def preprocess(img):
        img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
        model_preprocess = weights.transforms()
        return model_preprocess(img)

    return model, preprocess, module_names
