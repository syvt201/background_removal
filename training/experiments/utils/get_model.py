import segmentation_models_pytorch as smp


def get_model(model_name, encoder_name='resnet34', encoder_weights='imagenet', num_classes=1, activation=None):
    """
    Returns a segmentation model from segmentation_models_pytorch.

    Parameters:
    ----------
    model_name : str - Name of the model ('unet' or 'deeplabv3plus')
    encoder_name : str - Backbone encoder (e.g., 'resnet34', 'efficientnet-b0')
    encoder_weights : str - Pretrained weights ('imagenet' or None)
    num_classes : int - Output classes (default: 1 for binary segmentation)
    activation : str or None - Activation at output layer ('sigmoid', 'softmax2d', or None)

    Returns:
    -------
    model : nn.Module - Instantiated segmentation model
    """

    model_name = model_name.lower()

    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation
        )

    elif model_name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation
        )

    else:
        raise ValueError(f"Unsupported model_name '{model_name}' — choose from ['unet', 'deeplabv3']")

    return model
