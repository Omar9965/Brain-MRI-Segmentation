import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='efficientnet-b7',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation='sigmoid',
    decoder_attention_type='scse',
)