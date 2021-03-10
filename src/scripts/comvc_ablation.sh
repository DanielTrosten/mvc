python -m models.train -c mnist_contrast --model_config__loss_config__negative_samples_ratio -1 \
                                         --model_config__loss_config__adaptive_contrastive_weight 0
python -m models.train -c mnist_contrast --model_config__loss_config__negative_samples_ratio -1
python -m models.train -c mnist_contrast --model_config__loss_config__adaptive_contrastive_weight 0
python -m models.train -c mnist_contrast
