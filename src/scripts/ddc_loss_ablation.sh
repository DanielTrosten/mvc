# SiMVC
python -m models.train -c mnist --best_loss_term "tot" --model_config__loss_config__funcs "ddc_1"
python -m models.train -c mnist --best_loss_term "tot" --model_config__loss_config__funcs "ddc_2"
python -m models.train -c mnist --best_loss_term "tot" --model_config__loss_config__funcs "ddc_3"
python -m models.train -c mnist --best_loss_term "tot" --model_config__loss_config__funcs "ddc_1|ddc_2"
python -m models.train -c mnist --best_loss_term "tot" --model_config__loss_config__funcs "ddc_1|ddc_3"
python -m models.train -c mnist --best_loss_term "tot" --model_config__loss_config__funcs "ddc_2|ddc_3"
python -m models.train -c mnist --best_loss_term "tot" # Uses all losses by default

# ScMVC
python -m models.train -c mnist_contrast --best_loss_term "tot" --model_config__loss_config__funcs "ddc_1|contrast"
python -m models.train -c mnist_contrast --best_loss_term "tot" --model_config__loss_config__funcs "ddc_2|contrast"
python -m models.train -c mnist_contrast --best_loss_term "tot" --model_config__loss_config__funcs "ddc_3|contrast"
python -m models.train -c mnist_contrast --best_loss_term "tot" --model_config__loss_config__funcs "ddc_1|ddc_2|contrast"
python -m models.train -c mnist_contrast --best_loss_term "tot" --model_config__loss_config__funcs "ddc_1|ddc_3|contrast"
python -m models.train -c mnist_contrast --best_loss_term "tot" --model_config__loss_config__funcs "ddc_2|ddc_3|contrast"
python -m models.train -c mnist_contrast --best_loss_term "tot" # Uses all losses by default
