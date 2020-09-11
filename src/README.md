### Training Notes

- Don't change default_settings.yaml

- To modify settings for an experiment, create your own settings.yaml file. If you append it to the -es flag, it will override the default settings.

- All configs that are modifiable are located in src/configs or you can do `python3 src/external/spinup_main.py --help` to see all the configs.

- View logs on tensorboard (make sure you first ssh with port forwarding)
    - `tensorboard --logdir={log_dir} --port {server_port} --host localhost`