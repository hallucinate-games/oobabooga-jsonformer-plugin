# oobabooga-jsonformer-plugin

This is a plugin for the [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

This plugin forces models to output valid JSON of a specified schema using [JSONFormer](https://github.com/1rgs/jsonformer)

Install by cloning this repo into the `extensions` directory of `text-generation-webui` and then installing the dependencies into same conda environment that your `text-generation-webui` runs in. e.g.
```shell
$ conda activate textgen
$ cd text-generation-webui/extensions
$ git clone https://github.com/hallucinate-games/oobabooga-jsonformer-plugin.git jsonformer
$ cd jsonformer
$ pip install -r requirements.txt
```
Then restart the server with `--extensions jsonformer` or enable it via the UI.
