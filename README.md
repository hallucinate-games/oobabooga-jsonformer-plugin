# DEPRECATED: Use a model that supports function calling instead

# oobabooga-jsonformer-plugin

This is a plugin for the [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

This plugin forces models to output valid JSON of a specified schema. Most of the code was strongly inspired by [JSONFormer](https://github.com/1rgs/jsonformer) but adapted for use with oobabooga. Includes some additional tweaks to better handle the weird variety of output that can come from smaller models.

You can install in via the `text-generation-webui` using the github link to this repo or if you prefer to install it manually:

Install by cloning this repo into the `extensions` directory of `text-generation-webui` e.g.
```shell
$ cd text-generation-webui/extensions
$ git clone https://github.com/hallucinate-games/oobabooga-jsonformer-plugin.git jsonformer
```
Then restart the server with `--extensions jsonformer` or enable it via the UI.
