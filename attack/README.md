# Prepare synthetic and natural noise data

## Prepare synthetic noise data

1. Prepare raw data

You need to create a folder to store the raw data in the `data` files directory under the `llm_robust_eval` directory. The folder needs to contain two subfolders, `dev` and `test`. the `dev` folder is used to store the in-context examples used for sampling, and the `test` folder is used to store the test data. The files in created folder are organized in the following format.

```
.
└── WMT-News-fr-en
    ├── dev
    │   ├── clean.en
    │   └── clean.fr
    └── test
        ├── clean.en
        └── clean.fr
```

2. Generate character-level noise data through character-level attacks

Go to the `script` directory and run the following bash file:

```
bash character-attack.sh
```

- `--data_path` parameter is the path to the raw data.

In particular, you need to specify the path to the raw data. If you want to change the source and target languages, you also need to change the `cur_src_file_path`, `new_src_file_path`, `cur_tgt_file_path`, `new_tgt_file_path` path in the `character-attack.py` file. The same goes for the other files.

Additionally, if you want to attack Chinese text through character-level attacks, you can run the following file.

```
bash character-attack-zh.sh
```

3. Generate word-level noise data through word-level attacks

Run the following bash file:

```
bash word-attack.sh
```

- `--data_path` parameter is the path to the raw data.
- `--model_path` parameter is the cc.300.vec model downloaded from https://fasttext.cc/docs/en/crawl-vectors.html.

Additionally, if you want to attack Chinese text through word-level attacks, you can run the following file.

```
bash word-attack-zh.sh
```

4. Generate multi-level noise data through multi-level attacks

Run the following bash file:

```
bash multi-attack.sh
```

- `--data_path` parameter is the path to the raw data.
- `--model_path` parameter is the cc.300.vec model.

Additionally, if you want to attack Chinese text through multi-level attacks, you can run the following file.

```
bash multi-attack-zh.sh
```

## Prepare natural noise data

1. The natural noise data has been organized in the `natural` folder directory, which contains two linguistic directions, id-zh and zh-en, respectively.

2. Each translation direction contains ten types of natural noise data. Each noise has two files, one for the source language and one for the target language.