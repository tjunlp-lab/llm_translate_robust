# Getting Started

## Prepare synthetic and natural noise data

- See the `README` file in the `attack` folder for details on how to run the code.

## Calculate sample similarity

1. Calculate the embedding of all sentences and save them. 

- First, you need to go into the `data` folder you created to store the data, and create the `dev_emb` and `test_emb` folders. The files in created folder are organized in the following format.

```
├── dev
│   ├── character.zh
│   ├── clean.zh
│   ├── clean.en
│   ├── multi.zh
│   └── word.zh
├── dev_emb
├── test
│   ├── character.zh
│   ├── clean.zh
│   ├── clean.en
│   ├── multi.zh
│   └── word.zh
└── test_emb
```

- Then, run the following command to calculate the embedding of texts.

```python
bash cal_emb.sh
```

2. Calculate the similarity of the samples to be tested in the entire dataset and select top-5 to be left for in-context demonstrations selection.

```python
bash cal_sim.sh
```

## LLMs translation using in-context learning

- Conduct experiments using different noise type data as well as experimental setups with different sampling methods.

```python
bash main.sh
```
