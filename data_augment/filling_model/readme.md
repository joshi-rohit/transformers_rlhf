## Filling Model with T5

This project is used to restore the [MASK] position in the sentence through a generative model to achieve `Mask Then Filling` in UIE information extraction [data enhancement strategy] (../../UIE/readme.md).

[Mask Then Fill](https://arxiv.org/pdf/2301.02427.pdf) It is a data augmentation strategy for information extraction based on generative models. For a piece of text, we divide it into "key information segment" and "non-key information segment", and the segment containing keywords is called "key information segment". In the following example, the bold ones are `Key Information Fragments`, and the rest are `Non-Key Fragments`。

> **New Year's Eve** I flew back to **Chengdu** from **Beijing** Daxing Airport **.

We randomly [MASK] live a part of the "non-key fragment" so that it becomes: 

> **New Year’s Eve** I flew back to **Chengdu** from **Beijing** [MASK] **.

Then, feed the modified sentence to the filling model (T5-Fine Tuned) to restore the sentence and get the newly generated sentence:

> **New Year’s Eve** I started from **Beijing** Capital Airport, and **flew back** to **Chengdu**.

## 1. Environment installation

This project is implemented based on `pytorch` + `transformers`, please install related dependencies before running:

```sh
pip install -r ../requirements.txt
```

## 2. Dataset preparation

A part of the sample data is provided in the project, the data comes from the text data in the DuIE dataset, and the data is in `data/` 。

If you want to use `custom data` for training, you only need to construct the text with [MASK] according to the example data. You can also use `parse_data.py` to quickly generate training data based on `word granularity`:

```tsv
"Bortolaso Guillaume,法国籍[MASK]"中[MASK]位置的文本是：	运动员
"Bortolaso Guillaume, French Nationality [MASK]". The text in position [MASK] is:   Athlete
"歌曲[MASK]是由歌手海生演唱的一首歌曲"中[MASK]位置的文本是：	《情一动心就痛》
"The song [MASK] is a song sung by the singer Haisheng". The text at [MASK] is:   "Love Hurts When You Touch It"
...
```

Each line is separated by `\t` delimiter, the first part is `text with [MASK]`, and the second part is the original text (label) at `[MASK] position`. 



## 3. Model training

Modify the corresponding parameters in the training script `train.sh` to start model training:

```sh
python train.py \
    --pretrained_model "uer/t5-base-chinese-cluecorpussmall" \
    --save_dir "checkpoints/t5" \
    --train_path "data/train.tsv" \
    --dev_path "data/dev.tsv" \
    --img_log_dir "logs" \
    --img_log_name "T5-Base-Chinese" \
    --batch_size 128 \
    --max_source_seq_len 128 \
    --max_target_seq_len 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --logging_steps 50 \
    --valid_steps 500 \
    --device cuda:0
```

After the training is started correctly, the terminal will print the following information:

```python
...
 0%|          | 0/2 [00:00<?, ?it/s]
100%|██████████| 2/2 [00:00<00:00, 21.28it/s]
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 350134
    })
    dev: Dataset({
        features: ['text'],
        num_rows: 38904
    })
})
...
global step 2400, epoch: 1, loss: 7.44746, speed: 0.82 step/s
global step 2450, epoch: 1, loss: 7.42028, speed: 0.82 step/s
global step 2500, epoch: 1, loss: 7.39333, speed: 0.82 step/s
Evaluation bleu4: 0.00578
best BLEU-4 performence has been updated: 0.00026 --> 0.00578
global step 2550, epoch: 1, loss: 7.36620, speed: 0.81 step/s
...
```

The training curve will be saved in `logs/T5-Base-Chinese.png` file:

<img src='assets/T5-Base-Chinese.png'></img>

## 4. Model prediction

After completing the model training, run `inference.py` to load the trained model and apply:

```python
 if __name__ == "__main__":
    masked_texts = [
        '"《μVision2单片机应用程序开发指南》是2005年2月[MASK]图书，作者是李宇"中[MASK]位置的文本是：'
    ]
    inference(masked_texts)
```

```sh
python inference.py
```

The following inference results are obtained:

```python
maksed text: 
[
    '"μVision2 Microcontroller Application Development Guide" is a book [MASK] in February 2005 with Li Yu as the author". The text in the [MASK] position is:'
]
output: 
[
    'published by China Industry Press'
]
```
