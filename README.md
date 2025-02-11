# Chinese_Sentiment

Chinese Sentiment Analysis Model (Three-class Classification)

---

## Table of Contents
1. [Model Purpose](#model-purpose)
2. [References](#references)
3. [Training Dataset](#training-dataset)
4. [Training Process](#training-process)
5. [Results](#results)
6. [General usage](#general-usage)
7. [Simple docker usage](#simple-docker-usage)
8. [Todo](#todo)
9. [Citation](#citation)

---

## Model Purpose

- **Identify the sentiment of social media comments.**  

Example: 
  
| Comments | Labels |
|------------------|-----------------|
| 這個食物很好吃    | positive    |
| 這個食物很難吃   | negative    |
| 這個食物味道普通   | neutral    |

---

## References

- [Bert](https://github.com/google-research/bert)  
- Base model : [RoBERTa-wwm-ext, Chinese](https://drive.google.com/open?id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt)
- Framework : TensorFlow  

---

## Training Dataset

**test & train dataset source :** 

Crawl and extract comment data, then clean the data before feeding it into Sentiment Analysis using Azure AI services (testing has shown Azure to be the most accurate).

**val dataset source :** 

The validation set is manually evaluated to determine the most accurate sentiment of the comments.

**Annotation format :** csv  

| test | train | val |
|------------------|-----------------|-----------------|
| 91393    | 463708    | 1496    |

Data example: 

食物不好吃,negative

食物很好吃,positive

食物味道普通,neutral

---

## Training Process

Environment : [kaggle platform](https://www.kaggle.com/)

1. Download base model : [RoBERTa-wwm-ext, Chinese](https://drive.google.com/open?id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt)
2. Use the Kaggle platform for training (*if the dataset is too large, it may exceed Kaggle's maximum execution time limit, making training impossible).  

3. Place the `bert-code-new` folder, base model, and `sentiment-data` folder into the Kaggle dataset.

4. Use the following code

```bash
!python /kaggle/input/bert-code-new/run_classifier.py --task_name=mytask --do_train=true --do_eval=true --data_dir=/kaggle/input/sentiment-data --vocab_file=/kaggle/input/chinese-roberta-wwm-ext-l-12-h-768-a-12/vocab.txt --bert_config_file=/kaggle/input/chinese-roberta-wwm-ext-l-12-h-768-a-12/bert_config.json --init_checkpoint=/kaggle/input/chinese-roberta-wwm-ext-l-12-h-768-a-12/bert_model.ckpt --max_seq_length=300 --train_batch_size=16 --learning_rate=1e-5 --num_train_epochs=2.0 --output_dir=output
```
5. **Hardware and environment (kaggle):**
  - Training time : 11 hours  
  - Hardware : GPU P100  
  - Environment : Python 3.10.12
 
---

## Results

- **Model performance:**  
  - eval_accuracy : 0.796531
  - global_step = 59292

---

## General usage
[Trained model](https://huggingface.co/stevenhsu123/sentiment_test)

- Environment (local) : Python 3.8.16, macOS 14.6.1

1. **Install dependencies:**  
   ```bash
   pip install -r ./bert-code-new/requirements.txt
   ```
2. **Run inference:**  
   ```bash
   python ./bert-code-new/run_inference.py --task_name=mytask /
   --do_predict=true /
   --data_dir=test-for-inference /
   --vocab_file=chinese-roberta-wwm-ext-l12-h768-a12/vocab.txt /
   --bert_config_file=chinese-roberta-wwm-ext-l12-h768-a12/bert_config.json /
   --init_checkpoint=chinese-roberta-wwm-ext-l12-h768-a12 /
   --max_seq_length=512 /
   --output_dir=output_predict
   ```
3. **Example output:**  
   - output format : tsv 
   - Example output : 0.04118731(negative)  0.9210373(neutral)  0.037775367(positive)
 
## Simple docker usage

1. **Install docker images:**  
   ```bash
   ./build_images.sh
   ```
2. **Run inference:**  
   ```bash
   ./run_predict.sh
   ```
3. **Example output:**  
   - outputfile : output_predict
   - output format : tsv 
   - Example output : 0.04118731  0.9210373  0.037775367

## Todo
- fastapi & docker compose

## Citation
```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```
