# Efficient Indo-sentiment with Few Shot Learning and Distillation
Indonesia sentiment analysis using [SetFit](https://github.com/huggingface/setfit)([paper](https://arxiv.org/abs/2209.11055)) model for Few shot learning, trained with [Indonesia google play review](https://huggingface.co/datasets/jakartaresearch/google-play-review) dataset.
This project contains code and data for training and evaluating machine learning models for natural language processing (NLP) tasks, as well as code for deploying the trained models for inference.



## Dependencies

- Python 3.7 or higher
- NumPy 1.20 or higher
- Pandas 0.25 or higher
- scikit-learn 0.22 or higher
- PyTorch 1.7 or higher (if using the PyTorch models)

## Setup

1. Clone this repository:
```
https://github.com/randypangestu/efficient-Indo-sentiment.git
```


2. Install the dependencies:
```
pip install -r requirements.txt
```


## Usage

To train the model, run:

```
#default examples
python3 train_with_distill.py
```

To run inference on new data, use the `infer.py` script:
```
#default inference
python sentiment_prediction.py
```

## Results

The results of the training and evaluation are stored in the `results` directory.

# To Do
~~1. Create config for training~~
2. Test on local machine
3. Create setup.py
4. Create Dockerfile
5. Create huggingface endpoint
6. Create collab & gradio inference

# References
- Efficient Few-Shot Learning Without Prompts [paper](https://arxiv.org/abs/2209.11055), [github](https://github.com/huggingface/setfit)