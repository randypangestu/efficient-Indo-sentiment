import datasets
from setfit import SetFitModel


def result_postprocessing(result_proba):
    label_keys = {'1':'positive',
                  '0':'negative',
                  '2':'neutral'}
    negative_value = result_proba[0,0]
    positive_value = result_proba[0,1]
    if abs(negative_value - positive_value) > 0.2:
        class_predict = 2
    else:
        class_predict = result_proba.argmax()

    return label_keys[str(class_predict)]


if __name__ == '__main__':
    model = SetFitModel.from_pretrained(
        "randypang/indo-review-sentiment-minilm3"
    )
    while True:
        text = input()
        result = model.predict_proba([text])
        postproc_result = result_postprocessing(result)
        print('text {} is: a {} review'.format(text, postproc_result))