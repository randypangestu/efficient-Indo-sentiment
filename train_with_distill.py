from datasets import load_dataset
from setfit import sample_dataset
from setfit import SetFitModel, SetFitTrainer, DistillationSetFitTrainer

import argparse
from lib.utils import PerformanceBenchmark, plot_metrics
import logger
import yaml


def train_teacher(dataset, pretrained="firqaaa/indo-sentence-bert-base"):

    # Load pretrained model from the Hub
    teacher_model = SetFitModel.from_pretrained(pretrained)
    # Create trainer
    teacher_trainer = SetFitTrainer(
        model=teacher_model, train_dataset=dataset,
        num_epochs= 10
    )
    # Train!
    teacher_trainer.train()
    pb = PerformanceBenchmark(model=teacher_trainer.model, dataset=test_dataset, optim_type="indo sentence bert (teacher) 4 epoch")
    return teacher_trainer, pb


def train_distill(teacher_model, distill_dataset, student_model='sentence-transformers/paraphrase-MiniLM-L3-v2', model_name='distilled model' ):


    student_model = SetFitModel.from_pretrained(
        student_model
    )
    student_trainer = DistillationSetFitTrainer(
        teacher_model=teacher_model,
        train_dataset=distill_dataset,
        student_model=student_model,
    )
    student_trainer.train()

    pb = PerformanceBenchmark(

        student_trainer.student_model, test_dataset, model_name
    )
    return student_trainer, pb

def dataset_preparation(dataset_name='jakartaresearch/google-play-review'):
    dataset = load_dataset(dataset_name)
    # Create 2 splits: one for few-shot training, the other for knowledge distillation
    train_dataset = dataset["train"].train_test_split(seed=42)
    train_dataset_few_shot = sample_dataset(train_dataset["train"], num_samples=8)
    train_dataset_distill = train_dataset["test"].select(range(1000))
    test_dataset = dataset["test"]
    dataset_dict = {'few_shot':train_dataset_few_shot,
                    'distill':train_dataset_distill,
                    'test': test_dataset}
    return dataset_dict



def train_and_distill(dataset_dict, model_config):
    # prepare dataset for training
    train_dataset_few_shot = dataset_dict['few_shot']
    train_dataset_distill = dataset_dict['distill']
    test_dataset = dataset_dict['test'] # currently not used
    
    
    logger.info('TRAIN TEACHER MODEL')
    teacher_trainer, pb_teacher = train_teacher(dataset=train_dataset_few_shot, pretrained=model_config['train_params']['teacher_model'])
    logger.info('Finished training teacher')

    logger.info('TRAIN STUDENT MODEL WITH DISTILLATION')
    student_model_trainer, pb_student = train_distill(distill_dataset=train_dataset_distill,
                                                      student_model=model_config['train_params']['student_model'])
    logger.info('Finished training student model')
    
    return student_model_trainer 


def read_yaml_config(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def get_args():
    # create the argument parser
    parser = argparse.ArgumentParser()

    # add the arguments
    parser.add_argument("-c", "--config", , required=True, type=str, help="config file path")
    parser.add_argument("-s", "--save_path", required=True, type=str, help="number of items to process")
    
    # parse the arguments
    args = parser.parse_args()

    # print the arguments
    return args

if __name__ == '__main__':
    args = get_args()
    config = open_config(args.config)
    dataset_dict = dataset_preparation(config['train_params']['dataset'])
    student_model_trainer = train_and_distill(dataset_dict, config)
    student_model_trainer.student_model._save_pretrained(args.save_path)
   