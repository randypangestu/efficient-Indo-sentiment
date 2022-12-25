from datasets import load_dataset
from setfit import sample_dataset
from setfit import SetFitModel, SetFitTrainer, DistillationSetFitTrainer

from lib.utils import PerformanceBenchmark, plot_metrics

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

def train_distill(teacher_model, distill_dataset, student_model='"sentence-transformers/all-MiniLM-L6-v2"', model_name='MiniLM-L6 (distilled)' ):

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
        student_trainer.student_model, test_dataset, model_name"
    )
    return student_trainer, pb

def dataset_preprocess(dataset_name='jakartaresearch/google-play-review'):
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


if __name__ == '__main__':
    
    dataset_dict = dataset_preprocess(dataset_name)
    # Define the test set for evaluation
    train_dataset_few_shot = dataset_dict['few_shot']
    train_dataset_distill = dataset_dict['distill']
    test_dataset = dataset_dict['test']
    teacher_trainer, pb_teacher = train_teacher(dataset=train_dataset_few_shot, pretrained="firqaaa/indo-sentence-bert-base")
    perf_metrics = pb_teacher.run_benchmark()
    mini_model_trainer, pb_teacher_mini = train_teacher(dataset=train_dataset_few_shot, pretrained="sentence-transformers/paraphrase-MiniLM-L3-v2")
    perf_metrics.update(pb_teacher_mini.run_benchmark())
    plot_metrics(perf_metrics, "MiniLM-L3-v2 1 epoch")
    distill_name = 'MiniLM-L3 (distilled)'
    student_model_trainer, pb_student = train_distill(distill_dataset=train_dataset_distill,
                                                      student_model='sentence-transformers/paraphrase-MiniLM-L3-v2',
                                                      model_name=distill_name)
    plot_metrics(perf_metrics, distill_name)

