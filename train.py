import torch
from model.student import Student
from model.teacher1 import Classifier
from model.teacher2 import CBGRU
from trainer import training
from load import load_data


files = ['./dataset/re_data_word2vec_fragment_vectors.pkl',
         './dataset/re_data_FastText_fragment_vectors.pkl',
         './dataset/re_data.jsonl']
loader_t, loader_v, _ = load_data(files)

model = Student(300, 100, 300)
# model = GCN(100, 300, 1)
teacher_g = Classifier(
    classNum=1,
    dropout_rate=0.5,
    nfeat=100,
    nhid=11,
    nclass=1,
    n_layers=9,
    k=200,
    head=2,
    features_length=100
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
student = Student(300, 100, 300)
student.load_state_dict(torch.load('./model/re/student.pth'))
teacher_g.load_state_dict(torch.load('./model/re/teacher_g.pth'))
teacher_s = CBGRU(300, 100, 1)
teacher_s.load_state_dict(torch.load('./model/re/teacher_s.pth'))

training(loader_t, loader_v, 500, teacher_g, teacher_s, student)




