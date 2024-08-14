import torch
from model.student import Student
from load import load_data
from tester import test


files = ['./dataset/re_data_word2vec_fragment_vectors.pk',
         './dataset/re_data_FastText_fragment_vectors.pkl',
         './dataset/re_data.jsonl']
_, loader_t, _ = load_data(files)

student = Student(300, 100, 300)
student.load_state_dict(torch.load('./model/re/student.pth'))


test(student, loader_t)




