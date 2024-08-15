import torch
from model.student import Student
from load import load_data
from tester import test


files = ['./dataset/ar_data_word2vec_fragment_vectors.pkl',
         './dataset/ar_data_FastText_fragment_vectors.pkl',
         './dataset/ar_data.jsonl']
_, loader_t, _ = load_data(files)

student = Student(300, 100, 300)
student.load_state_dict(torch.load('./model/ar/student.pth'))


test(student, loader_t)




