import torch
import torch.nn.functional as F
from torch import optim
from tqdm.auto import tqdm

from data_utils import get_mnist_train_test
from neural_network_utils import StudentNet, TeacherNet, kd_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KnowledgeDistillation:
    def __init__(self, teacher, student, train_loader, test_loader, T=3, alpha=1):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.T = T
        self.alpha = alpha

        self._student_optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
    
    def _train(self):
        self.teacher.eval()
        self.student.train()

        for i, (x, y) in enumerate(tqdm(self.train_loader)):
            self._student_optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            teacher_logits = self.teacher(x)
            student_logits = self.student(x)

            loss = kd_loss(student_logits, teacher_logits, y, self.T, self.alpha)
            # loss = F.cross_entropy(student_outputs, y)
            loss.backward()

            self._student_optimizer.step()
    
    def _test(self):
        self.student.eval()

        correct = 0
        total = 0
        for i, (x, y) in enumerate(tqdm(self.test_loader)):
            x = x.to(device)
            y = y.to(device)

            predictions = self.student(x)
            max_indices = torch.max(predictions, dim=1)[1]

            correct += (max_indices == y).sum().item()
            total += y.size(0)
        
        acc = correct / total
        print(f"Test accuracy: {round(acc * 100, 1)}%")
    
    def distill(self, epochs):
        for epoch in range(epochs):
            self._train()
            self._test()


def main():
    teacher = TeacherNet()
    teacher.load_state_dict(torch.load(".models/teacher.pth"))

    student = StudentNet()

    train_loader, test_loader = get_mnist_train_test()

    kd = KnowledgeDistillation(teacher, student, train_loader, test_loader)

    kd.distill(epochs=1)


if __name__ == "__main__":
    main()