import torch
from torch import tensor


def run():
    print(torch.cuda.get_device_name(0))
    x = tensor([[1.0], [2.0], [3.0]],
               requires_grad=True)
    A = tensor([[1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]], requires_grad=True)
    print(x.shape)
    print(A.shape)
    out = torch.matmul(A, x)
    print(out)
    out = out.sum()
    out.backward()
    print(x.grad)


if __name__ == '__main__':
    run()
