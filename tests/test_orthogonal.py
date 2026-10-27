import unittest
import torch
from torch import nn
from task2.orthogonal import orthogonal_loss, cosine_matrix

class FakePrompt(nn.Module):
    e_layers=[0]
    num_heads=1
    num_experts=3
    def __init__(self, vectors):
        super().__init__()
        for e, v in enumerate(vectors):
            self.register_parameter(f'e_pk_0_{e}_0',nn.Parameter(torch.tensor(v,dtype=torch.float32).reshape(1,-1)))

class TestOrthogonal(unittest.TestCase):
    def test_orthogonal_vectors_zero(self):
        p=FakePrompt([[1,0,0],[0,1,0],[0,0,1]])
        self.assertAlmostEqual(orthogonal_loss(p).item(),0.0,places=6)
    def test_collinear_vectors_penalized(self):
        p=FakePrompt([[1,0,0],[-1,0,0],[1,0,0]])
        self.assertAlmostEqual(orthogonal_loss(p).item(),1.0,places=6)
    def test_backward(self):
        p=FakePrompt([[1,1,0],[1,0,1],[0,1,1]])
        orthogonal_loss(p).backward()
        self.assertTrue(all(param.grad is not None and torch.isfinite(param.grad).all() for param in p.parameters()))
    def test_matrix(self):
        mat=cosine_matrix(torch.eye(3))
        self.assertTrue(torch.allclose(mat,torch.eye(3)))

if __name__=='__main__': unittest.main()
