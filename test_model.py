from model import MNISTModel
from train import train_model
import torch
import pytest
from model import MNISTModel
import torch.nn.functional as F

def test_parameter_count():
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_accuracy():
    accuracy = train_model()
    assert accuracy >= 95.0, f"Model accuracy is {accuracy}%, should be at least 95%"

def test_model_output_shape():
    model = MNISTModel()
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected output shape {(batch_size, 10)}, got {output.shape}"

def test_model_forward_pass():
    model = MNISTModel()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert torch.allclose(torch.exp(output).sum(), torch.tensor(1.0), atol=1e-5), "Softmax outputs don't sum to 1"

def test_model_gradients():
    model = MNISTModel()
    x = torch.randn(1, 1, 28, 28, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN values"


# test_parameter_count()