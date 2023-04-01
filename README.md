# objTensor
objTensor is a very lightweight header-only deep learning library written in C++. It natively supports fully connected layers, convolutional layers, LSTM and GRU layers, as well as commonly used activation functions. It is easy to use, has automatic gradient capabilities, and is wrapped with smart pointers. The model building process is similar to PyTorch.
#### simple computation with objTensor:
```C++
objTensor data = objTensor({0.5, 0.2}); // data tensor
objTensor w = objTensor({1, 2, 3, 4}).reshape({2,2}).rg(); // weight tensor
objTensor m = data % w; // matrix multiply
objTensor r = m.relu().softmax(); // apply relu and softmax
r.print("r");
objTensor label = objTensor({1, 0}); // label tensor
objTensor d = (r - label);
d = (d * d).mean(); // delta value
d.backward(); // apply backward
d.print("d");
w.d().grad.print("grad of a"); // now we get grad of weight tensor
```
#### simple computation with pytorch:
```python
data = torch.tensor([0.5, 0.2])
w = torch.tensor([[1.0,2],[3,4]], requires_grad=True)
m = torch.matmul(data, w)
r = torch.nn.functional.softmax(torch.relu(m))
print(r)
label = torch.tensor([1, 0.0])
d = r - label
d = torch.mean(torch.mul(d, d))
d.backward()
print(d)
print(w.grad)
```
objTensor has advanced memory management and caching strategies that minimize memory operations during the training process, making its performance exceed that of common deep learning libraries when using CPU. Below is a performance test of training a simple LSTM network using the MNIST database:

objTensor relies on third-party BLAS computing libraries. On macOS, the Accelerate framework can be used. On Linux or Windows, OpenBLAS needs to be installed and imported during compilation. The project's source code only describes how to compile and test files on macOS.
