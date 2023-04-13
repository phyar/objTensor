# objTensor
* objTensor is a very lightweight header-only deep learning library written in C++. It natively supports fully connected layers, convolutional layers, LSTM and GRU layers, as well as commonly used activation functions. It is easy to use, has automatic gradient capabilities, and is wrapped with smart pointers. The model building process is similar to PyTorch.
  #### Simple computation with objTensor:
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
  #### Simple computation with PyTorch:
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

  #### Build a model with objTensor:
  ```C++
  struct ModelLSTM : public Module
  {
      LSTM lstm;
      Linear ln;

      ModelLSTM()
      {
          lstm = LSTM(128, 2);
          ln = Linear(10);
      }

      objTensor forward(const objTensor& x)
      {
          objTensor out = x;
          out = lstm.forward(out);
          out = ln.forward(out).softmax();
          return out;
      }
  };
  ```
  #### Build a model with PyTorch:
  ```python
  class LSTM(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers, num_classes):
          super(LSTM, self).__init__()
          self.hidden_size = hidden_size
          self.num_layers = num_layers
          self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
          self.fc = nn.Linear(hidden_size, num_classes)

      def forward(self, x):
          h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
          c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

          out, _ = self.lstm(x, (h0, c0))

          out = self.fc(out[:, -1, :])

          return out
  ```

* objTensor has advanced memory management and caching strategies that minimize memory operations during the training process, making its performance exceed that of common deep learning libraries when using CPU. Below is a performance test of training a simple LSTM network using the MNIST database:
  #### Time required per epoch
  ||objTensor|PyTorch|
  |---|---|---|
  |Intel i5 3.0G (6 cores)|2.3|9.8|
  |Apple M1 Pro (8 cores)|1.0|4.0|

* objTensor relies on third-party BLAS computing libraries. On macOS, the Accelerate framework can be used. On Linux or Windows, OpenBLAS needs to be installed and imported during compilation. The project's source code only describes how to compile and test files on macOS.
