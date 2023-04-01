
#include "src/objTensor.h"

using namespace std;
using namespace objt;

string dataDir = "./DataSets";

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

void modelTest()
{
    objTensor xt, yt;
    objTensor xt_test, yt_test;
    objTensor xt_valid, yt_valid;
    
    DataSets::read_mnist(xt, yt, xt_test, yt_test, dataDir + "/MNIST_data/");
    //DataSets::read_cifar10(xt, yt, xt_test, yt_test, dataDir + "/cifar-10-batches-bin/");

    TimeServer ts;
    ts.Set();

    //ConvNet model;
    ModelLSTM model;
    auto fStep = [&](Module &m, int e, int step)
    {
        if((step+1)%10==0)
        {
            cout<<"=";
            cout.flush();
        }
    };
    auto fEpoch = [&](Module &m, int e)
    {   
        cout<<endl<<"Epoch: "<<e<<" insNum: "<<objTensor::insNum()<<" paras: "<<m.opt.parameters.size()<<", "<<m.opt.numScalars()<<"  ";
        cout<<ts.GetElapsed()<<"ms"<<endl;
        cout<<"d: "<<m.d_sum<<"  acc: "<<m.acc;
        //d.print();
        
        Scalar netAcc = m.test(xt_test, yt_test);
        cout<<"  testAcc: "<<netAcc<<"  "<<endl;
        
        ts.GetElapsed();
    };
    
    model.fit(xt, yt, 200, 0.1/2, fStep, fEpoch, 100, 4);

}
int main()
{
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
    
    modelTest();
    
    return 0;
}
