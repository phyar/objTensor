//
//  objTensor.h
//  objTensor
//
//  Created by Phyar on 2019/8/31.
//  Copyright © 2019 Phyar. All rights reserved.
//

#ifndef objTensor_hpp
#define objTensor_hpp

//#define EIGEN_DEFAULT_TO_ROW_MAJOR
////#define EIGEN_DONT_VECTORIZE
//#include "Eigen/Dense"

//#define USE_DOUBLE

#ifdef USE_DOUBLE
#define Scalar double
#define GEMM cblas_dgemm
#else
#define Scalar float
#define GEMM cblas_sgemm
#endif


#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include "cblas.h"
#endif

//#include <stdio.h>
#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <random>
#include "Parallel.h"
#include "Util.h"

#ifndef NDEBUG
#   define ASSERT_M(condition, message) \
do { \
if (! (condition)) { \
std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
<< " line " << __LINE__ << ": " << message << std::endl; \
std::terminate(); \
} \
} while (false)
#   define ASSERT_F(condition, message) \
do { \
if (! (condition)) { \
std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
<< " line " << __LINE__ << ": " << std::endl; \
message();\
std::terminate(); \
} \
} while (false)
#   define ASSERT(condition) ASSERT_M(condition, "")
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

namespace objt{
    
using namespace std;



struct TShape : public vector<int>
{
    vector<int> strides;
    
//    template <typename T> void setV(T t){ push_back((int)t); }
//    template<typename T, typename... Args> void setV(T t, Args... args) // recursive variadic function
//    { push_back(t); setV(args...); }
//
//    template<typename... Args> TensorShape(Args... args)
//    {
//        setV(args...);
//        updateStrides();
//    }
    
    TShape() {}
//    TShape(const initializer_list<int> &v) : vector<int>(v)
//    {}
//    TShape(int v) { push_back(v); }
    TShape(int v1, int v2) { push_back(v1); push_back(v2); }
    TShape(int v1, int v2, int v3) { push_back(v1); push_back(v2); push_back(v3); }
    TShape(int v1, int v2, int v3, int v4) { push_back(v1); push_back(v2); push_back(v3); push_back(v4); }
    
    bool operator==(const TShape &shape)
    {
        int maxSize = (int)max(size(), shape.size());
        TShape shapeA = extend(maxSize);
        TShape shapeB = shape.extend(maxSize);
        for(int i=0; i<maxSize; i++)
        {
            if(shapeA[i] != shapeB[i]) return false;
        }
        return true;
    }
    int& operator[](int i)
    {
        if(i<0) i += size();
        return vector<int>::operator[](i);
    }
    const int& operator[](int i) const
    {
        if(i<0) i += size();
        return vector<int>::operator[](i);
    }
    void updateStrides()
    {
        strides.resize(size());
        strides.back()=1;
        for(int i=(int)size()-2; i>=0; i--)
        {
            strides[i] = strides[i+1] * (*this)[i+1];
        }
    }
    void getCoor(vector<int> &coor, int index) const
    {
        coor.resize(size());
        for(int i=0; i<strides.size(); i++)
        {
            coor[i] = index/strides[i];
            index -= coor[i]*strides[i];
        }
    }
    int getIndex(vector<int> &coor) const
    {
        int index=0;
        for(int i=0; i<strides.size(); i++)
        {
            index += coor[i]*strides[i];
        }
        return index;
    }
    
    int flatSize(int axis=0) const
    {
        if(this->empty()) return 0;
        int size=1;
        for(int i=axis; i<this->size(); i++)
        {
            size *= (*this)[i];
        }
        return size;
    }
    void print() const
    {
        for(int i=0; i<this->size(); i++)
        {
            cout<<(*this)[i];
            if(i<this->size()-1) cout<<"x";
        }
        cout<<endl;
    }
    friend ostream & operator<<(ostream &os, const TShape &shape)
    {
        for(int i=0; i<shape.size(); i++)
        {
            os<<shape[i];
            if(i<shape.size()-1) cout<<"x";
        }
        os<<endl;
        return os;
    }
    TShape extend(int len) const
    {
        TShape shape = *this;
        while(shape.size()<len)
        {
            shape.insert(shape.begin(), 1);
        }
        shape.updateStrides();
        return shape;
    }
    TShape simplify() const
    {
        TShape shape = *this;
        if(shape.size() == 0) return shape;
        while(shape[0] == 1 && shape.size()>1)
        {
            shape.erase(shape.begin());
        }
        shape.updateStrides();
        return shape;
    }
    TShape subShape(int axis, int n) const
    {
        TShape shape = *this;
        shape[axis]=n;
        shape.updateStrides();
        return shape;
    }
};

enum objtConst
{
    AXIS_NONE = 9999999u,
};

#define Self() ObjTensor& self = *this
#define ConstSelf() const ObjTensor& self = *this
template <class T>
struct ObjTensor : public shared_ptr<T> //ptrTensor//
{
    typedef shared_ptr<T> base;
    ObjTensor():shared_ptr<T>(new T())
    {
    }
    ObjTensor(T* p):shared_ptr<T>(p)
    {
        //cout<<"init"<<endl;
    }
    ObjTensor(const vector<Scalar> &vec):shared_ptr<T>(new T())
    {
        fill(TShape((int)vec.size(), 1));
        setData(vec);
    }
    ObjTensor(Scalar s):shared_ptr<T>(new T())
    {
        Self();
        fill(TShape(1, 1));
        self[0] = s;
    }
//    ObjTensor(const TensorShape &shape, const vector<Scalar> &vec):shared_ptr<T>(new T())
//    {
//        d().data=vec;
//        d().shape=shape;
//    }
//    ObjTensor(const TensorShape &shape, Scalar v):shared_ptr<T>(new T())
//    {
//        fill(shape, v);
//    }
//    ObjTensor(const TensorShape &shape):shared_ptr<T>(new T())
//    {
//        fill(shape);
//    }
    ~ObjTensor()
    {
        
    }

    void setShape(const TShape &_shape)
    {
        d().shape = _shape;
        d().shape.updateStrides();
    }
    ObjTensor& fill(const TShape &_shape)
    {
        setShape(_shape);
        resize(d().shape.flatSize());
        return *this;
    }
    ObjTensor& fill(const TShape &_shape, Scalar v)
    {
        Self();
        fill(_shape);
        for(int i=0; i<self.size(); i++) self[i] = v;
        return *this;
    }
    ObjTensor& fillValue(Scalar v)
    {
        Self();
        for(int i=0; i<self.size(); i++) self[i] = v;
        return *this;
    }
    
    ObjTensor toOneHot(int n)
    {
        ConstSelf();
        ObjTensor out;
        TShape shape = TShape(d().shape.flatSize(), n);
        out.fill(shape);
        for(int row = 0; row<shape[0]; row++)
        {
            for(int i=0; i<shape[1]; i++)
            {
                out[row*n+i] = (i==(int)self[row]) ? 1.0 : 0.0;
            }
        }
        return out;
    }
    static uint32_t myRand()
    {
        static std::mutex mutex;
        std::unique_lock<std::mutex> locker(mutex);
        static std::mt19937 rnd((unsigned int)time(0));
        return rnd();
    }
    static vector<int> randSeq(int n)
    {
        vector<int> seq;
        seq.resize(n);
        for(int i=0; i<n; i++) seq[i] = i;
        
        auto swap = [](int &a, int &b)
        {
            int tmp = a;
            a = b;
            b = tmp;
        };
        for(int i=n-1; i>0; i--)
        {
            int j = myRand()%(i+1);
            swap(seq[i], seq[j]);
        }
        return seq;
    }
    static Scalar RandReal(Scalar s=1.0)
    {
        return (myRand()%1000)/1000.0*s;
    }
    static Scalar RandRealRange(Scalar s)
    {
        return RandReal(2*s)-s;
    }
    ObjTensor& fillRand(const TShape &_shape, Scalar range=0.1)
    {
        Self();
        fill(_shape);
        for(int i=0; i<self.size(); i++) self[i] = RandRealRange(range);
        return *this;
    }
    ObjTensor& fillNormal(const TShape &_shape, Scalar range=0.1)
    {
        Self();
        fill(_shape);
        
        static std::mt19937 rnd(time(0));
        std::normal_distribution<> dist(0, range);
        for(int i=0; i<self.size(); i++)
        {
            self[i] = dist(rnd);
        }
        return *this;
    }
    
//    ObjTensor& initVar(const TensorShape &_shape, Scalar range=0.01)
//    {
//        return fillRand(_shape, range).rg();
//    }
    ObjTensor& reshape(const TShape &_shape)
    {
        ASSERT(d().shape.flatSize() == _shape.flatSize());
        setShape(_shape);
        return *this;
    }
    template<typename... Args> ObjTensor& reshape(Args... args)
    {
        TShape _shape(args...);
        return reshape(_shape);
    }
    static ObjTensor scalar(Scalar s)
    {
        ObjTensor out;
        out.push_back(s);
        out.d().shape=TShape(1, 1);
        return out;
    }
    ObjTensor& rg(bool _rg=true)
    {
        d().requires_grad = _rg;
        return *this;
    }
    T& d()
    {
        return *this->get();
    }
    const T& d() const
    {
        return *this->get();
    }
//    const ObjTensor& getRef() const
//    {
//        this->get()->ref_count++;
//        return *this;
//    }
    void print(string title="")
    {
        printSub();
        cout<<",";
        d().shape.print();
        if(title.length()>0) cout<<"--- "<<title<<" ---"<<endl;
        cout<<endl;
    }
    void printSub() const
    {
        ConstSelf();
        TShape shape = d().shape.simplify();
        if(shape.size()<=1)
        {
            cout<<"[";
            for(int i=0; i<self.size(); i++)
            {
                cout<<self[i];
                if(i<self.size()-1) cout<<",";
            }
            cout<<"]";
            //cout<<endl;
        }
        else
        {
            cout<<"[";
            for(int i=0; i<d().shape[0]; i++)
            {
                ObjTensor sub=subTensor(0, i, 1);
                sub.printSub();
                if(i<d().shape[0]-1) cout<<","<<endl;
            }
            cout<<"]";
        }
    }

    ObjTensor toShape(const TShape &_shape) const
    {
        ObjTensor out;
        out.copyData(*this);
        out.reshape(_shape);
        
        out.setInputs(*this, nullptr);
        out->grad_func = [](ObjTensor<T> &t, int k)
        {
            ObjTensor g_out;
            g_out.copyData(t->grad);
            g_out.reshape(t->input[0]->shape);
            return g_out;
        };
        return out;
    }
    ObjTensor flatten() const
    {
        ConstSelf();
        if(self.d().shape.size() == 2) return self;
        TShape shape(self.d().shape[0], self.d().shape.strides[0]);
        return toShape(shape);
    }
    
    ObjTensor subTensor(int axis, int start, int n) const
    {
        //return subTensor2(axis, start, n);
        
        ObjTensor out;
        if(d().shape.size()<=1) return out;
        TShape shape = d().shape.subShape(axis, n);
        out.fill(shape.simplify());
        
        auto f = [&](const Scalar& di, int dindex, int oindex)
        {
            out[oindex] = di;
        };
        opSub(axis, start, n, f);
        
        ObjTensor sub_info=ObjTensor({(Scalar)axis, (Scalar)start, (Scalar)n});
        out.setInputs(*this, nullptr);
        out.d().sub_info = sub_info;
        out->grad_func = [](ObjTensor<T> &t, int k)
        {
            t->grad.d().sub_info = t.d().sub_info;
            return t->grad;
        };
        
        return out;
    }
    
    template <typename Func>
    void opSub(int axis, int start, int n, const Func &f) const
    {
        ConstSelf();
        vector<int> st=d().shape.strides;
        TShape shape = d().shape.subShape(axis, n);
        int blockSizeOut = shape.flatSize(axis);
        int blockSize = d().shape.flatSize(axis);
        int nBlocks = d().shape.flatSize()/blockSize;
        
        for(int i=0; i<nBlocks; i++)
        {
            int index = i * blockSize + start * st[axis];
            int oindex = i * blockSizeOut;
            for(int j=0; j<blockSizeOut; j++)
            {
                f(self[index], index, oindex);
                index++;
                oindex++;
            }
        }
    }
    
    ObjTensor concat(const ObjTensor& t) const
    {
        ConstSelf();
        ObjTensor out;
        int blockSizeA = d().shape.back();
        int blockSizeB = t.d().shape.back();
        int nBlocks = d().shape.flatSize()/blockSizeA;
        TShape shape = d().shape;
        shape.back() += blockSizeB;
        out.fill(shape);
        
        Scalar *pDataOut = &out[0];
        const Scalar *pDataA = &self[0];
        const Scalar *pDataB = &t[0];
        for(int i=0; i<nBlocks; i++)
        {
            memcpy(pDataOut, pDataA, sizeof(Scalar)*blockSizeA);
            pDataOut += blockSizeA;
            pDataA += blockSizeA;
            memcpy(pDataOut, pDataB, sizeof(Scalar)*blockSizeB);
            pDataOut += blockSizeB;
            pDataB += blockSizeB;
        }
        
        out.setInputs(*this, t);
        out->grad_func = [](ObjTensor<T> &t, int k)
        {
            int axis = 0;
            for(int i=0; i<t.d().shape.size(); i++)
            {
                if(t.d().shape[i] != t->input[0].d().shape[i])
                {
                    axis = i;
                    break;
                }
            }
            if(k==0) return t->grad.subTensor(axis, 0, t->input[0]->shape[axis]);
            else return t->grad.subTensor(axis, t->input[0]->shape[axis], t->input[1]->shape[axis]);
        };
        
        return out;
    }
    
    void setInputs(const ObjTensor& u, const ObjTensor& v)
    {
        if(T::option().no_grad) return;
        
        d().input.clear();
        if(u) d().input.push_back(u);
        if(v) d().input.push_back(v);
        updateRequiresGrad();
    }
    template <typename Func>
    ObjTensor opReduce(int axis, const Func &f) const
    {
        ConstSelf();
        if(axis<0) axis += d().shape.size();
        ASSERT((axis>=0 && axis<d().shape.size()) || axis==AXIS_NONE);
        
        TShape oshape = d().shape;
        if(axis==AXIS_NONE)
        {
            oshape = TShape(1, 1);
            
            ObjTensor out;
            out.fill(oshape);
            out[0] = f(self);
            return out;
        }
        else
        {
            oshape[axis] = 1;
            
            ObjTensor out;
            out.fill(oshape);
            vector<int> coor_out;
            ObjTensor row;
            row.fill(TShape(1, d().shape[axis]));
            for(int i=0; i<out.size(); i++)
            {
                out.d().shape.getCoor(coor_out, i);
                int index = d().shape.getIndex(coor_out);
                for(int j=0; j<d().shape[axis]; j++)
                {
                    row[j] = self[index];
                    index += d().shape.strides[axis];
                }
                out[i] = f(row);
            }
            return out;
        }
        //oshape.updateStrides();
        
    }
    ObjTensor floor()
    {
        ConstSelf();
        ObjTensor out;
        out.fill(d().shape);
        for(int i=0; i<self.size(); i++)
        {
            out[i] = ::floor(self[i]);
        }
        return out;
    }
    ObjTensor operator == (const ObjTensor &t) const
    {
        ConstSelf();
        ObjTensor out;
        out.fill(d().shape);
        for(int i=0; i<self.size(); i++)
        {
            Scalar dis = self[i] - t[i];
            out[i] = (dis * dis < 1e-12) ? 1.0 : 0.0;
        }
        return out;
    }
    ObjTensor argMax(int axis) const
    {
        auto f = [](const ObjTensor &t)
        {
            //vector<Scalar> &data = t->data;
            int index=-1;
            Scalar value=0;
            for(int i=0; i<t.size(); i++)
            {
                if(t[i]>value || index<0)
                {
                    value=t[i];
                    index=i;
                }
            }
            return index;
        };
        
        return opReduce(axis, f);
    }
    
    ObjTensor sum(int axis=AXIS_NONE) const
    {
        auto f = [](const ObjTensor &t)
        {
            //vector<Scalar> &data = t->data;
            Scalar s = 0.0;
            for(int i=0; i<t.size(); i++)
            {
                s += t[i];
            }
            return s;
        };
        
        ObjTensor out=opReduce(axis, f);
        
        out.setInputs(*this, nullptr);
        out->grad_func = [](ObjTensor<T> &t, int k)
        {
            return t->grad;
        };
        
        //cout<<endl;
        return out;
    }
    ObjTensor mean(int axis=AXIS_NONE) const
    {
        ConstSelf();
        //Scalar n = axis >= 0 ? d().shape[axis] : self.size();
        Scalar n = axis == AXIS_NONE ? self.size() : d().shape[axis];
        return sum(axis)*(1.0/n);
    }
    
    ObjTensor operator+(const ObjTensor &t) const
    {
        auto f = [](Scalar &out, const Scalar &in1, const Scalar &in2){
            out = in1 + in2;
        };
        ObjTensor out=opBasic(t, f);

        out.setInputs(*this, t);
        out->grad_func = [](ObjTensor<T> &t, int k)
        {
            return t->grad;
        };
        return out;
    }
    ObjTensor operator-(const ObjTensor &t) const
    {
        return *this + (-t);
    }
    ObjTensor operator-() const
    {
        return *this * -1.0;
    }
    ObjTensor operator*(const ObjTensor &t) const
    {
        auto f = [](Scalar &out, const Scalar &in1, const Scalar &in2){
            //cout<<in1<<","<<in2<<endl;
            out = in1 * in2;
        };
        ObjTensor out=opBasic(t, f);
        
        out.setInputs(*this, t);
        out->grad_func = [](ObjTensor<T> &t, int k)
        {
            if(k == 0) return t->grad * t->input[1];
            else return t->grad * t->input[0];
        };
        return out;
    }
    void checkShape(const ObjTensor &t) const
    {
        int maxSize = (int)max(d().shape.size(), t.d().shape.size());
        TShape shapeA = d().shape.extend(maxSize);
        TShape shapeB = t.d().shape.extend(maxSize);
        for(int i=0; i<maxSize; i++)
        {
            bool eq = shapeA[i] == shapeB[i];
            bool one = (shapeA[i] == 1) || (shapeB[i] == 1);
            auto f = [&](){shapeA.print(); shapeB.print();};
            ASSERT_F(eq || one, f);
        }
    }
    template <typename Func>
    ObjTensor opBasic(const ObjTensor &t, const Func &f) const
    {
        ConstSelf();
        if(size()<t.size()) return t.opBasic(*this, f);
        if(t.size() == 0) return ObjTensor();
        
        checkShape(t);
        
        ObjTensor out;
        out.fill(d().shape);
        
        if(size() == t.size())
        {
            for(int i=0; i<size(); i++)
            {
                //out[i] = d().data[i] + t[i];
                f(out[i], self[i], t[i]);
            }
        }
        else if(d().shape.size() == 2)
        {
            TShape shape=t.d().shape.extend(2);
            if(shape.flatSize()==1)
            {
                for(int i=0; i<size(); i++)
                {
                    f(out[i], self[i], t[0]);
                }
            }
            else if(shape[0] == 1)
            {
                for(int i=0; i<size(); i++)
                {
                    int ti = i % t.d().shape.back();
                    f(out[i], self[i], t[ti]);
                }
            }
            else
            {
                for(int i=0; i<size(); i++)
                {
                    int ti = i / d().shape.back();
                    f(out[i], self[i], t[ti]);
                }
            }
        }
        else
        {
            int maxSize = (int)max(d().shape.size(), t.d().shape.size());
            TShape shapeA = d().shape.extend(maxSize);
            TShape shapeB = t.d().shape.extend(maxSize);
            TShape shapeMax;
            for(int i=0; i<maxSize; i++)
            {
                shapeMax.push_back(max(shapeA[i], shapeB[i]));
            }
            ObjTensor a = extend(shapeMax);
            ObjTensor b = t.extend(shapeMax);
            for(int i=0; i<size(); i++)
            {
                f(out[i], a[i], b[i]);
            }
        }
        
        return out;
    }
    ObjTensor extend(const TShape &_extShape) const
    {
        ConstSelf();
        TShape extShape = _extShape.simplify();
        TShape simShape = d().shape.simplify();
        if(simShape ==  extShape) return *this;
        ObjTensor out;
        TShape shape = d().shape;
        if(shape.size()<extShape.size())
        {
            shape = d().shape.extend((int)extShape.size());
        }
        
        TShape oshape = shape;
        
        int axis=-1;
        for(int i=(int)shape.size()-1; i>=0; i--)
        {
            if(shape[i]==1 && extShape[i]>1)
            {
                axis=i;
                break;
            }
        }
        if(axis == -1) return *this;
        //cout<<axis<<endl;
        int st=shape.strides[axis];
        int n=(int)self.size()/st;
        oshape[axis] = extShape[axis];
        //oshape.print();
        out.fill(oshape);
        int ost=out.d().shape.strides[axis];
        for(int i=0; i<n; i++)
        {
            for(int j=0; j<extShape[axis]; j++)
            {
                memcpy(&out[i*ost+j*st], &self[i*st], sizeof(Scalar)*st);
            }
        }
        
        return out.extend(extShape);
    }
    ObjTensor& operator+=(const ObjTensor &t)
    {
        *this = *this + t;
        return *this;
    }
    ObjTensor& operator*=(const ObjTensor &t)
    {
        *this = *this * t;
        return *this;
    }
    void sgd(Scalar lr)
    {
        Self();
        for(int i=0; i<self.size(); i++)
        {
            self[i] -= self.d().grad[i] * lr;
        }
    }

    ObjTensor transpose() const
    {
        ConstSelf();
        ObjTensor out;
        
        TShape shapeA = d().shape.extend(2);
        int r0 = shapeA[0], c0 = shapeA[1];
        int r1 = c0, c1 = r0;
        out.fill(TShape(r1, c1));
        
//        auto mA=Eigen::Map<const MyMatrix>(&d().data[0], r0, c0);
//        auto mC=Eigen::Map<MyMatrix>(&out[0], c0, r0);
//        mC = mA.transpose();
        
        for(int i=0; i<r1; i++)
        {
            Scalar *pOut = &out[i*c1];
            const Scalar *pSrc = &self[i];
            for(int j=0; j<c1; j++)
            {
                *pOut = *pSrc;
                pOut++;
                pSrc += c0;
            }
        }
        
        out.setInputs(*this, nullptr);
        out->grad_func = [](ObjTensor<T> &t, int k)
        {
            return t->grad.transpose();
        };
        return out;
    }
    void extendShape(const ObjTensor &t)
    {
        if(d().shape.size() == t.d().shape.size()) return;
        int maxSize = max(d().shape.size(), t.d().shape.size());
        d().shape.extendShape(maxSize);
        t.d().shape.extendShape(maxSize);
    }
    ObjTensor operator % (const ObjTensor &t) const
    {
        ConstSelf();
        //extendShape(t);
        ObjTensor out;
        int maxSize = 2;//max(d().shape.size(), t.d().shape.size());
        //cout<<maxSize<<endl;
        TShape shapeA = d().shape.extend(maxSize);
        TShape shapeB = t.d().shape.extend(maxSize);
        //shapeA.print();
        //shapeB.print();
        int outRows = shapeA[0];
        int outCols = shapeB[1];
        out.fill(TShape(outRows, outCols));
        
        if(1)
        {
            bool transpose1=false;
            bool transpose2=false;
            int m=outRows;
            int n=outCols;
            int r0 = shapeA[0], c0 = shapeA[1];
            int r1 = shapeB[0], c1 = shapeB[1];
            int k=transpose1 ? r0 : c0;
            //cout<<m<<","<<n<<","<<k<<","<<endl;
            Scalar alpha=1.0, beta=0.0;
            CBLAS_TRANSPOSE t1 = transpose1 ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE t2 = transpose2 ? CblasTrans : CblasNoTrans;
            GEMM(CblasRowMajor, t1, t2, m, n, k, alpha, &self[0], c0, &t[0], c1, beta, &out[0], n);
        }
        else
        {
//            int r0 = shapeA[0], c0 = shapeA[1];
//            int r1 = shapeB[0], c1 = shapeB[1];
//            auto mA=Eigen::Map<const MyMatrix>(&self[0], r0, c0);
//            auto mB=Eigen::Map<const MyMatrix>(&t[0], r1, c1);
//            auto mC=Eigen::Map<MyMatrix>(&out[0], r0, c1);
//            
//            mC.noalias() = mA * mB;
        }
        
        //out->op = "%";

        out.setInputs(*this, t);
        out->grad_func = [](ObjTensor<T> &t, int k)
        {
            if(k == 0) return t->grad % t->input[1].transpose();
            else return t->input[0].transpose() % t->grad;
        };
        
        return out;
    }
    static void ScalarAdd(Scalar *pDest, Scalar *pSrc, int n)
    {
        for(int i=0; i<n; i++) pDest[i] += pSrc[i];
    }
    ObjTensor embed(const ObjTensor &t) const
    {
        ConstSelf();
        
        ASSERT(self.d().shape.size()==2 && t.d().shape.size()==2 && t.d().shape[1]==1);
        
        ObjTensor out;
        int rows = t.d().shape[0];
        int cols = self.d().shape[1] * t.d().shape[1];
        out.fill({rows, cols});
        
        int dense_size = self.d().shape[1];
        for(int i=0; i<t.size(); i++)
        {
            memcpy(&out[i*dense_size], &self[(int)t[i]*dense_size], sizeof(Scalar)*dense_size);
        }
        
        out.setInputs(t, *this);
        out->grad_func = [](ObjTensor<T> &t, int k)
        {
            ObjTensor g_out;
            g_out.fill(t->input[1].d().shape, 0.0);
            int dense_size = g_out.d().shape[1];
            int rows = t->input[0].d().shape[0];
            for(int i=0; i<rows; i++)
            {
                int index = (int)t->input[0][i];
                //memcpy(&g_out[index*dense_size], &t->grad[i*dense_size], sizeof(Scalar)*dense_size);
                ScalarAdd(&g_out[index*dense_size], &t->grad[i*dense_size], dense_size);
            }
            return g_out;
        };
        
        return out;
    }
    ObjTensor dropOut(Scalar ratio, bool scale=true) const
    {
        ConstSelf();
        ObjTensor tp;
        TShape shape = d().shape;
        shape[0] = 1;
        tp.fill(shape);
        Scalar mag = 1.0/(1.0 - ratio);
        if(!scale) mag = 1.0;
        for(int i=0; i<tp.size(); i++)
        {
            Scalar n = RandReal();
            tp[i] = n<ratio ? 0.0 : mag;
        }
        return self * tp;
    }
    
    template <typename Func, typename Funcg>
    ObjTensor opPoint(const Func &f, const Funcg &gf) const
    {
        ConstSelf();
        ObjTensor out;
        out.fill(d().shape);
        
        for(int i=0; i<self.size(); i++)
        {
            Scalar x = self[i];
            out[i] = f(x);
        }
        
        out.setInputs(*this, nullptr);
        out->grad_func = [=](ObjTensor<T> &t, int k)
        {
            ObjTensor g_out;
            if(t.size()!=t->grad.size()) return g_out;
            g_out.fill(t.d().shape);
            for(int i=0; i<t.size(); i++)
            {
                Scalar fx = t[i];
                Scalar x = t->input[0][i];
                g_out[i] = t->grad[i] * gf(x, fx);
            }
            
            return g_out;
        };
        
        return out;
    }
    ObjTensor sigmoid()
    {
        //return softsig(1);
        
//        auto nf=[](Scalar x)
//        {
//            Scalar exp_value = exp(x);
//            return exp_value / (1.0+exp_value);
//        };
        auto Rough_f=[](Scalar value)
        {
            Scalar x = ::abs(value);
            Scalar x2 = x*x;
            Scalar e = 1.0 + x + x2*0.555 + x2*x2*0.143;
            return 1.0 / (1.0 + (value > 0 ? 1.0 / e : e));
        };
        
        auto f=[](Scalar x)
        {
            return (1.0 + std::tanh(x * 0.5)) * 0.5;
        };
        auto gf=[](Scalar x, Scalar fx)
        {
            return fx * (1.0 - fx);
        };
        return opPoint(Rough_f, gf);
    }
    ObjTensor tanh()
    {
        //return softsign(1);
        
        auto Rough_f=[](Scalar value)
        {
            Scalar x = ::abs(value) * 2.0;
            Scalar x2 = x*x;
            Scalar e = 1.0 + x + x2*0.555 + x2*x2*0.143;
            return 2.0 / (1.0 + (value > 0 ? 1.0 / e : e)) - 1.0;
        };
        
        auto f=[](Scalar x)
        {
            return std::tanh(x);
        };
        auto gf=[](Scalar x, Scalar fx)
        {
            return 1.0 - fx * fx;
        };
        return opPoint(Rough_f, gf);
    }
    ObjTensor softsig(Scalar alpha=1.0)
    {
        auto f=[=](Scalar x)
        {
            x *= alpha;
            Scalar t = x / (1.0 + fabs(x));
            //return t;
            return (1.0 + t) * 0.5;
        };
        auto gf=[=](Scalar x, Scalar fx)
        {
            return fx * (1.0 - fx) * alpha;
            //Scalar t = 1.0 + fabs(x);
            //return alpha*0.5*1.0 / (t * t);
        };
        return opPoint(f, gf);
    }

    ObjTensor gaussian()
    {
        auto f=[](Scalar x)
        {
            return std::exp(-x*x);
        };
        auto gf=[](Scalar x, Scalar fx)
        {
            return -2*x*fx;
        };
        return opPoint(f, gf);
    }
    ObjTensor relu(Scalar k=0.0)
    {
        auto f=[=](Scalar x)
        {
            Scalar s = x > 0.0 ? 1.0 : k;
            return x * s;
        };
        auto gf=[=](Scalar x, Scalar fx)
        {
            return x > 0.0 ? 1.0 : k;
        };
        return opPoint(f, gf);
    }
    ObjTensor slu() //Scalar k=0.0
    {
        auto f=[=](Scalar x)
        {
            return x<0.0 ? 0.0 : (x<1.0 ? x : 1.0);
        };
        auto gf=[=](Scalar x, Scalar fx)
        {
            return x<0.0 ? 0.0 : (x<1.0 ? 1 : 0.0);
        };
        return opPoint(f, gf);
    }
    ObjTensor tlu() //Scalar k=0.0
    {
        auto f=[=](Scalar x)
        {
            return x<0.0 ? 0.0 : (x<1.0 ? x : 0.0);
        };
        auto gf=[=](Scalar x, Scalar fx)
        {
            return x<0.0 ? 0.0 : (x<1.0 ? 1 : 0.0);
        };
        return opPoint(f, gf);
    }
    ObjTensor softsign(Scalar alpha=1.0)
    {
        auto f=[=](Scalar x)
        {
            x *= alpha;
            Scalar t = x / (1.0 + fabs(x));
            return t;
            //return (1.0 + t) * 0.5;
        };
        auto gf=[=](Scalar x, Scalar fx)
        {
            Scalar t = 1.0 + fabs(x);
            return 1.0 / (t * t) * alpha;
        };
        return opPoint(f, gf);
    }
    ObjTensor softsign1(Scalar alpha=1.0)
    {
        auto f=[=](Scalar x)
        {
            x *= alpha;
            Scalar t = x / (1.0 + fabs(x));
            //return t;
            return (1.0 + t) * 0.5;
        };
        auto gf=[=](Scalar x, Scalar fx)
        {
            Scalar t = 1.0 + fabs(x);
            return alpha*0.5*1.0 / (t * t);
        };
        return opPoint(f, gf);
    }
    ObjTensor sin()
    {
        auto f=[=](Scalar x)
        {
            return ::sin(x);
        };
        auto gf=[=](Scalar x, Scalar fx)
        {
            return ::cos(x);
        };
        return opPoint(f, gf);
    }
    
    ObjTensor mse(const ObjTensor &t)
    {
        ObjTensor out;

        out = *this - t;
        out = (out * out).mean();
        
        return out;
    }
    
    ObjTensor softmax() const
    {
        ConstSelf();
        ObjTensor out;
        out.fill(d().shape);
        
        int cols = d().shape[-1];
        int rows = (int)self.size()/cols;
        for(int r=0; r<rows; r++)
        {
            const Scalar *pRowIn = &self[r*cols];
            Scalar *pRowOut = &out[r*cols];
            Scalar max_v = pRowOut[0];
            for(int i=0; i<cols; i++)
            {
                if(max_v<pRowOut[i]) max_v=pRowOut[i];
            }
            Scalar sum=0.0;
            for(int i=0; i<cols; i++)
            {
                pRowOut[i]=exp(pRowIn[i]-max_v);
                sum+=pRowOut[i];
            }
            for(int i=0; i<cols; i++)
            {
                pRowOut[i]/=sum;
            }
        }
        
        out.setInputs(*this, nullptr);
        out->grad_func = [=](ObjTensor<T> &t, int k)
        {
            ObjTensor g_out;
            g_out.fill(t.d().shape);
            int cols = t.d().shape[-1];
            int rows = (int)t.size()/cols;
            for(int r=0; r<rows; r++)
            {
                Scalar *pRow = &t[r*cols];
                Scalar *pGrad = &t->grad[r*cols];
                Scalar *pGrad_Out = &g_out[r*cols];
                for (int i=0; i<cols; i++)
                {
                    pGrad_Out[i]=0.0;
                    for (int j=0; j<cols; j++)
                    {
                        Scalar d = (j == i) ? pRow[j] * (1.0 - pRow[i]) : -pRow[j] * pRow[i];
                        pGrad_Out[i] += d * pGrad[j];
                    }
                }
            }
            
            return g_out;
        };
        
        return out;
    }
    

    Scalar length() const
    {
        ConstSelf();
        Scalar len = 0.0;
        Scalar factor = 1.0/size();
        for(int i=0; i<size(); i++)
        {
            len += self[i] * self[i] * factor;
        }
        return sqrt(len);
    }
    
    void updateRequiresGrad()
    {
        T& t=d();
        if(t.input.empty()) return;
        t.requires_grad = false;
        for(auto &u : t.input)
        {
            if(u->requires_grad)
            {
                t.requires_grad=true;
                break;
            }
        }
    }
    
    void addSub(const ObjTensor &t)
    {
        Self();

        auto f = [&](const Scalar& di, int dindex, int oindex)
        {
            self[dindex] += t[oindex];
        };
        int axis = t.d().sub_info[0];
        int start = t.d().sub_info[1];
        int n = t.d().sub_info[2];
        opSub(axis, start, n, f);
        
        //return out;
    }
    
    void initGrad()
    {
        d().grad=ObjTensor().fill(d().shape, 0.0);
    }
    void addGrad(const ObjTensor &g)
    {
        unique_lock<std::mutex> lock(d().grad_mutex);
        
        //static int count = 0;
        //cout<<"addGrad()"<<count++<<endl;
        
        if(d().shape.flatSize()!=g.d().shape.flatSize() && !d().grad) initGrad();
        
        if(d().shape == g.d().shape)
        {
            if(d().grad) d().grad = d().grad + g;
            else d().grad = g;
        }
        else if(g.d().sub_info)
        {
            d().grad.addSub(g);
        }
        else
        {
            // while add a matrix grad to bias vector
            //cout<<"add sum.";
            ObjTensor sg = g;
            int maxSize = (int)max(d().shape.size(), g->shape.size());
            //cout<<"maxSize: "<<maxSize<<endl;
            TShape shapeA = d().shape.extend(maxSize);
            TShape shapeB = g->shape.extend(maxSize);
            sg.reshape(shapeB);
            //shapeA.print();
            //shapeB.print();
            for(int i=0; i<maxSize; i++)
            {
                if(shapeA[i] == 1) sg = sg.sum(i); // && shapeB[i]>1
            }
            
            if(d().grad) d().grad = d().grad + sg;
            else d().grad = sg;
        }
    }
    
    void backward(ObjTensor g=nullptr)
    {
        //static int count = 0;
        //cout<<"backward()"<<count++<<endl;
        
        if(!d().requires_grad) return;
        
        if(!g)
        {
            g=ObjTensor();
            g.fill(d().shape, 1.0);
        }
        
        auto list = getTopoList();
        std::reverse(list.begin(), list.end());
        //cout<<"topo list: "<<list.size()<<endl;

        addGrad(g);
        
        for(int i=0; i<list.size(); i++)
        {
            //list[i].print();
            list[i].applyGrad();
            
            if(list[i].d().grad)
            {
                //cout<<list[i].d().name<<": "<<list[i].d().grad.length()<<endl;
                unique_lock<std::mutex> lock(list[i].d().grad_mutex);
                list[i].d().grad.clearInputs();
            }
        }

    }
    
    void applyGrad()
    {
        T& t=d();
        if(!t.grad_func) return;
        
        for(int i=0; i<t.input.size(); i++)
        {
            if(t.input[i]->requires_grad)
            {
                ObjTensor g_out = t.grad_func(*this, i);
                t.input[i].addGrad(g_out);
            }
        }
        
    }
    
    vector<ObjTensor> getTopoList()
    {
        return getTopoList({*this});
    }
    static vector<ObjTensor> getTopoList(const vector<ObjTensor> &ts)
    {
        vector<ObjTensor> list;
        set<ObjTensor> visited;
        for(auto &t:ts)
        {
            t.getTopoListSub(list, visited);
        }
        return list;
    }
    void getTopoListSub(vector<ObjTensor> &list, set<ObjTensor> &visited) const
    {
        if(visited.count(*this)>0) return;
        visited.insert(*this);
        
        const T& t=d();
        for(auto &u : t.input)
        {
            u.getTopoListSub(list, visited);
        }
        list.push_back(*this);
        //print("topo: ");
    }
    
    vector<ObjTensor> getParameters()
    {
        return getParameters({*this});
    }
    static vector<ObjTensor> getParameters(const vector<ObjTensor> &ts)
    {
        vector<ObjTensor> paras;
        auto list = getTopoList(ts);
        cout<<"TopListSize: "<<list.size()<<endl;
        for(int i=0; i<list.size(); i++)
        {
            T& t=list[i].d();
            if(t.input.empty() && t.requires_grad)
            {
                paras.push_back(list[i]);
            }
        }
        return paras;
    }
    void zero_grad()
    {
        return zero_grad({*this});
    }
    
    static void zero_grad(const vector<ObjTensor> &ts)
    {
        vector<ObjTensor> paras;
        auto list = getTopoList(ts);
        for(int i=0; i<list.size(); i++)
        {
            T& t=list[i].d();
            list[i].updateRequiresGrad();
            
            unique_lock<std::mutex> lock(t.grad_mutex);
            t.grad=nullptr;
            //list[i].initGrad();
        }
    }
   
    ObjTensor detach()
    {
        //ObjTensor out(d().shape, d().data);
        ObjTensor out;
        out.copyData(*this);
        out->requires_grad = d().requires_grad;
        return out;
    }
    ObjTensor& copyData(const ObjTensor& t)
    {
        Self();
        fill(t.d().shape);
        memcpy(&self[0], &t[0], sizeof(Scalar)*t.size());
        return *this;
    }
    ObjTensor& clearInputs()
    {
        d().input.clear();
        return *this;
    }
    
    ObjTensor shuffle(const vector<int> &seq) const
    {
        ASSERT (seq.size() == d().shape[0]);
        
        ConstSelf();
        
        ObjTensor out;
        out.fill(self.d().shape);
        int stride = d().shape.strides[0];
        for(int i=0; i<seq.size(); i++)
        {
            memcpy(&out[i*stride], &self[seq[i]*stride], sizeof(Scalar)*stride);
        }
        return out;
    }
    
    vector<Scalar>& getData() const
    {
        return this->get()->getData();
    }
    Scalar& operator[](int i)
    {
        return getData()[i];
    }
    const Scalar& operator[](int i) const
    {
        return getData()[i];
    }
    size_t size() const
    {
        return getData().size();
    }
    void resize(size_t s)
    {
        //getData().resize(s);
        d().resize(s);
    }
    ObjTensor& setData(const vector<Scalar> &v)
    {
        setShape(TShape(1, v.size()));
        getData() = v;
        return *this;
    }

    static int insNum(int i=0)
    {
        return T::insNum(i);
    }
    static void clean(int i=0)
    {
        T::option().cleanPool();
    }
};

template <class T>
ObjTensor<T> operator-(Scalar st, const ObjTensor<T> &t)
{
    return ObjTensor<T>(st) - t;
}


struct TensorBase
{
    vector<Scalar> *pData;//datav;
    TShape shape;
    bool requires_grad = false;
    function<ObjTensor<TensorBase>(ObjTensor<TensorBase> &, int)> grad_func;
    ObjTensor<TensorBase> grad;
    vector< ObjTensor<TensorBase> > input;
    ObjTensor<TensorBase> sub_info;
    vector<int> idata;
    void *pHelper;
    std::mutex grad_mutex;
    string name;
    string op;
    
    TensorBase():grad(nullptr),sub_info(nullptr)
    {
        insNum(1);
        //pData = new vector<Scalar>();
        pData = option().getBuffer(0);
    }
    ~TensorBase()
    {
        insNum(-1);
        //delete pData;
        option().returnBuffer(pData);
    }
    static int insNum(int i=0)
    {
        std::unique_lock<std::mutex> lock(option().ins_mutex);
        static int num = 0;
        num += i;
        ASSERT(num<20000);
        return num;
    }
    vector<Scalar>& getData()
    {
        return *pData;//datav;
    }
    void resize(int s)
    {
//        getData().resize(s);
//        return;
        
        option().returnBuffer(pData);
        pData = option().getBuffer(s);
    }
    
    struct TensorOption
    {
        int ins_num = 0;
        std::mutex ins_mutex;
        bool no_grad = false;
        bool enable_pool = true;
        std::mutex pool_mutex;
        std::unordered_map< uint64_t, std::map<void*, int> > tensor_pool;
        //std::unordered_map< uint64, std::map<void*, int> > tensor_using;
        
        void print()
        {
            int n_tensor_pool = 0;
            for (auto it=tensor_pool.begin(); it!=tensor_pool.end(); ++it)
            {
                n_tensor_pool += it->second.size();
                //cout<<"pool: "<<it->first<<", "<<it->second.size()<<endl;
            }
            cout<<"tensor_pool: "<<n_tensor_pool<<endl;
            
//            int n_tensor_using = 0;
//            for (auto it=tensor_using.begin(); it!=tensor_using.end(); ++it)
//            {
//                n_tensor_using += it->second.size();
//                //cout<<"using: "<<it->first<<", "<<it->second.size()<<endl;
//            }
//            cout<<"tensor_using: "<<n_tensor_using<<endl;
        }
        void cleanPool()
        {
            int sum_size = 0, n = 0;
            for(auto it=tensor_pool.begin(); it!=tensor_pool.end(); ++it)
            {
                auto tensor_map = it->second;
                for(auto vec=tensor_map.begin(); vec!=tensor_map.end(); ++vec)
                {
                    vector<Scalar>* pVec = (vector<Scalar>*)vec->first;
                    sum_size += pVec->size();
                    delete pVec;
                }
                n += tensor_map.size();
            }
            tensor_pool.clear();
            
            cout<<"clean tensor: "<<n<<" size:"<<sum_size<<endl;
        }
        uint64_t getPoolSize()
        {
            uint64_t poolSize = 0;
            for (auto it=tensor_pool.begin(); it!=tensor_pool.end(); ++it)
            {
                poolSize += it->second.size();
            }
            return poolSize;
        }
        vector<Scalar>* getBuffer(int s)
        {
            std::unique_lock<std::mutex> lock(pool_mutex);
            
            vector<Scalar>* out;
            if(!enable_pool)
            {
                out = new vector<Scalar>();
                out->resize(s);
                return out;
            }
            
            int key = s;
            auto& ts_pool = tensor_pool[key];
            //auto& ts_using = tensor_using[key];
            if(ts_pool.size()>0)
            {
                //cout<<"found tensor."<<endl;
                auto it = ts_pool.begin();
                out = (vector<Scalar>*)it->first;
                ts_pool.erase(out);
                //ts_using[out] = 1;
            }
            else
            {
                out = new vector<Scalar>();
                out->resize(s);
                //ts_using[out] = 1;
            }
            return out;
        }
        void returnBuffer(vector<Scalar>* t)
        {
            std::unique_lock<std::mutex> lock(pool_mutex);
            
            if(!t) return;
            if(t->empty() || !enable_pool)
            {
                delete t;
                return;
            }
            int key = t->size();
            auto& ts_pool = tensor_pool[key];
            //auto& ts_using = tensor_using[key];
            
            if(getPoolSize()>20000)
            {
                //cout<<"pool is full. "<<endl;
                delete t;
                //ts_using.erase(t);
            }
            else
            {
                ts_pool[t] = 1;
            }
        }
    };
    
    static TensorOption & option()
    {
        static TensorOption opt;
        return opt;
    }
    
//    void reset()
//    {
//        requires_grad = false;
//        grad_func = nullptr;
//        u = nullptr;
//        v = nullptr;
//        grad = nullptr;
//        subIndex.clear();
//    }
    
    
};

typedef ObjTensor<TensorBase> objTensor;



struct NoTrace
{
    NoTrace()
    {
        TensorBase::option().no_grad = true;
    }
    ~NoTrace()
    {
        TensorBase::option().no_grad = false;
    }
};

struct Optimizer
{
    vector<objTensor> parameters;
    Optimizer(){}
    Optimizer(const vector<objTensor> &paras)
    {
        parameters = paras;
    }
    void step(Scalar lr=0.01)
    {
        for(int i=0; i<parameters.size(); i++)
        {
            objTensor &p = parameters[i];
            //if(!p->grad) continue;
            //p.print();
            //p.copyData(p - p->grad * lr);
            p.sgd(lr);
        }
    }
    int numScalars()
    {
        int n = 0;
        for(int i=0; i<parameters.size(); i++)
        {
            n += parameters[i].d().shape.flatSize();
        }
        return n;
    }
};

struct Module
{
    Optimizer opt;
    //objTensor d;
    Scalar lr;
    Scalar acc;
    Scalar d_sum;
    vector<objTensor> paras;
    
    Scalar min_d_sum;
    int min_d_epoc;
    
    bool training = false;
    
    virtual objTensor forward(const objTensor &x) = 0;
    auto getFx()
    {
        return std::bind(&Module::forward, this, std::placeholders::_1);
    }
    
    template <typename FuncStep, typename FuncEpoch>
    void fit(const objTensor &xt, const objTensor &yt, int epochs, Scalar _lr, FuncStep fStep, FuncEpoch fEpoch, int batch_size=100, int n_threads=1, bool shuffle=false)
    {
        lr = _lr;
        
        min_d_sum = -1.0;
        min_d_epoc = -1;
        
        TimeServer ts;
        int rows = xt.d().shape[0];
        int n_batch = rows/batch_size;
        for(int e=0; e<epochs; e++)
        {
            //objTensor d_sum = objTensor().fill({1});
            d_sum = 0.0;
            objTensor d[n_threads];
            int acc_count[n_threads];
            for(int i=0; i<n_threads; i++) acc_count[i] = 0;
            
            for(int i=0; i<n_batch; i++)
            {
                int index=i*batch_size;//
                if(shuffle) index=objTensor::myRand()%(rows-batch_size);
                objTensor xs=xt.subTensor(0, index, batch_size);
                objTensor ys=yt.subTensor(0, index, batch_size);
                
                int tb_size = xs.d().shape[0]/n_threads;
                //cout<<"tb_size "<<tb_size<<endl;
                training = true;
                
                auto thf = [&](int tid)
                {
                    objTensor x = xs.subTensor(0, tb_size*tid, tb_size);
                    objTensor y = ys.subTensor(0, tb_size*tid, tb_size);
                    objTensor net = forward(x);//f(x);
                    d[tid] = net.mse(y)*tb_size;
                    d[tid].zero_grad();
                    
                    objTensor eq = net.argMax(1) == y.argMax(1);
                    acc_count[tid] += eq.sum()[0];
                };
                auto thf2 = [&](int tid)
                {
                    d[tid].backward();
                };
                
                //ts.Set();
                
                objt::exe_all(thf, n_threads);
                //cout<<"forward: "<<ts.GetElapsed()<<endl;
                objt::exe_all(thf2, n_threads);
                //cout<<"backward: "<<ts.GetElapsed()<<endl;
                
                training = false;
                auto nt = NoTrace();
                
                //auto paras = d[0].getParameters();
                if(paras.empty()) paras = d[0].getParameters();
                opt = Optimizer(paras);
                opt.step(lr);
                
                //cout<<"step: "<<ts.GetElapsed()<<endl;
                
                for(int i=0; i<n_threads; i++)
                {
                    d_sum += d[i][0];
                }
                
                {
                    //objt::MutexLocker locker(0);
                    fStep(*this, e, i);
                }
                
            }
            
            Scalar acc_sum = 0.0;
            
            for(int i=0; i<n_threads; i++)
            {
                acc_sum += acc_count[i];
            }
            acc = acc_sum / rows;
            
            {
                //objt::MutexLocker locker(0);
                fEpoch(*this, e);
            }
            
            if(d_sum < min_d_sum || min_d_sum<0)
            {
                min_d_sum = d_sum;
                min_d_epoc = e;
            }
            
            if(e - min_d_epoc >= 3 && lr > 1e-6)
            {
                lr *= 0.5;
                min_d_epoc = e;
                cout<<"Reduce lr: "<<lr<<endl;
            }
            
        }
    }
    
    Scalar test(const objTensor &x_test, const objTensor &y_test, int batch_size=25)
    {
        objTensor out;
        out.fill(y_test.d().shape);
        int rows = y_test.d().shape[0];
        int cols = y_test.d().shape[1];
        int n_batch = rows/batch_size;
        if(rows - n_batch*batch_size>0) n_batch++;
        objt::for_i(0, n_batch, [&](int i, int tid)
        //for(int i=0; i<n_batch; i++)
        {
            int index = i * batch_size;
            int len=min(batch_size, rows-index);
            objTensor x = x_test.subTensor(0, index, len);
            objTensor y = forward(x);
            memcpy(&out[index * cols], &y[0], sizeof(Scalar)*y.size());
        }
        );
        
        objTensor eq = out.argMax(1) == y_test.argMax(1);
        Scalar acc = eq.mean()[0];
        return acc;
    }
    
    void rg(bool _rg)
    {
        for(int i=0; i<paras.size(); i++) paras[i].rg(_rg);
    }
};

struct Linear : public Module
{
    objTensor W, B;
    int size_in=0, size_out=0;
    Scalar init_value;
    
    Linear()
    {}
    Linear(int _size_out, Scalar value=0.1)
    {
        size_out = _size_out;
        init_value = value;
    }
    void checkShape(const objTensor &x)
    {
        static std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        
        int _size_in = x.d().shape.back();
        TShape shape(_size_in, size_out);
        //shape.print();
        if(shape != W.d().shape)
        {
            W.fillRand(shape, init_value).rg();
            B.fillRand(TShape(1, size_out), init_value).rg();
            W->name = "linear W";
            B->name = "linear B";
        }
    }
    void reset()
    {
        W = objTensor();
        B = objTensor();
    }
    objTensor forward(const objTensor &xt)
    {
        ASSERT(size_out > 0);
        objTensor x = xt.flatten();
        checkShape(x);
        return x % W + B;
    }
};

struct Linear2 : public Module
{
    objTensor W1, W2, B;
    int size_in=0, size_out=0;
    
    Linear2(){}
    Linear2(int _size_out)
    {
        size_out = _size_out;
    }
    void checkShape(const objTensor &x1, const objTensor &x2)
    {
        static std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        
        int _size_in1 = x1.d().shape.back();
        int _size_in2 = x2.d().shape.back();
        TShape shape1(_size_in1, size_out);
        TShape shape2(_size_in2, size_out);
        if(shape1 != W1.d().shape || shape2 != W2.d().shape)
        {
            W1.fillRand(shape1).rg();
            W2.fillRand(shape2).rg();
            B.fillRand(TShape(1, size_out)).rg();
        }
    }
    
    objTensor forward(const objTensor &x)
    {
        objTensor out;
        //checkShape(x);
        //return x % W1 + B;
        return out;
    }
    objTensor forward(const objTensor &x1, const objTensor &x2)
    {
        checkShape(x1, x2);
        return x1 % W1 + x2 % W2 + B;
    }
};

struct Mlp : public Module
{
    vector<Linear> lns;
    
    Mlp(){}
    Mlp(int _hidden_size, int _size_out)
    {
        lns.push_back(Linear(_hidden_size));
        lns.push_back(Linear(_size_out));
    }
    Mlp(const initializer_list<int> &list)
    {
        for(auto s : list)
        {
            lns.push_back(Linear(s));
        }
    }
    
    objTensor forward(const objTensor &x)
    {
        objTensor y=x;
        for(int i=0; i<lns.size(); i++)
        {
            y = lns[i].forward(y);
            if(i<lns.size()-1) y = y.relu();
            else y = y.softmax();
        }
        return y;
    }
};

struct FeatureMlp : public Module
{
    Mlp mlp;
    vector<int> featIndex;
    int numFeat;
    
    FeatureMlp(){}
    FeatureMlp(int nFeat, int maxIndex, const initializer_list<int> &list)
    {
        mlp = Mlp(list);
        numFeat = nFeat;
        initFeatIndex(nFeat, maxIndex);
    }
    void initFeatIndex(int nFeat, int maxIndex)
    {
        if(nFeat<=0 || maxIndex<=0) return;
        featIndex.resize(nFeat);
        for(int i=0; i<nFeat; i++)
        {
            int index = rand() % maxIndex;
            featIndex[i] = index;
        }
    }
    
    objTensor select(const objTensor &t, vector<int>& index)
    {
        if(numFeat<=0) return t;
        objTensor out;
        out.fill(TShape(t.d().shape[0], index.size()));
        for(int r=0; r<t.d().shape[0]; r++)
        {
            for(int i=0; i<index.size(); i++)
            {
                int pos_t = r * t.d().shape[1] + index[i];
                int pos_out = r * (int)index.size() + i;
                out[pos_out] = t[pos_t];
            }
        }
        return out;
    };
    
    objTensor forward(const objTensor &x)
    {
        objTensor sx = select(x, featIndex);
        sx = mlp.forward(sx);//.softmax(); //.sigmoid();
        return sx;
    }
    auto getFx()
    {
        return std::bind(&FeatureMlp::forward, this, std::placeholders::_1);
    }
};

struct FeatureNet : public Module
{
    struct Feature
    {
        FeatureMlp feat;
        Scalar score;
        
        Feature(){}
        Feature(const FeatureMlp& f, const Scalar s)
        {
            feat = f;
            score = s;
        }
    };
    vector<Feature> feats;
    int maxFeats = 32;
    
    FeatureNet(){}
    FeatureNet(int maxfeats)
    {
        maxFeats = maxfeats;
    }
    bool addFeature(FeatureMlp& feat, Scalar score)
    {
        bool added = false;
        Feature nf(feat, score);
        for(int i=0; i<feats.size(); i++)
        {
            if(nf.score > feats[i].score)
            {
                feats.insert(feats.begin()+i, nf);
                added = true;
                break;
            }
        }
        if(feats.size() < maxFeats && !added)
        {
            feats.push_back(nf);
            added = true;
        }
        if(feats.size() > maxFeats) feats.pop_back();
        return added;
    }
    
    objTensor forward(const objTensor &x)
    {
        objTensor out;
        for(int i=0; i<feats.size(); i++)
        {
            if(i == 0) out = feats[i].feat.forward(x);
            else out += feats[i].feat.forward(x);
        }
        out *= 1.0/feats.size();
        return out;
    }
    
    auto getFx()
    {
        return std::bind(&FeatureNet::forward, this, std::placeholders::_1);
    }
};

enum PADDING
{
    PADDING_SAME = 0,
    PADDING_VALID,
};
struct Conv : public Module
{
    int data_in_width;
    int data_in_height;
    int data_in_channel;
    int k_width;
    int k_height;
    int out_channel;
    int stride=1;
    int padding=0;
    bool bias;
    
    int data_out_width;
    int data_out_height;
    int pad_width;
    int pad_height;
    Linear linear;
    
    
    Conv(){}
    ~Conv()
    {
    }
    Conv(int _k_width, int _k_height, int _out_channel, int _stride=1, int _padding=PADDING_SAME, int use_bias=1) //int width, int height, int channel,
    {
        k_width=_k_width;
        k_height=_k_height;
        out_channel=_out_channel;
        stride=_stride;
        padding=_padding;
        bias=use_bias;
        
        int cols=out_channel;
        int rows=k_height*k_width*data_in_channel;
        linear = Linear(cols, 0.1);
        
    }
    
    
    void createPad(objTensor& paded, objTensor& pDataRow)
    {
        if(padding==PADDING_VALID)
        {
            paded = pDataRow;
            return;
        }
        //int pad_width=data_in_width+k_width/2*2;
        //int pad_height=data_in_height+k_height/2*2;
        int pad_line_size=pad_width*data_in_channel;
        int data_in_line_size=data_in_width*data_in_channel;
        paded.fill(TShape(pad_height*pad_line_size, 1), 0.0);
        Scalar *p_pad=&paded[k_height/2*pad_line_size+k_width/2*data_in_channel];
        Scalar *p_data=&pDataRow[0];
        for(int y=0; y<data_in_height; y++)
        {
            memcpy(p_pad, p_data, sizeof(Scalar)*data_in_line_size);
            p_pad+=pad_line_size;
            p_data+=data_in_line_size;
        }
    }
    void unPad(Scalar* pPaded, Scalar* pDataRow)
    {
        //int pad_width=data_in_width+k_width/2*2;
        //int pad_height=data_in_height+k_height/2*2;
        int pad_line_size=pad_width*data_in_channel;
        int data_in_line_size=data_in_width*data_in_channel;
        //paded.fill(pad_height*pad_line_size, 0.0);
        Scalar *p_pad=&pPaded[k_height/2*pad_line_size+k_width/2*data_in_channel];
        Scalar *p_data=&pDataRow[0];
        for(int y=0; y<data_in_height; y++)
        {
            //memcpy(p_pad, p_data, sizeof(Scalar)*data_in_line_size);
            memcpy(p_data, p_pad, sizeof(Scalar)*data_in_line_size);
            p_pad+=pad_line_size;
            p_data+=data_in_line_size;
        }
    }
    void getPadSub(Scalar* pSub, Scalar* pPaded, int x, int y)
    {
        //int pad_width=data_in_width+k_width/2*2;
        int pad_line_size=pad_width*data_in_channel;
        int sub_line_size=k_width*data_in_channel;
        Scalar *p_pad=&pPaded[y*pad_line_size+x*data_in_channel];
        Scalar *p_sub=&pSub[0];
        for(int y=0; y<k_height; y++)
        {
            memcpy(p_sub, p_pad, sizeof(Scalar)*sub_line_size);
            p_pad+=pad_line_size;
            p_sub+=sub_line_size;
        }
    }
    void reversePadSub(Scalar* pSub, Scalar* pPaded, int x, int y)
    {
        //int pad_width=data_in_width+k_width/2*2;
        int pad_line_size=pad_width*data_in_channel;
        int sub_line_size=k_width*data_in_channel;
        Scalar *p_pad=&pPaded[y*pad_line_size+x*data_in_channel];
        Scalar *p_sub=&pSub[0];
        for(int y=0; y<k_height; y++)
        {
            //memcpy(p_sub, p_pad, sizeof(Scalar)*sub_line_size);
            for(int x=0; x<sub_line_size; x++) p_pad[x]+=p_sub[x];
            p_pad+=pad_line_size;
            p_sub+=sub_line_size;
        }
    }
    
    objTensor unroll(const objTensor& data_in)
    {
        //ASSERT(data_in.d().shape[1] == data_in_width*data_in_height*data_in_channel);
        objTensor mat_in;
        int cols=k_height*k_width*data_in_channel;
        int rows_per_image = data_out_width * data_out_height;//data_in_width*data_in_height/(stride*stride);
        int rows=data_in.d().shape[0]*rows_per_image;
        mat_in.fill(TShape(rows, cols));
        //int mat_in_stride = mat_in.d().shape.strides[0];
        
        //TimeServer ts;
        //ts.Set();
        //objt::fori(0, data_in.d().shape[0], [&](int i, int tid)
        for(int i=0; i<data_in.d().shape[0]; i++)
        {
            objTensor row, paded;
            row = data_in.subTensor(0, i, 1);
            createPad(paded, row);
            int row_index=i*rows_per_image;//data_in_width*data_in_height;
            for(int y=0; y<data_in_height; y+=stride)
            {
                for(int x=0; x<data_in_width; x+=stride)
                {
                    getPadSub(&mat_in[row_index*cols], &paded[0], x, y);
                    row_index++;
                }
            }
        }
        //);
        //cout<<ts.GetElapsed()<<endl;
        
        objTensor out = mat_in;
        out.setInputs(data_in, nullptr);
        //out.d().info = {data_in_width, data_in_height, data_in_channel, k_width, k_height};
        out.d().pHelper = this;
        out->grad_func = [=](objTensor &t, int k)
        {
            Conv* pConv = (Conv*)t.d().pHelper;
            int data_in_width = pConv->data_in_width;
            int data_in_height = pConv->data_in_height;
            int data_in_channel = pConv->data_in_channel;
            int k_width = pConv->k_width;
            int k_height = pConv->k_height;
            int pad_width = pConv->pad_width; //data_in_width+k_width/2*2;
            int pad_height = pConv->pad_height; //data_in_height+k_height/2*2;
            int pad_line_size = pad_width*data_in_channel;
            
            objTensor g_out;
            objTensor conv_grad_in = t->grad;
            objTensor ori_in = t->input[0];
            g_out.fill(ori_in.d().shape);
            int cols_out = ori_in.d().shape.strides[0]; //ori_in.d().shape[-1];
            int rows_out = (int)ori_in.size()/cols_out;
            
            int cols_grad = t->grad.d().shape.strides[0]; //t->grad.d().shape[-1];
            //ASSERT(pad_line_size == cols_grad);
            
            //cout<<cols_out<<","<<rows_out<<","<<cols_grad<<endl;
            
            
            //objt::fori(0, rows_out, [&](int i, int tid)
            for(int i=0; i<rows_out; i++)
            {
                objTensor pad_grad_in;
                pad_grad_in.fill(TShape(pad_line_size*pad_height, 1), 0.0);
                Scalar *p_pad_grad_in=&pad_grad_in[0];
                
                int row_index=i*rows_per_image;//data_in_width*data_in_height;
                for(int y=0; y<data_in_height; y+=stride)
                {
                    for(int x=0; x<data_in_width; x+=stride)
                    {
                        pConv->reversePadSub(&conv_grad_in[row_index*cols_grad], p_pad_grad_in, x, y);
                        row_index++;
                    }
                }
                
                pConv->unPad(p_pad_grad_in, &g_out[i*cols_out]);
            }
            //);
            
            return g_out;
        };
        
        return out;
    }
    
    objTensor forward(const objTensor& data_in)
    {
        TShape shape = data_in.d().shape;
        ASSERT(shape.size() == 3 || shape.size() == 4);
        if(shape.size() == 3)
        {
            shape.push_back(1);
            shape.updateStrides();
        }
        data_in_height=shape[1];
        data_in_width=shape[2];
        data_in_channel=shape[3];
        
        if(padding == PADDING_VALID)
        {
            pad_width=data_in_width;
            pad_height=data_in_height;
        }
        else //if(padding == PADDING_SAME)
        {
            pad_width=data_in_width+k_width/2*2;
            pad_height=data_in_height+k_height/2*2;
        }
        
        data_out_width=ceil(data_in_width / (Scalar)stride);
        data_out_height=ceil(data_in_height / (Scalar)stride);
        
        objTensor out = unroll(data_in);
        out = linear.forward(out);
        
        shape[1] = data_out_height;
        shape[2] = data_out_width;
        shape[3] = out_channel;
        
        out = out.toShape(shape);
        
        return out;
    }
    
    objTensor toPatch(const objTensor& data_in)
    {
        TShape shape = data_in.d().shape;
        ASSERT(shape.size() == 3 || shape.size() == 4);
        if(shape.size() == 3)
        {
            shape.push_back(1);
            shape.updateStrides();
        }
        data_in_height=shape[1];
        data_in_width=shape[2];
        data_in_channel=shape[3];
        
        if(padding == PADDING_VALID)
        {
            pad_width=data_in_width;
            pad_height=data_in_height;
        }
        else //if(padding == PADDING_SAME)
        {
            pad_width=data_in_width+k_width/2*2;
            pad_height=data_in_height+k_height/2*2;
        }
        data_out_width=ceil(data_in_width / (Scalar)stride);
        data_out_height=ceil(data_in_height / (Scalar)stride);
        
        objTensor out = unroll(data_in);
        
        //shape[1] = data_in.size()/();
        shape[2] = k_width*k_height;
        shape[3] = data_in_channel;
        shape[1] = data_in.size()/(shape[0]*shape[2]*shape[3]);
        
        out = out.toShape(shape);
        
        return out;
    }
    
    static objTensor maxPool(const objTensor& data_in)
    {
        TShape shape = data_in.d().shape;
        ASSERT(shape.size() == 3 || shape.size() == 4);
        if(shape.size() == 3)
        {
            shape.push_back(1);
            shape.updateStrides();
        }
        int data_in_height=shape[1];
        int data_in_width=shape[2];
        int data_in_channel=shape[3];
        
        objTensor out;
        int out_height=data_in_height/2;
        int out_width=data_in_width/2;
        shape[1] = out_height;
        shape[2] = out_width;
        shape.updateStrides();
        out.fill(shape);
        out.d().idata.resize(out.size());
        
        int rows = shape[0];
        int cols_in = data_in.d().shape.strides[0];
        int cols_out = shape.strides[0];
        //objt::fori(0, in.rows, [&](int n, int tid)
        for(int n=0; n<rows; n++)
        {
            int vIn=n*cols_in;
            int vOut=n*cols_out;

            for(int y=0; y<out_height; y++)
            {
                int out_line_size=out_width*data_in_channel; //getOutLineSize();
                int o=y*out_line_size;
                o += vOut;
                //Scalar *pOut=&pvOut[o];
                for(int x=0; x<out_width; x++)
                {
                    int in_line_size=data_in_width*data_in_channel;
                    int base=y*2*in_line_size+x*2*data_in_channel;
                    base += vIn;
                    
                    int p[4] = {base, base+data_in_channel, base+in_line_size, base+in_line_size+data_in_channel};

                    for(int c=0; c<data_in_channel; c++)
                    {
                        out[o] = data_in[p[0]];
                        for(int i=0; i<4; i++)
                        {
                            if(out[o] < data_in[p[i]])
                            {
                                out[o] = data_in[p[i]];
                                out.d().idata[o] = p[i];
                            }
                        }
                        o++;
                        p[0]++; p[1]++; p[2]++; p[3]++;
                      
                    }
                }
            }
        }
        //);
        
        out.setInputs(data_in, nullptr);
        out->grad_func = [](objTensor &t, int k)
        {
            objTensor g_out;
            g_out.fill(t->input[0].d().shape, 0.0);
            for(int i=0; i<t.d().idata.size(); i++)
            {
                int index = t.d().idata[i];
                g_out[index] = t->grad[i];
            }
            return g_out;
        };
        return out;
    }
    
};

struct RNN : public Module
{
    int h_size;
    int n_frames;
    vector<objTensor> hs;
};
struct LSTM : public RNN
{
    Linear lnHX, lnY;
    LSTM() : LSTM(128, 1)
    {
        
    }
    LSTM(int _h_size, int _n_frames=1)
    {
        h_size = _h_size;
        n_frames = _n_frames;
        lnHX = Linear(h_size*4);
        lnY = Linear(h_size);
    }
    objTensor forward(const objTensor& x_in)
    {
        //ASSERT(xx.d().shape.size() == 3);
        objTensor xx = x_in.flatten();
        int all_size = xx.d().shape.strides[0];
        int frame_size = all_size/n_frames;
        //cout<<n_frames<<","<<frame_size;
        objTensor h, c;
        h.fill({xx.d().shape[0], h_size}, 0.0);
        c.fill({xx.d().shape[0], h_size}, 0.0);
        //static TimeServer ts;
        //ts.Set();
        for(int i=0; i<n_frames; i++)
        {
            //cout<<"i: "<<i<<endl;
            objTensor x = xx.subTensor(1, i*frame_size, frame_size);
            objTensor hx = h.concat(x);
            objTensor lhx = lnHX.forward(hx);
            //cout<<"lnHx: "<<ts.GetElapsed()<<endl;
            objTensor f = lhx.subTensor(1, h_size*0, h_size).sigmoid();
            objTensor it = lhx.subTensor(1, h_size*1, h_size).sigmoid();
            objTensor ct = lhx.subTensor(1, h_size*2, h_size).tanh();
            objTensor o = lhx.subTensor(1, h_size*3, h_size).sigmoid();
            
            c = c * f + it * ct;
            h = o * c.tanh();
            //cout<<"lnHx2: "<<ts.GetElapsed()<<endl;
        }
        //cout<<ts.GetElapsed()<<endl;
        
        return h;
        //return lnY.forward(h).sigmoid();
    }
};
    
struct GRU : public RNN
{
    Linear2 Z, R, H;
    int version = 1;
    
    GRU() : GRU(128, 1)
    {
        
    }
    GRU(int _h_size, int _n_frames=1)
    {
        h_size = _h_size;
        n_frames = _n_frames;
        Z = Linear2(h_size);
        R = Linear2(h_size);
        H = Linear2(h_size);
    }
    objTensor forward(const objTensor& x_in)
    {
        objTensor xx = x_in.flatten();
        int all_size = xx.d().shape.strides[0];
        int frame_size = all_size/n_frames;
        
        objTensor h;
        h.fill({xx.d().shape[0], h_size}, 0.0);
        for(int i=0; i<n_frames; i++)
        {
            objTensor x = xx.subTensor(1, i*frame_size, frame_size);
            if(version == 0)
            {
                objTensor z = Z.forward(h, x).sigmoid();
                objTensor r = R.forward(h, x).sigmoid();
                objTensor th = H.forward(r * h, x).tanh();
                h = (1.0 - z) * h + z * th;
            }
            else
            {
                objTensor z = Z.forward(h, x).sigmoid();
                //objTensor r = R.forward(h, x).sigmoid();
                objTensor th = H.forward(h, x).tanh();
                h = (1.0 - z) * h + z * th;
            }
        }
        return h;
    }
};
    
struct ConvNet : public Module
{
    vector<Conv> conv, conv2;
    Linear ln, ln2;
    LSTM lstm;
    GRU gru;
    
    //ConvNet(){}
    ConvNet()
    {
        int nk[]={64, 64, 64, 256, 256}; //{32, 64, 128, 256, 256};
        int ks = 3;
        for(int i=0; i<2; i++)
        {
            int stride = i==0? 2 : 1;
            conv.push_back(Conv(ks, ks, nk[i]*1, stride));
            conv2.push_back(Conv(ks, ks, nk[i]*1, stride));
        }
        ln = Linear(512);
        ln2 = Linear(10);
        int hsize = 128;
        lstm = LSTM(hsize, 8);
        gru = GRU(hsize, 8);
    }
    objTensor forward(const objTensor& x)
    {
        objTensor out = x;
        for(int i=0; i<conv.size(); i++)
        {
            //out = Conv::maxPool(out);
            out = conv[i].forward(out).relu();
            out = conv2[i].forward(out).relu();
            //out = Conv::maxPool(out);
            if(training) out = out.dropOut(0.5);
        }
        //out = lstm.forward(out).relu();
        //out = gru.forward(out).relu();
        //if(training) out = out.dropOut(0.5);
        out = ln.forward(out).relu();
        //cout<<"relu out : "<<out.length()<<endl;
        //if(training) out = out.dropOut(0.5);
        out = ln2.forward(out).softmax(); //.sigmoid();//
        return out;
    }
};
    



} //namespace


#include "DataSets.h"

#endif /* agTensor_hpp */
