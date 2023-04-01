//
//  DataSets.h
//  objTensor
//
//  Created by Phyar on 2019/4/2.
//  Copyright © 2019 Phyar. All rights reserved.
//

#ifndef DataSets_hpp
#define DataSets_hpp


#include "objTensor.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
typedef unsigned char uchar;

namespace objt {
    
using namespace std;

struct DataSets{
    
    static vector<vector<uchar>> read_mnist_images(string full_path)
    {
        int number_of_images;
        int image_size;
        
        auto reverseInt = [](int i) {
            unsigned char c1, c2, c3, c4;
            c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
            return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        };
        
        
        ifstream file(full_path, ios::binary);
        
        if(file.is_open()) {
            int magic_number = 0, n_rows = 0, n_cols = 0;
            //cout<<sizeof(magic_number);
            
            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);
            //cout<<"magic: "<<magic_number<<endl;
            
            if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");
            
            file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
            file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
            file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);
            
            //cout<<"number_of_images: "<<number_of_images<<endl;
            
            image_size = n_rows * n_cols;
            
            uchar** _dataset = new uchar*[number_of_images];
            vector<vector<uchar>> images;
            for(int i = 0; i < number_of_images; i++) {
                _dataset[i] = new uchar[image_size];
                file.read((char *)_dataset[i], image_size);
                vector<uchar> image;
                image.resize(image_size);
                memcpy(image.data(), _dataset[i], image_size);
                //image.push_mem(_dataset[i], image_size);
                images.push_back(image);
            }
            //return _dataset;
            return images;
        } else {
            throw runtime_error("Cannot open file `" + full_path + "`!");
        }
    }
    
    static vector<uchar> read_mnist_labels(string full_path)
    {
        int number_of_labels;
        
        auto reverseInt = [](int i) {
            unsigned char c1, c2, c3, c4;
            c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
            return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
        };
        
        typedef unsigned char uchar;
        
        ifstream file(full_path, ios::binary);
        
        if(file.is_open()) {
            int magic_number = 0;
            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);
            
            if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");
            
            file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);
            
            uchar* _dataset = new uchar[number_of_labels];
            vector<uchar> labels;
            for(int i = 0; i < number_of_labels; i++) {
                file.read((char*)&_dataset[i], 1);
                labels.push_back(_dataset[i]);
            }
            return labels;
            //return _dataset;
        } else {
            throw runtime_error("Unable to open file `" + full_path + "`!");
        }
    }
    
    static vector<vector<uchar>> toY(vector<vector<uchar>>& images)
    {
        vector<vector<uchar>> out;
        out.resize(images.size());
        for(int i=0; i<images.size(); i++)
        {
            for(int j=0; j<1024; j++)
            {
                uchar R=images[i][j];
                uchar G=images[i][j+1024];
                uchar B=images[i][j+1024*2];
                
                //            Y = 0.299 R + 0.587 G + 0.114 B
                //            U = - 0.1687 R - 0.3313 G + 0.5 B + 128
                //            V = 0.5 R - 0.4187 G - 0.0813 B + 128
                
                Scalar Y = 0.299*R + 0.587*G + 0.114*B;
                //if(Y>255) cout<<"error"<<Y;
                out[i].push_back((uchar)Y);
            }
        }
        return out;
    }
    
    static vector<vector<uchar>> toRGB(vector<vector<uchar>>& images)
    {
        vector<vector<uchar>> out;
        out.resize(images.size());
        for(int i=0; i<images.size(); i++)
        {
            for(int j=0; j<1024; j++)
            {
                uchar R=images[i][j];
                uchar G=images[i][j+1024];
                uchar B=images[i][j+1024*2];
                
                out[i].push_back(R);
                out[i].push_back(G);
                out[i].push_back(B);
            }
        }
        return out;
    }
    
    static void readcifar10(vector<vector<uchar>>& images, vector<uchar>& labels, string full_path)
    {
        
        ifstream file(full_path, ios::binary);
        
        if(file.is_open()) {
            int num=10000;
            images.resize(num);
            labels.resize(num);
            for(int i = 0; i < num; i++) {
                file.read((char*)&labels[i], 1);
                images[i].resize(1024*3);
                file.read((char*)&images[i][0], images[i].size());
            }
            
            //images=toY(images);
            images=toRGB(images);
            //return _dataset;
        } else {
            throw runtime_error("Unable to open file `" + full_path + "`!");
        }
    }
    
    static void toOneHot(int n, int index, vector<Scalar> &out)
    {
        out.resize(n);
        for(int i=0; i<n; i++)
        {
            out[i]=(i==index?1:0);
        }
    }
    static void toOneHot(int n, int index, Scalar *pOut)
    {
        for(int i=0; i<n; i++)
        {
            pOut[i]=(i==index?1:0);
        }
    }
    static void toNHot(Scalar *pIn, int old_n, int scale, Scalar *pOut)
    {
        for(int i=0; i<old_n*scale; i++)
        {
            pOut[i]=pIn[i/scale];
        }
    }
    
    static void read_mnist(objTensor &x_train, objTensor &y_train, objTensor &x_test, objTensor &y_test, string dir)
    {
        cout<<"Loading mnist datasets..."<<endl;
        if(dir.length()>0) dir = dir + "/";
        vector<vector<uchar>> images=read_mnist_images(dir + "train-images-idx3-ubyte");
        vector<uchar> labels=read_mnist_labels(dir + "train-labels-idx1-ubyte");
        
        cout<<"images: "<<images.size()<<" labels: "<<labels.size()<<endl;
        
        //Linear mnist;
        int data_size=60000;
        int train_size=50000;//data_size*0.8;
        int imageSize = 28*28;
        x_train.fill(TShape({50000, 28, 28}));
        x_test.fill(TShape({10000, 28, 28}));
        y_train.fill(TShape(50000, 10));
        y_test.fill(TShape(10000, 10));
        for(int i=0; i<images.size(); i++)
        {
            if(i<train_size)
            {
                for(int j=0; j<images[i].size(); j++)
                {
                    x_train[i*imageSize+j]=images[i][j]/255.0;
                }
            }
            else
            {
                for(int j=0; j<images[i].size(); j++)
                {
                    x_test[(i-50000)*imageSize+j]=images[i][j]/255.0;
                }
            }
        }
        
        vector<Scalar> out;
        for(int i=0; i<data_size; i++)
        {
            toOneHot(10, labels[i], out);
            if(i<train_size)
            {
                for(int j=0; j<out.size(); j++)
                {
                    y_train[i*10+j]=out[j];
                }
            }
            else
            {
                for(int j=0; j<out.size(); j++)
                {
                    y_test[(i-50000)*10+j]=out[j];
                }
            }
        }
    }
    
    static void read_cifar10(objTensor &x_train, objTensor &y_train, objTensor &x_test, objTensor &y_test, string dir)
    {
        cout<<"Loading cifar10 datasets..."<<endl;
        if(dir.length()>0) dir = dir + "/";
        
        int imageSize = 32*32*3;
        x_train.fill(TShape({50000, 32,32,3}));
        x_test.fill(TShape({10000, 32,32,3}));
        y_train.fill(TShape(50000, 10));
        y_test.fill(TShape(10000, 10));
        for(int b=0; b<6; b++)
        {
            
            vector<vector<uchar>> images;
            vector<uchar> labels;
            string full_path;
            int offset = b;
            
            objTensor *x, *y;
            
            if(b<5)
            {
                full_path=dir+"data_batch_"+to_string(b+1)+".bin";
                x = &x_train;
                y = &y_train;
            }
            else
            {
                full_path=dir+"test_batch.bin";
                offset = 0;
                x = &x_test;
                y = &y_test;
            }
            
            readcifar10(images, labels, full_path);
            
            vector<Scalar> out;
            for(int i=0; i<images.size(); i++)
            {
                for(int j=0; j<images[i].size(); j++)
                {
                    (*x)[(offset*10000+i)*imageSize+j]=images[i][j]/255.0;
                }
                
                toOneHot(10, labels[i], out);
                for(int j=0; j<out.size(); j++)
                {
                    (*y)[(offset*10000+i)*10+j]=out[j];
                }
            }
            
        }
    }
};
    
} //namespace
#endif /* mnist_reader_hpp */
