//
//  Archive.hpp
//  objTensor
//
//  Created by Phyar on 2019/6/19.
//  Copyright © 2019 Phyar. All rights reserved.
//

#ifndef Archive_hpp
#define Archive_hpp

#include <stdio.h>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>

#include <fstream>

namespace objt {

using namespace std;

enum ArchiveDir
{
    DIR_LEFT=0,
    DIR_RIGHT,
};
struct Archive
{
    vector<char> data;
    int dir=DIR_LEFT;
    int version=0;
    int pointer=0;
    
    Archive()
    {
        *this & version;
    }
    bool left()
    {
        return dir==DIR_LEFT;
    }
    void save(string fileName)
    {
        std::ofstream t(fileName);
        t.write(&data[0], data.size());
    }
    void load(string fileName)
    {
        std::ifstream t;
        int length;
        t.open(fileName);      // open input file
        t.seekg(0, std::ios::end);    // go to the end
        length = (int)t.tellg();           // report location (this is the length)
        t.seekg(0, std::ios::beg);    // go back to the beginning
        data.resize(length);
        t.read(&data[0], length);       // read the whole file into the buffer
        t.close();                    // close file handle
        
        dir=DIR_RIGHT;
        pointer=0;
        *this & version;
    }
    
    template<class T>
    Archive& save_basic(const T &a)
    {
        size_t old_size = data.size();
        data.resize(data.size()+sizeof(T));
        memcpy(&data[old_size], (char*)&a, sizeof(T));
        return (*this);
    }
    template<class T>
    Archive& load_basic(T &a)
    {
        memcpy((char*)&a, &data[pointer], sizeof(T));
        pointer+=sizeof(T);
        return (*this);
    }
    
    char* getPointer()
    {
        return &data[pointer];
    }
    void movePointer(int n)
    {
        pointer+=n;
    }
    
    template<class T>
    Archive& basic(T &t)
    {
        if(left()) save_basic(t);
        else load_basic(t);
        return *this;
    }
    Archive& operator&(int &t)
    {
        return basic(t);
    }
    Archive& operator&(size_t &t)
    {
        return basic(t);
    }
    Archive& operator&(float &t)
    {
        return basic(t);
    }
    Archive& operator&(double &t)
    {
        return basic(t);
    }
    
    template<class T>
    Archive& operator&(vector<T>& a)
    {
        if(left())
        {
            size_t size=a.size();
            *this & size;
            
            for(int i=0; i<a.size(); i++) (*this) & a[i];
        }
        else
        {
            size_t size=0;
            *this & size;
            
            a.resize(size);
            for(int i=0; i<a.size(); i++) (*this) & a[i];
        }
        
        return *this;
    }
    
    
    
};

} //namespace
#endif /* Archive_hpp */
