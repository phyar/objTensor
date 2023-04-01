//
//  Parallel.hpp
//  objTensor
//
//  Created by Phyar on 2019/8/31.
//  Copyright © 2019 Phyar. All rights reserved.
//

#ifndef PARALLEL_H
#define PARALLEL_H

#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
//#include <condition_variable>
#include <future>
#include <map>
#include <string>

namespace objt{

using namespace std;

struct blocked_range {
    typedef size_t const_iterator;
    blocked_range(size_t begin, size_t end) : begin_(begin), end_(end) {}
    blocked_range(size_t begin, size_t end, size_t i) : begin_(begin), end_(end), tid_(i) {}
    
    const_iterator begin() const { return begin_; }
    const_iterator end() const { return end_; }
    
public:
    size_t begin_;
    size_t end_;
    size_t tid_;
};

template <class T, typename Func>
T parallel_for(size_t begin,
               size_t end,
               const Func &f) {
    assert(end >= begin);
    size_t nthreads  = std::thread::hardware_concurrency();
    size_t blockSize = (end - begin) / nthreads;
    if (blockSize * nthreads < end - begin) blockSize++;
    
    std::vector<std::future<T> > futures;
    
    size_t blockBegin            = begin;
    size_t blockEnd              = blockBegin + blockSize;
    if (blockEnd > end) blockEnd = end;
    size_t tid=0;
    
    for (size_t i = 0; i < nthreads; i++) {
        tid=i;
        futures.push_back(
                          std::move(std::async(std::launch::async, [blockBegin, blockEnd, tid, &f] {
            T r=f(blocked_range(blockBegin, blockEnd, tid));
            return r;
        })));
        
        blockBegin += blockSize;
        blockEnd = blockBegin + blockSize;
        if (blockBegin >= end) break;
        if (blockEnd > end) blockEnd = end;
    }
    
    for (auto &future : futures) future.wait();
    
    T sum=0.0;
    for (auto &future : futures) sum+=future.get();
    
    return sum;
}

template <typename Func>
int parallel_for(size_t begin,
                 size_t end,
                 const Func &f,
                 size_t num_threads) {
    assert(end >= begin);
    size_t nthreads  = num_threads;
    if(nthreads == 0) nthreads = std::thread::hardware_concurrency()/2;
    size_t blockSize = (end - begin) / nthreads;
    if (blockSize * nthreads < end - begin) blockSize++;
    
    std::vector<std::future<void> > futures;
    
    size_t blockBegin            = begin;
    size_t blockEnd              = blockBegin + blockSize;
    if (blockEnd > end) blockEnd = end;
    size_t tid=0;
    
    for (size_t i = 0; i < nthreads; i++) {
        tid=i;
        futures.push_back(
                          std::move(std::async(std::launch::async, [blockBegin, blockEnd, tid, &f] {
            f(blocked_range(blockBegin, blockEnd, tid));
        })));
        
        blockBegin += blockSize;
        blockEnd = blockBegin + blockSize;
        if (blockBegin >= end) break;
        if (blockEnd > end) blockEnd = end;
    }
    
    for (auto &future : futures) future.wait();
    
    return (int)nthreads;
}

template <typename Func>
int for_i(size_t begin, size_t end, const Func &f, size_t max_threads=0)
{
    if(max_threads==1)
    {
        for (size_t i = begin; i < end; ++i) {
            f(i, 0);
        }
        return 1;
    }
    else
    {
        return parallel_for(begin, end,
                            [&](const blocked_range &r) {
                                for (size_t i = r.begin(); i < r.end(); i++) {
                                    f(i, r.tid_);
                                }
                            }, max_threads);
    }
    return 0;
}

template <typename Func>
int exe_all(const Func &f, size_t num_threads=0)
{
    size_t nthreads  = num_threads;
    if(nthreads == 0) nthreads = std::thread::hardware_concurrency()/2;
    if(nthreads == 1)
    {
        f(0);
        return (int)nthreads;
    }
    
    std::vector<std::future<void> > futures;
    
    for (int i = 0; i < nthreads; i++) {
        int tid=i;
        futures.push_back(std::move(std::async(std::launch::async, [tid, &f] {
            f(tid);
        })));
        
    }
    
    for (auto &future : futures) future.wait();
    
    return (int)nthreads;
}

struct MutexLocker : std::unique_lock<std::mutex>
{
    //std::mutex mutex;
    MutexLocker(uint64_t m_id):std::unique_lock<std::mutex>(*MutexLocker::getMutex(m_id))
    {
    }
    static std::mutex* getMutex(uint64_t m_id)
    {
        static map<uint64_t, std::mutex*> mutexPool;
        if(mutexPool.count(m_id)==0)
        {
            mutexPool[m_id] = new std::mutex();
        }
        return mutexPool[m_id];
    }
};

struct PtrMutex : public shared_ptr<std::mutex>
{
    PtrMutex():shared_ptr<std::mutex>(std::make_shared<std::mutex>())
    {
        
    }
};

struct ParaUtil
{
    static void setPriority(std::thread &t)
    {
        sched_param sch;
        int policy;
        pthread_getschedparam(t.native_handle(), &policy, &sch);
        //cout<<policy<<endl;
        sch.sched_priority = 99;//sched_get_priority_max(SCHED_FIFO);//20;
        if (pthread_setschedparam(t.native_handle(), SCHED_RR, &sch)) {
            std::cout << "Failed to setschedparam: " << std::strerror(errno) << '\n';
        }
    }
};
    
    
} //namespace
#endif
