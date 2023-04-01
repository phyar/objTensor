//
//  Util.h
//  Util
//
//  Created by Phyar on 2019/2/24.
//

#ifndef Util_hpp
#define Util_hpp

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
namespace objt{

#define TIME_TYPE double
inline TIME_TYPE GetTime()
{
    struct timeval tv;
    struct timezone tz;
    gettimeofday (&tv , &tz);
    TIME_TYPE ret=tv.tv_sec*1000+tv.tv_usec/1000.0;
    //printf("%u, %d\n", ret, tv.tv_sec);
    return ret;
}
struct TimeServer
{
    TIME_TYPE begin;
    TIME_TYPE last;
    TIME_TYPE now;
    int lastfps;
    int frames;
    float afps;
    bool fpsUpdated;
    TimeServer()
    {
        lastfps=0;//GetTime();//0;
        frames=0;
        afps=0;
        fpsUpdated=false;
        //Set();
    }
    void Set(){begin=last=GetTime();}
    TIME_TYPE GetElapsed()
    {
        now=GetTime();
        TIME_TYPE ret=(TIME_TYPE)(now-last);
        last=now;
        return ret;
    }
    float fps()
    {
        fpsUpdated=false;
        now=GetTime();
        if(now-lastfps>=1000)
        {
            afps=frames*1000.0f/(now-lastfps);
            frames=0;
            lastfps=now;
            fpsUpdated=true;
        }
        frames++;
        return afps;
    }
    bool getUpdated()
    {
        return fpsUpdated;
    }
    float getFps()
    {
        return afps;
    }
    int All(){    return (int)(GetTime()-begin);}
};

} //namespace
#endif /* Util_hpp */
