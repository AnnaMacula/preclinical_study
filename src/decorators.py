#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:17:46 2022

@author: user
"""
import functools
from multiprocessing import Pool, cpu_count#, IMapIterator, starmapstar

from tqdm import tqdm


def at_lightspeed(in_parallel, pool=cpu_count(), quiet=True, description=""):
    
    def decorator(arg_gen):
    
        @functools.wraps(arg_gen)
        def wrapper(*args, **kwargs):
            
            # print(in_parallel.__name__)
            
            PARALLEL_ARGS_FUNZ=arg_gen(*args, **kwargs)
            if not quiet:     
                print("Multiprocessing engaged.")  
            
            try:
                desc=kwargs["description"]
            except:
                desc=description
                
            with Pool(pool) as p:           
                  TEMP = p.starmap(in_parallel, tqdm(PARALLEL_ARGS_FUNZ, total=len(PARALLEL_ARGS_FUNZ), desc=desc))
            
            if not quiet:    
                print("Multiprocessing completed.")
            
          
            return TEMP          
 
        return wrapper
    
    return decorator


def make_compatible(args_positions=(0,1)):
    
    # args_positions = args_positions or (0,1)

    def decorator(func):
       
        @functools.wraps(func)   
        def wrapper(*args):
            # print(func, args, args_positions)  
            return func(*[args[position] for position in args_positions])
        
        return wrapper
    
    return decorator    