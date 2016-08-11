#!/usr/bin/env python                                                           
import sys
import numpy as np
print np.percentile(map(float, sys.stdin.readlines()),
                    np.linspace(0, 100, 11))
