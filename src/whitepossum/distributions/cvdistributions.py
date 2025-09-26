import secrets
import numpy as np
def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53)
    # in [0, 1)
    return a + (b - a) * u


def exponentialdist(l):
    """
    This uses uniform() to generate l random sample for the exponential distribution.

    x = - ( ln(y) / alpha) 

    **Parameters**

    l
      Lamda value to pass to the function.
    """

    dist = []
    # start from 1, add one to l because lamda > 0
    for i in range(1, l+1):
      # get one y-value
      y = uniform()
      x = -( np.log(y) / i)
      dist.append(x)

    return dist

def poissondist(l):
    """
    This uses inverse transform sampling to compute a sample of the posson distibution. 

    **Parameters**

    l
      Lamda value to pass to the function.
    """

    dist = []



    

    return dist




# if __name__ == '__main__':
#    la = 10
   
#    print(exponentialdist(la))