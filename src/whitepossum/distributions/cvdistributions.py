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
    This uses uniform() to generate a sample for the exponential distribution.

    x = - ( ln(y) / alpha) 

    Arguments:
    l
      Lamda value to pass to the function.
    """
    if  l < 0:
       raise ValueError("Lamda must be greater than 0")


    # get one y-value
    y = uniform()
    x = -( np.log(y) / l)
    return x

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