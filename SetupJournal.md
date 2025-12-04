# Journal to Record Setup Woes
- I first ran into this problem, see below. 

    ```md 
    Traceback (most recent call last):
      File "/Users/franklinho/Documents/Work, Research, Finance/Harvard/robobee3d/template/uprightmpc2.py", line 383, in <module>
        controlTest(upc, 500, useMPC=True, hlInterval=5)
      File "/Users/franklinho/Documents/Work, Research, Finance/Harvard/robobee3d/template/uprightmpc2.py", line 139, in controlTest
        uquad, log['accdes'][ti,:] = mdl.update(p, Rb, dq, pdes, dpdes, sdes)
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    TypeError: update(): incompatible function arguments. The following argument types are supported:
        1. (self: uprightmpc2py.UprightMPC2C, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 3]"], arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[6, 1]"], arg3: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"], arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"], arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"], arg6: typing.SupportsFloat | typing.SupportsIndex) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float32], "[6, 1]"]]
    
    Invoked with: <uprightmpc2py.UprightMPC2C object at 0x106ae7b70>, array([ 0.52  ,  0.    , -0.1275]), array([[ 0.8776, -0.2298, -0.4207],
           [ 0.    ,  0.8776, -0.4794],
           [ 0.4794,  0.4207,  0.7702]]), array([ 0.1  ,  0.   , -0.051,  0.   ,  0.   ,  0.   ]), array([0, 0, 0]), array([0., 0., 0.]), array([0, 0, 1])
    ```
  - seems like there is not enough arguments passed in `mdl.update(p.RB...)`
  - Hence, just feed in `actualT0 = -1` (only use if greater than 0) 


### OSQP update problem 
- Another problem was that osqp was too old (last run in `0.5.0`, while the new version is `1.X.X`)
- It really wants `q,l,u` to be fed in, so we just set it to 0 as some kind of default value. 
