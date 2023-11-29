def print_values(V, g):
  """
  A function that ptint the values.
  This function will draw our grid in ASCII, and inside each state 
  It will print the value corresponding to that state.
  
  Parameters
  ----------
  - `V`: A value dictionary
  - `g`: Grid 

  """ 
  
  for i in range(g.rows): # Loop over the grid rows
    print("---------------------------")
    for j in range(g.cols): # Loop over the columns in each row
      # Get the value for the current position [i,j]
      # we using default value of 0 (if we dont get any value, we get 0.)
      v = V.get((i,j), 0) 
      # check if the value is not negative (for print in appropirate formate)
      if v >= 0: 
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  """
  Very similar to the `print_values()` function. 
  This function is print the policy of our agant in the grid.
  in other words, in each state in the grid, what will be the policy?
  
  Parameters
  ----------
  - `P`: the policy (which action to do in this state?)
  - `g`: the grid world (the env)
  """

  for i in range(g.rows): # Loop over the grid rows
    print("---------------------------")
    for j in range(g.cols): # loop over the grig columns
      # given the state (i,j - location on the grid), give the the action based on the policy
      a = P.get((i,j), ' ')  
      print("  %s  |" % a, end="")
    print("")
