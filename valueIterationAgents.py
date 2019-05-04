# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    

    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    
    state = mdp.getStates()
    self.old = self.values.copy()
    for i in range(0,iterations):
        
        for states in state:
            action = mdp.getPossibleActions(states)
            
            if (mdp.isTerminal(states)):
                continue 
            else:
               max_val= -float('inf') 
               for act in action:
                   q_val= self.getQValue(states,act)
                   max_val = max(max_val,q_val)
               self.values[states] = max_val
        self.old = self.values.copy()
                   
            
            
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    
    qval = 0 
    trans_probs = self.mdp.getTransitionStatesAndProbs(state, action)
    for future_state, transition in trans_probs :
        R =self.mdp.getReward(state, action, future_state)
        dis_val = self.discount * self.old[future_state]
        qval += transition * (R+dis_val)
    return qval
    
    util.raiseNotDefined()

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    pos_actions = self.mdp.getPossibleActions(state)
    pot_act = pos_actions
    act_max_val = -float('inf')
    policy = None
    for c in pot_act:
        act_val = self.getQValue(state,c)
        if act_val > act_max_val:
           act_max_val = act_val
           policy = c
    
    return policy 
    util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
