## Solve the XOR problem

## let's build a sigmoidal activation
g <- function(x) { 1 / (1+ exp(-x)) }

## be careful about interpreting gprime!!!
gprime <- function(x) { y <- g(x); y * (1 - y) }

#curve(g, xlim=c(-3, 3))
#curve(gprime, xlim=c(-3, 3))
bias = -1

epsilon = 0.5

data = matrix( c(0, 0, bias, 0,
                 0, 1, bias, 1,
                 1, 0, bias, 1,
                 1, 1, bias, 0), 4, 4, byrow=TRUE)

inputs = t(data[,1:3])
targets = data[,4]


I=2 #excluding bias
J=2 #eclusing bias
K=1

W1 = matrix(runif(J*(I+1)), J, I+1) #+1 for bias
W2 = matrix(runif(K*(J+1)), K, J+1)


## "ideal weights"
ideal <- FALSE
if (ideal) {
  W1 = rbind( c(4, 4, 6),
             c(5, 5, 2))
  W2 = rbind( c(-8, 8, 4))
}

y_j = matrix(0, J+1, 1)             ## outputs of hidden units
delta_j = rep(0,J)                  ## delta for hidden units

nepoch = 20000
errors = rep(0, nepoch)

for (epoch in 1:nepoch) {

  ## accumulate errors for weight matrices
  DW1 = matrix(0, J, I+1)
  DW2 = matrix(0, K, J+1)
  epoch_err = 0.0


  for (i in 1:ncol(inputs)) {
    
    ## Step 1. Forward propagation activity, adding
    ## bias activity along the way.

    ## 1a - input to hidden
    y_i = inputs[,i,drop=FALSE] # keep as col vector
    
    ## input to hidden
    a_j = W1 %*% y_i

    for (q in 1:J) {
      y_j[q] = g(a_j[q])
    }
    y_j[J+1] = bias
    

    ## 1b - hidden to output

    a_k = W2 %*% y_j
    y_k = g(a_k)
    ##cat(sprintf("%.3f %.3f %.3f %.3f o %.3f %.3f\n", a_j[1,1], a_j[2,1], y_j[1,1], y_j[2,1], a_k, y_k))

    ## 1c - compare output to target
    t_k = targets[i]
    error = sum(0.5 * (t_k - y_k)^2)
    epoch_err = epoch_err + error


    ## Step 2.  Back propagate activity, calculating
    ## errors and dw along the way.
    
    
    ## 2a - output to hidden
    delta_k = gprime(a_k) * (t_k - y_k)
    for (q in 1:(J+1)) {
      for (r in 1:K) {
        DW2[r,q] = DW2[r,q] + y_j[q] * delta_k[r]
      }
    }

    ## 2b - calculate delta for hidden layer

    for (q in 1:J) {
      delta_j[q] = gprime(a_j[q]) * delta_k[1] * W2[1,q]
    }

    ## 2c - calculate error for input to hidden weights    
    for (p in 1:(I+1)) {
      for (q in 1:J) {
        DW1[q,p] = DW1[q,p] + delta_j[q] * y_i[p]
      }
    }
  }

  

  ## end of an epoch - now update weights
  errors[epoch] = epoch_err
  if ((epoch %%50)==0) {
    print(epoch_err)
  }
  W2 = W2 + (epsilon*DW2)
  W1 = W1 + (epsilon*DW1)
}

plot(errors)



print_activations <-  function() {
  ## Helper function to show state of all the units in the network.
  y_j = matrix(0, J+1, 1)
  
  n_inputs = ncol(inputs)
  ncol = I + J + K
  activations = matrix(0, n_inputs, ncol)
  for (i in 1:n_inputs) {
    y_i = inputs[,i, drop=FALSE]
    a_j = W1 %*% y_i
    for (q in 1:J) {
      y_j[q] = g(a_j[q])
    }
    y_j[J+1] = bias

    a_k = W2 %*% y_j
    y_k = g(a_k)
    ##browser()
    activations[i,] = c( y_i[1:I], y_j[1:J], y_k)
  }
  activations
}

print_activations()
