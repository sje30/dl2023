##  Perceptron learning rule.
## master: ~/txt/cam/teaching/dl/code/

data = read.table("xor.dat")
ninputs = nrow(data)
targets = data[,3]
inputs = cbind( as.matrix(data[,1:2]), rep(-1,ninputs))


epsilon = 0.03


set.seed(100)

plotdata = function() {
  plot(data[,1], data[,2], pch=19,
       xlab="x[1]", ylab="x[2]",
       col=ifelse(data[,3]==1, "red", "blue"), asp=1)

  a = wts[3]/wts[2]                     #slope
  b = -wts[1]/wts[2]                    #intercept
  abline(a=a, b=b)
}


##wts = runif(Nin)
## start with a deliberately poor set of initial weights.
wts = c(1, 1, 1.5)
plotdata()


for (epoch in 1:100) {
  order = sample(ninputs);
  error = 0;
  for (i in order) {
    x = inputs[i,]
    t = targets[i]
    y = sum ( x * wts )
    y = y > 0;
    error = error + (0.5*(t - y)^2);
    dw = epsilon * ( t - y ) * x
    wts = wts + dw;
  }
  title =sprintf('epoch %d error %.3f\n', epoch, error)
  plotdata()
  title(main=title)
  Sys.sleep(0.1)
  if (error == 0)
    break;
}

plotdata()
