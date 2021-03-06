library(surveillance)
library(ggplot2)
library(patchwork)
library(plotrix)
library(sp)
library(xtable)


EM_algorithm<-function(confirmed_real){
  dmax = length(confirmed_real)
  
  #PMF of the incubation time is an interval censored lognorm distribution
  #with mean 7 truncated at 40.
  inc.pmf = c(0,(plnorm(1:dmax,mu,sigma) - plnorm(0:(dmax-1),mu,sigma))/plnorm(dmax,mu,sigma))
  Y = table(factor(rep(1:dmax, times = confirmed_real),levels = 1:dmax))
  #Convert Y to an sts object
  Ysts <- sts(Y)
  
  #Call non-parametric back-projection function with hook function but
  #without bootstrapped confidence intervals
  bpnp.control = list(k=2,eps=rep(0.005,2),iter.max=rep(250,2),B=-1,verbose=TRUE)
  #Fast C version (use argument: eq3a.method="R")!
  sts.bp = backprojNP(Ysts, incu.pmf=inc.pmf,
                      control=modifyList(bpnp.control,list(eq3a.method="R")))
  infection_est = upperbound(sts.bp)
  
  #Do the convolution for the expectation
  confirmed_est = matrix(0,ncol=ncol(sts.bp),nrow=nrow(sts.bp))
  #Loop over all series
  for (j in 1:ncol(sts.bp)) {
    #Loop over all time points
    for (t in 1:nrow(sts.bp)) {
      #Convolution, note support of inc.pmf starts at zero (move idx by 1)
      i <- seq_len(t)
      confirmed_est[t,j] <- sum(inc.pmf[t-i+1] * infection_est[i,j],na.rm=TRUE)
    }
  }
  res = list(confirmed_est, infection_est)
  return (res)
}


# Bootstrap采样
boostrap <- function(dataset){
  #Non-parametric back-projection including boostrap CIs. B=1000 is only
  #used for illustration in the documentation example
  #In practice use a realistic value of B=1000 or more.
  diagnosed_real = dataset
  #PMF of the incubation time is an interval censored lognorm distribution
  #with mean 7 truncated at 40.
  inc.pmf = c(0,(plnorm(1:dmax,mu,sigma) - plnorm(0:(dmax-1),mu,sigma))/plnorm(dmax,mu,sigma))
  Y = table(factor(rep(1:dmax, times = diagnosed_real),levels = 1:dmax))
  Ysts = sts(Y)
  bpnp.control = list(k=2,eps=rep(0.005,2),iter.max=rep(250,2),B=-1,verbose=TRUE)
  bpnp.control2 = modifyList(bpnp.control, list(hookFun=NULL,k=2,B=50,eq3a.method="R"))
  sts.bp = backprojNP(Ysts, incu.pmf=inc.pmf, control=bpnp.control2)
  lcbound = sts.bp@ci[1,,1]
  ucbound = sts.bp@ci[2,,1]
  res = list(lcbound,ucbound)
  return (res)
}


setwd('D:/2019-nCov/ML_v2')
# 参数
mu = log(7.2)
sigma = log(15.1/7.2)/1.6449


# 导入数据
# region_name = c('India', 'Brazil', 'Russia', 'Mexico', 'Colombia', 'Peru', 'Iran', 'South Africa', 'Pakistan')
region_name = 'India'
disease_info = read.csv(paste0("./data/disease_hist/",region_name,".csv"), header = TRUE, encoding='UTF-8') 
disease_info$date = as.Date(disease_info$date)


test_num = c(30, 20, 10)
estimate = vector(mode='list', 3)
for(i in 1:3){
  sample_num = nrow(disease_info)
  temp = disease_info[1:(sample_num-test_num[i]),]$confirmed_delta
  estimate[[i]] = EM_algorithm(temp)
}


area = vector(mode='list', 3)
for(i in 1:3){
  temp = data.frame(date=disease_info$date, confirmed_delta_real=disease_info$confirmed_delta,
                    confirmed_delta_est=c(estimate[[i]][[1]], rep(10, times=test_num[i])), 
                    infection_delta_est=c(estimate[[i]][[2]], rep(10, times=test_num[i])))
  colnames(temp) = c("date", "confirmed_delta_real", "confirmed_delta_est", "infection_delta_est")
  area[[i]] = temp
}


col = c()
for(i in 1:14){
  col = c(col,paste0('infection_rate_', i))
}

for(i in 1:3){
  dmax = nrow(area[[i]])
  infection_rate = vector(mode='list', 14)
  for(window_size in 1:14){
    infection_rate[[window_size]] = 0
    for(j in (window_size+1):dmax){
      temp = sum(area[[i]]$infection_delta_est[(j-window_size):j])
      infection_rate[[window_size]] = c(infection_rate[[window_size]], area[[i]]$infection_delta_est[j] / temp) 
    }
    area[[i]] = cbind(area[[i]], c(rep(0, times=window_size-1),infection_rate[[window_size]]))
  }
  colnames(area[[i]]) = c("date", "confirmed_delta_real", "confirmed_delta_est", "infection_delta_est", col)
  write.csv(area[[i]], paste0('./result/disease_pre/10/',region_name, '_', test_num[i], '.csv'), row.names = FALSE) 
}

