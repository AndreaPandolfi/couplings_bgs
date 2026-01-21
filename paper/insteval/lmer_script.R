require(lme4)
df = InstEval

out = lmer("y ~ 1 + (1|d) + (1|s) + (1|studage) + (1|lectage) + (1|service)", data = df)

t = as.data.frame(VarCorr(out))
names = t$grp
tau = 1/(t$vcov)

names(tau)=names
tau


# s & d
# s         d         Residual 
# 9.4149102 3.6531701 0.7208871 

# s & d & studage & lectage
# s           d         lectage     studage       Residual 
# 9.3592072   3.7295522 128.4982743 338.3219925   0.7226101 

# s           d             lectage      studage      service       Residual 
# 9.4079960   3.7436722     143.4601876  392.8648024  398.4144663   0.7227555 
