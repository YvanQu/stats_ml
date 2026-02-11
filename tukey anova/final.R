rawdata <- read.table('TramplingData.dat', header=T)
names(rawdata)


rawdata$DISTURB <- factor(rawdata$DISTURB, levels=c('exclusion', 'normal', '3x'))

rawdata$MONTH <- factor(rawdata$MONTH, levels=c('jun', 'sep', 'dec'))

nested <- rawdata

rawdata$PLOT <- factor(rawdata$PLOT)

nested$PLOT = nested$PLOT %% 2

nested$PLOT <- factor(nested$PLOT)

fligner.test(PCTCOV~factor((DISTURB:PLOT):MONTH), data=nested)
nested.vars <- with(nested, tapply(PCTCOV, DISTURB:PLOT:MONTH, var, na.rm=T))
nested.means <- with(nested, tapply(PCTCOV, DISTURB:PLOT:MONTH, mean, na.rm=T))
plot(nested.vars, nested.means)
sort(nested.vars)
max(nested.vars)/sum(nested.vars)
1/(1+(2/qf(1-.05/3, 1, 2)))

barplot(with(nested,tapply(PCTCOV,list(PLOT,DISTURB),mean)),beside=T, ylim=c(0,52), legend=c(levels(nested$PLOT)),col=c(1, 'grey', 0))
barplot(with(nested,tapply(PCTCOV,list(MONTH,DISTURB),mean)),beside=T, ylim=c(0,52), legend=c(levels(nested$MONTH)),col=c(1, 'grey', 0))
with(rawdata, boxplot(PCTCOV~factor(DISTURB:MONTH)))

nested.aov <- aov(PCTCOV~DISTURB+PLOT%in%DISTURB + DISTURB*MONTH + PLOT%in%DISTURB*MONTH, nested)
summary(nested.aov)

1-pf(11.01,2,6)
1-pf(7.37,2,3)
1-pf(5.18,4,6)

Q = qtukey(.95, 3, 12)
123.8/(5.7/Q)^2
Q*sqrt(123.8/54)
5.7127+3.7083
5.7127+12.04166
5.7127+8.3333333

TukeyHSD(nested.aov, "DISTURB")