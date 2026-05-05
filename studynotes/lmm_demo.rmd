
### 4.1 LMM demo: the Pigs
```{r}
library(Pmisc)
library(tidyverse)

# load data from internet
ldaUrl = "http://www.maths.lancs.ac.uk/~diggle/lda/Datasets/lda.dat"
(ldaFile = Pmisc::downloadIfOld(ldaUrl))
pigs = t(read.table(ldaFile, skip = 269, nrow = 48, header = T))
# t(), transpose, lay out the data (weight of pigs over days) vertically.  

dim(pigs) # 48 pigs over wieghted for 9 days.


#?reshape


pigsLong = reshape(as.data.frame(t(pigs)),
                   varying = list(y = rownames(pigs)), 
                   direction = "long",
                   v.names = "y")

dim(pigsLong) # 423 records, each has 3 variables.


library("nlme")
pigsLme = lme(y ~ time, random = ~1 | id, data = pigsLong)
summary(pigsLme) 
# returns a model summary of coef estimates, 
# and the dsitribution of the random effect, and the AIC/BIC/logLik 


# distribution of the random effect 
hist(pigsLme$coef$random$id, main = "Histogram of RF(id)")


# install.packages("Pmisc", repos = "http://r-forge.r-project.org")

knitr::kable(Pmisc::lmeTable(pigsLme), digits = 4, escape = FALSE)
# the lmeTable function in the Pmisc package summarize 
# all useful info for LMM you would need. 


```


### 4.2 Interpretation: London School 
```{r echo=FALSE, fig.align='center'}

knitr::include_graphics("slide_1.png")

```


### 4.3 Protein in cows' milk      


```{r}
# load the data 
cowStart = c(barley = 101, mixed=155, lupins = 213)
cowLen = c( barley = 25, mixed=27, lupins = 27)
cows = mapply(function(start, len, diet){
  res = scan(ldaFile, skip=start-1, nlines=len*2, 
             quiet=TRUE)
  res = as.data.frame(matrix(res, nrow=len, 
                             byrow=TRUE))
  res$diet = diet
  res}, start=cowStart, len=cowLen,
  diet=names(cowStart), SIMPLIFY=FALSE)
cows = do.call(rbind, cows)
dim(cows)
glimpse(cows)


# reshape the data to long format
cowLong = reshape(cows, 
                  direction='long', v.names='protein',
                  varying = paste0("V", 1:19), 
                  times = 0:18)
cowLong = cowLong[cowLong$protein > 0,]
cowLong$diet = factor(cowLong$diet) # factorize diet
dim(cowLong)
glimpse(cowLong)
```


#### 4.3.1 Consider time as a factor...
```{r}
# fit the LMM, where we set time as a factor. why? 
cowLme = lme(protein ~ factor(time) +
               factor(diet), random = ~1 |
               id, data = cowLong)
knitr::kable(Pmisc::lmeTable(cowLme), digits = 4, escape = FALSE)

# plot the data, no obvious trend, and there is only
# level-difference among different treatments. 
plot(cowLme$data$time, cowLme$fitted[,'fixed'],
     col=as.integer(cowLme$data$diet), xlab='time', 
     ylab='weight')
legend("topright", 
       legend = c("Barley", "lupins", "mixed"), 
       col = c(1,2,3), pch = 1)

```


#### 4.3.2 Treatment-by-time model: time interacts treatment... 

```{r}
cowLong$t = factor(cowLong$time)
cowLmeInt = lme(protein ~ t * diet, random = ~1 | id, data = cowLong)

knitr::kable(Pmisc::lmeTable(cowLmeInt), digits = 4, escape = FALSE)

# if we takethe interaction into consideration, 
# then there is more than a by-level effect. 
plot(cowLme$data$time, cowLmeInt$fitted[,'fixed'],
     col=as.integer(cowLmeInt$data$diet), xlab='time', 
     ylab='weight')
legend("topright", 
       legend = c("Barley", "lupins", "mixed"), 
       col = c(1,2,3), pch = 1)


```




#### 4.3.3 Which one is better? LR test

```{r}

# The defult method is REML. But, to use LR test, 
# we must use the ML. Because REML does not calculate the maximum loglike. 
cowLmeMl = nlme::lme(as.formula(cowLme$call$fixed), 
            cowLme$data,as.formula(cowLme$call$random), 
            method = "ML")
cowLmeIntMl = nlme::lme(as.formula(cowLmeInt$call$fixed),
                        cowLmeInt$data,
                        as.formula(cowLmeInt$call$random), 
                        method = "ML")


lmtest::lrtest(cowLmeIntMl, cowLmeMl)

```



#### 4.3.4 Contrast Model 

```{r}
# fit the constrast model, with the 2nd order time terms
cowLong$tLupins = cowLong$tMixed = cowLong$time
cowLong$tLupins[cowLong$diet != "lupins"] = 0
cowLong$tMixed[cowLong$diet != "mixed"] = 0
cowLong$t = factor(cowLong$time)
cowLong$tl2 = cowLong$tLupins^2
cowLong$tm2 = cowLong$tMixed^2
cowLmeContrast = lme(protein ~ t + diet + 
                       tLupins + tMixed +
                       + tl2 + tm2, random = ~1 | id, 
                     data = cowLong)

knitr::kable(Pmisc::lmeTable(cowLmeContrast), digits = 4, 
             escape = FALSE)



# LR test for the constract models: with vs without the 
# 2nd order time terms
cowLmeContrastMl = nlme::lme(as.formula(cowLmeContrast$call$fixed),
                             cowLmeContrast$data,
                as.formula(cowLmeContrast$call$random),
                method = "ML")
cowLmeContrastLineMl = nlme::lme(protein ~ t + diet + 
                                   tLupins + tMixed, 
                                 cowLmeContrast$data,
                as.formula(cowLmeContrast$call$random),
                method = "ML")
lmtest::lrtest(cowLmeMl, cowLmeContrastLineMl, 
               cowLmeContrastMl)
```




### 4.4 School leaver's data 

```{r}
sUrl = "http://www.bristol.ac.uk/cmm/media/migrated/jsp.zip"
(schoolFile = Pmisc::downloadIfOld(sUrl))

school = read.fwf(grep("DAT$", schoolFile, 
                       value = TRUE),
                  widths = c(2, 1, 1, 1, 2, 4, 2, 2, 1), 
                  col.names = c("school", "class", 
                                "gender","socialClass","ravensTest",
                                "student", "english", "math", "year"))

school$socialClass = factor(school$socialClass, 
                            labels = c("I","II", "IIIn", "IIIm", 
                                       "IV", "V", "longUnemp", 
                                       "currUnemp", "absent"))

school$gender = factor(school$gender, labels = c("f", "m"))

school[1:12, c("school", "class", "student", "year", 
               "gender","math", "english"), ]

```



#### 4.4.1 Multi-level models 

```{r}
# fit the multi-level lmm
schoolLme = lme(math ~ gender + socialClass, 
                random = ~1 | school/class/student, 
                data = school)
```


```{r}
# the following SD are relative to tau...
summary(schoolLme$modelStruct)

# and the tau is: 
schoolLme$sigma

# model estimates for the fixed effects 
knitr::kable(summary(schoolLme)$tTable[, -3], 
             digits = 3)

knitr::kable(Pmisc::lmeTable(schoolLme), digits = 4, escape = FALSE)

```




#### 4.4.2 Random Slope Models 

```{r}
cowLmeRs = lme(protein ~ t + diet + tLupins + tMixed,
               random = ~ 1 + time | id, data=cowLong)


summary(cowLmeRs$modelStruct)

cowLmeRs$sigma # this is our tau

theTable = summary(cowLmeRs)$tTable[, -3]
knitr::kable(theTable[grep("^t[1-5]", 
                           rownames(theTable),
                           invert = TRUE), ], 
             digits = 3)


knitr::kable(Pmisc::lmeTable(cowLmeRs), digits = 4, escape = FALSE)



```





















