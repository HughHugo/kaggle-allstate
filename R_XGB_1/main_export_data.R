# This script is loosely ported from Python script by modkzs, Misfyre and others
# https://www.kaggle.com/misfyre/allstate-claims-severity/encoding-feature-comb-modkzs-1108-72665
# However it gets a little better results maybe due to
# different realisations of scaling and Box Cox transformations in R and Python
#
# Please run on your local to get high scores

# ----------------------------------- load libraries ------------------------------
#data
library(readr)
library(data.table)
#statistics
library(Metrics)
library(scales)
library(Hmisc)
#data sciense
library(caret)
library(xgboost)
library(e1071)
#strings
library(stringr)
library(forecast)
# ----------------------------------- tools & variables ----------------------------

# fair objective 2 for XGBoost

amo.fairobj2 <- function(preds, dtrain) {

  labels <- getinfo(dtrain, "label")
  con <- 2
  x <- preds - labels
  grad <- con * x / (abs(x) + con)
  hess <- con ^ 2 / (abs(x) + con) ^ 2

  return(list(grad = grad, hess = hess))

}

# MAE Metric for XGBoost

amm_mae <- function(preds
                    , dtrain) {

  labels <- xgboost::getinfo(dtrain, "label")
  elab <- as.numeric(labels)
  epreds <- as.numeric(preds)
  err <- mae(elab, epreds)

  return(list(metric = "amm_mae", value = err))

}

# shift applying to the dependent variable ('loss')

shft <- 200

# names of categorical features for feature engineering

new.cat.raw <- c("cat80","cat87","cat57","cat12","cat79","cat10","cat7","cat89","cat2","cat72",
             "cat81","cat11","cat1","cat13","cat9","cat3","cat16","cat90","cat23","cat36",
             "cat73","cat103","cat40","cat28","cat111","cat6","cat76","cat50","cat5",
             "cat4","cat14","cat38","cat24","cat82","cat25")

new.cat.raw <- merge(data.frame(f1 = new.cat.raw), data.frame(f2 = new.cat.raw))
new.cat.raw <- data.table(new.cat.raw)
new.cat.raw <- new.cat.raw[as.integer(str_extract_all(f1, "[0-9]+")) >
    as.integer(str_extract_all(f2, "[0-9]+"))]

new.cat.raw[, f1 := as.character(f1)]
new.cat.raw[, f2 := as.character(f2)]


#-------------------------------------- load data -----------------------------------------

# load data
tr <- fread("~/Documents/github/kaggle-allstate/input/train.csv", showProgress = TRUE)
ts <- fread("~/Documents/github/kaggle-allstate/input/test.csv", showProgress = TRUE)

# merge to single data set
ttl <- rbind(tr, ts, fill=TRUE)
remove(tr)
remove(ts)



#-------------------------------------- feature engineering -------------------------------
# new categorical features
# a bit straightforward approach so you can try to improve it

for (f in 1:nrow(new.cat.raw)) {

  f1 <- new.cat.raw[f, f1]
  f2 <- new.cat.raw[f, f2]

  ttl[, eval(as.name(paste(f1, f2, sep = "_"))) := paste0(ttl[, eval(as.name(f1))], ttl[, eval(as.name(f2))])]
}

# categorical features to range ones
# was very slow - must be much faster and much R-approach now

utf.A <- utf8ToInt("A")

for (f in colnames(ttl)[colnames(ttl) %like% "^cat"]) {

  ttl[, eval(as.name(f)) := mapply(function(x, id) {

    if (id == 1) print(f)

    x <- utf8ToInt(x)
    ln <- length(x)

    x <- (x - utf.A + 1) * 26 ^ (ln - 1:ln - 1)
    x <- sum(x)
    x

  }, eval(as.name(f)), .I)]

}

# remove skewness

for (f in colnames(ttl)[colnames(ttl) %like% "^cont"]) {

  tst <- e1071::skewness(ttl[, eval(as.name(f))])
  if (tst > .25) {
    if (is.na(ttl[, BoxCoxTrans(eval(as.name(f)))$lambda])) next
    ttl[, eval(as.name(f)) := BoxCox(eval(as.name(f)), BoxCoxTrans(eval(as.name(f)))$lambda)]
  }
}

# scale

for (f in colnames(ttl)[colnames(ttl) %like% "^cont"]) {
  ttl[, eval(as.name(f)) := scale(eval(as.name(f)))]
}

# save
#save(ttl, file="~/Documents/github/kaggle-allstate/cache/ttl.pipe.Rda")
#load("~/Documents/github/kaggle-allstate/cache/ttl.pipe.Rda")
write_csv(ttl, "~/Documents/github/kaggle-allstate/R_XGB_1/ttl.csv")
write.csv(ttl, "~/Documents/github/kaggle-allstate/R_XGB_1/ttl.csv", row.names = F)


save_ttl = data.frame(ttl)
write_csv(save_ttl, "~/Documents/github/kaggle-allstate/R_XGB_1/ttl.csv")
