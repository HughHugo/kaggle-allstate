library(readr)
library(data.table)
library(Rtsne)
library(Metrics)

print(getwd())
path <- "../../cache/layer_0/"
train <- read_csv(paste0(path,"train_tfidf.csv"))
train <- as.data.frame(train)
dim(train)
test <- read_csv(paste0(path,"test_tfidf.csv"))
test <- as.data.frame(test)
dim(test)
TsneDim <- 3 # 2 or 3
tmp <- as.factor(train$loss)
train$id <-NULL
test$id <- NULL
train$loss <- NULL
x <- rbind(train,test)
dim(x)
N <- nrow(train)
rm(train)
rm(test)
gc()

print("constructing t-sne")
set.seed(6174)
tsne <- Rtsne(x, dims = TsneDim, perplexity = 50, initial_dims = 50,
theta = 0.5, check_duplicates = F, pca = T, verbose=TRUE, max_iter = 500)

colors = rainbow(length(unique(tmp)))
names(colors) = unique(tmp)



train <- tsne$Y[1:N,]
test <- tsne$Y[(N+1):nrow(tsne$Y),]
if(TsneDim==2){
colnames(train) <- c("dim1","dim2")
colnames(test) <- c("dim1","dim2")
}
if(TsneDim==3){
colnames(train) <- c("dim1","dim2","dim3")
colnames(test) <- c("dim1","dim2","dim3")
}

train <- as.data.frame(train)
test <- as.data.frame(test)
for(i in 1:3){
train[,i] <- as.character(train[,i])
test[,i] <- as.character(test[,i])
}
gc()

write_csv(train,paste0(path,"tfidf_train_tsne.csv"),col_names=T)
write_csv(test,paste0(path,"tfidf_test_tsne.csv"),col_names=T)
