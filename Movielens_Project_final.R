#This script will generate a prediction algorithm for movie ratings based on the 10M movielens dataset

#####################################
# Create edx set, validation set ####
#####################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
#Add 2 libraries to the default code as we will need them later
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("tinytex", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##################################
# Starting with my own code  #####
##################################

########################### Understand, Wrangling and Analysing data ############################

#----------------------------- Understanding and Wrangling ---

#Have a look at the dataset:
head(edx, 50)
dim(edx)

#How many different movies are in the dataset
n_distinct(edx$movieId)

#How many different users are in the dataset
n_distinct(edx$userId)

#Check if we have some 0 in the ratings or any NA in one of the columns:
edx %>% filter(rating == 0) %>% tally()
sum(is.na(edx))

#Split title column in two columns: "title" and "year" and adjust classes of col.
edx1 <- edx %>% extract(title, c("title","year"), regex = "^(.*)\\s[(](\\d{4})[)]$")
edx1 <- edx1 %>% mutate(year = as.integer(year), genres = as.factor(genres))
head(edx1)

#edx1 Dimensions:
dim(edx1)

#Better readability by bringing columns closer together which belong together
edx1 <- edx1[,c(2,5,6,7,1,3,4)]
head(edx1)

#----------------------------- Analysing data ---
# Set settings for Themes:
myTheme <- theme(axis.title = element_text(size = 14),
  axis.text.x = element_text(size = 10), 
  axis.text.y = element_text(size = 10))

#Visualising the Gaps of not rated movies by a graph of 100 random users and 
#100 random movies:
users <- sample(unique(edx1$userId), 100)
rafalib::mypar()
edx1 %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

#First insights: Some movies get more rated then others:
edx1 %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies") +
  labs(x = "Amount of ratings movies got", y="Amount of movies per 'amount of rating'-category") +
  myTheme

#First insights: Some users are rating more than others:
edx1 %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users") +
  labs(x = "Number of ratings users have done", y="Amount of users per 'amount of rating'-category") +
  myTheme

#Look at the distribution of the ratings:
edx1 %>% 
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "black") + 
  ggtitle("Distribution of Ratings") +
  labs(x = "Rating scale", y="Amount of entries per Rating option") +
  myTheme

################### Building Test-, Training sets and RMSE Function ##############

#----------------------------- Building Test- and Training sets ---

#Split edx1 dataset in a edx_train and edx_test set - use 15% of data for test set
set.seed(15, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = edx1$rating, times = 1, p = 0.15, list = FALSE)
edx_train <- edx1[-test_index,]
temp <- edx1[test_index,]

# Make sure userId and movieId in edx_test set are also in edx_train set
edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from edx_test set back into edx_train set
removed <- anti_join(temp, edx_test)
edx_train <- rbind(edx_train, removed)

#remove unnessecary objects
rm(test_index, temp, removed, edx1)

#----------------------------- Build RMSE function ---

#RMSE function:
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##################### Building the Algorithm #####################################

#----------------------------- Calculate mu and test against RMSE ---

#Calculating mu 
mu <- mean(edx_train$rating)
mu

#Calculate RMSE for 1st intermediate result based on mu:
Mu_rating <- edx_test %>% 
  mutate(pred = mu) %>%
  .$pred

RMSE_InterOne <- RMSE(Mu_rating, edx_test$rating)

#Visualize the RMSE result:
rmse_results <- tibble(method="Mu", RMSE = RMSE_InterOne, Improvement = 0, )
rmse_results %>% knitr::kable()

#----------------------------- Add movie bias (b_i) ---

#Calculating b_i (lineare regression would crash my computer so I do an approx.)
bi_avg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu), n = n())

#Calculate RMSE for 2nd intermediate result:
MuBi_rating <- edx_test %>% 
  left_join(bi_avg, by='movieId') %>% 
  mutate(pred = mu + b_i) %>%
  .$pred

RMSE_InterTwo <- RMSE(MuBi_rating, edx_test$rating)

#Improvement:
Imp <- RMSE_InterOne-RMSE_InterTwo

#Visualize the RMSE result:
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Mu, b_i",
                                     RMSE = RMSE_InterTwo, Improvement = Imp ))
rmse_results %>% knitr::kable()

#----------------------------- Add user bias (b_u) ---

#Calculating b_u
bu_avg <- edx_train %>% 
  left_join(bi_avg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#Calculate RMSE for 3rd intermediate result:
MuBiBu_rating <- edx_test %>% 
  left_join(bi_avg, by='movieId') %>% 
  left_join(bu_avg, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

RMSE_InterThree <- RMSE(MuBiBu_rating, edx_test$rating)
RMSE_InterThree

#Improvement:
Imp <- RMSE_InterTwo-RMSE_InterThree

#Visualize the RMSE result:
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Mu, b_i, b_u",
                                     RMSE = RMSE_InterThree, Improvement = Imp ))
rmse_results %>% knitr::kable()

#------------ Analyse if small amounts of ratings per movie or users could influence result ---

#Show % of residuals still bigger than |1| after deduction of mu, movie- and user bias
#- based on the edx_train set:
temp <- edx_train %>% 
  left_join(bi_avg, by="movieId") %>%
  left_join(bu_avg, by='userId') %>%
  mutate(residual = rating - (mu+b_i+b_u)) 
round(nrow(filter(temp,abs(temp$residual) > 1))/nrow(edx_train),2)

#To compare: Average nb. of ratings per movie (mean and median) for whole edx_train set:
Movie_Count <- edx_train %>% dplyr::count(movieId)
mean(Movie_Count$n)
median(Movie_Count$n)

#To compare: Average nb. of ratings per movie (mean and median) only for residuals > |1|
temp_2 <- filter(temp,abs(temp$residual)>1)
Movie_Count_s <- temp_2 %>% dplyr::count(movieId)
mean(Movie_Count_s$n)
median(Movie_Count_s$n)

#Visualize the effect of low amount of movie-ratings and high residuals (scales package required)
temp_3 <- temp_2 %>% group_by(movieId) %>% summarize(n=n(), residual=mean(abs(residual))) %>% 
  arrange(desc(abs(residual)))

temp_3 %>% ggplot(aes(residual,n)) + 
  geom_point() + 
  scale_y_continuous(trans = log2_trans()) + 
  geom_line(aes(y=median(Movie_Count$n)), color="red") +
  geom_line(aes(y=median(Movie_Count_s$n)))+
  ggtitle("Distribution movie ratings (residuals > |1|") +
  geom_text(aes(x = 3, y = 40, 
                label = paste("Median of residuals > |1| =",paste(median(Movie_Count_s$n)))))+
  geom_text(aes(x = 3, y = 140 ,
                label = paste("Median of total edx_train set =",paste(median(Movie_Count$n)))))+
  myTheme

#Do we see similar effect for users?
#To compare: Average number of ratings per user (mean and median) for whole edx_train set:
User_Count <- edx_train %>% dplyr::count(userId)
mean(User_Count$n)
median(User_Count$n)

#To compare: Average number of ratings per user (mean and median) only for residuals > |1|
temp_2 <- filter(temp,abs(temp$residual)>1)
User_Count_s <- temp_2 %>% dplyr::count(userId)
mean(User_Count_s$n)
median(User_Count_s$n)

#Visualize the effect of low amount of user-ratings and high residuals
temp_3 <- temp_2 %>% group_by(userId) %>% summarize(n=n(), residual=mean(abs(residual))) %>% 
  arrange(desc(abs(residual)))

temp_3 %>% ggplot(aes(residual,n)) + 
  geom_point() + 
  scale_y_continuous(trans = log2_trans()) + 
  geom_line(aes(y=median(User_Count$n)), color="red") +
  geom_line(aes(y=median(User_Count_s$n)))+
  ggtitle("Distribution user ratings (residuals > |1|") +
  geom_text(aes(x = 3.5, y = 15, 
                label = paste("Median of residuals > |1| =",paste(median(User_Count_s$n)))))+
  geom_text(aes(x = 3.5, y = 70,
                label = paste("Median of total edx_train set =",paste(median(User_Count$n)))))+
  myTheme

#----------------------------- Add correction by Regularization ---

#Calculating the regularized b_i (b_i_reg) and b_u (b_u_reg) - including lambda optimization
lambdas <- seq(0, 15, 0.5)
rmses <- sapply(lambdas, function(lambda){
  bi_reg <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i_reg = sum(rating - mu)/(n()+lambda))
  bu_reg <- edx_train %>% 
    left_join(bi_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_reg = sum(rating - b_i_reg - mu)/(n()+lambda))
  predicted_ratings <- 
    edx_test %>% 
    left_join(bi_reg, by = "movieId") %>%
    left_join(bu_reg, by = "userId") %>%
    mutate(pred = mu + b_i_reg + b_u_reg) %>%
    .$pred
  return(RMSE(predicted_ratings, edx_test$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

#Calculating b_i_reg based on best lambda value:
bi_reg_avg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i_reg = sum(rating - mu)/(n()+lambda))

#Calculating b_u_reg based on best lambda value:
bu_reg_avg <- edx_train %>% 
  left_join(bi_reg_avg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u_reg = sum(rating - b_i_reg - mu)/(n()+lambda))

#Calculate RMSE for 4th intermediate result based on regularized biases values:
Reg_rating <- edx_test %>% 
  left_join(bi_reg_avg, by='movieId') %>% 
  left_join(bu_reg_avg, by='userId') %>%
  mutate(pred = mu + b_i_reg + b_u_reg) %>%
  .$pred

RMSE_InterFour <- RMSE(Reg_rating, edx_test$rating)
RMSE_InterFour

#Improvement:
Imp <- RMSE_InterThree-RMSE_InterFour

#Visualize the RMSE result:
rmse_results <- bind_rows(rmse_results,
                          tibble(method="'Regularized'",
                                     RMSE = RMSE_InterFour, Improvement = Imp ))
rmse_results %>% knitr::kable()

#----------------------------- Use SVD to better predict remaining error ---
#recommenderlab package required
#We test two appraoches:
# 1. Add p*q (user effects*principal components) based on a PCA decompostion (done via SVD as prcomp() was
#    far too slow...)
# 2. Creat a fully filled out Rating Matrix via SDV and take colMeans into our algorithm:

#Calculate predictions for edx_train set:
Reg_rating_train <- edx_train %>% 
  left_join(bi_reg_avg, by='movieId') %>% 
  left_join(bu_reg_avg, by='userId') %>%
  mutate(pred = b_i_reg + b_u_reg) %>%
  .$pred

#Establish Rating-Matrix with Recommenderlab function - first deduct rating-predictions from real
#ratings so only the residuals remain:
PreMatrix_edx_train <- edx_train %>% 
  select(userId, movieId, rating) %>% 
  mutate(rating = rating - Reg_rating_train)
rRM <- as(PreMatrix_edx_train, "realRatingMatrix")

#Visualize the distribution of rRM-Matrix:
hist(getRatings(rRM), breaks=100)

#Analyse provided methods within recommender function:
recommender_models <- recommenderRegistry$get_entries(dataType = "realRatingMatrix")
recommender_models

#Get parameter details from SVD method:
recommender_models$SVD_realRatingMatrix$parameter

#Use Recommender() function with method "SVD" - use default parameter
r_rRM <- Recommender(rRM, method = "SVD")
str(r_rRM@model$svd)

#   Start with PCA calculation:

#Creat the PCA values scores (x), loading (rotation) and Variation (sdev^2)
# 1. Step: Calculate covariance matrix of rRM via SDV parameters:
cov_M <- sweep(r_rRM@model$svd$v, 2, r_rRM@model$svd$d^2, FUN="*") %*% t(r_rRM@model$svd$v)/(ncol(rRM)-1)

# 2. step: Calculate Variation (sdev^2 value from prcomp() function) - loading/rotation given thanks
# to V-Matrix from SDV:
eig_value <- sapply(i <- 1:ncol(r_rRM@model$svd$v),function(i) {(cov_M[i,]%*%r_rRM@model$svd$v[,i])/
    r_rRM@model$svd$v[i,i]})
sdev_calc <- sqrt(eig_value)

# 3. step: Calculate score (x value from prcomp() function):
pcaScores <- getRatingMatrix(rRM) %*% r_rRM@model$svd$v

# Visualize Variation of PCs:
barplot(eig_value/sum(eig_value), xlab = "PC", ylab = "Percent of total Variance explained", cex.names=0.8,
        ylim = c(0,1))
eig_value

#Calculate b_c based on calculated PCA parameters - we only have to calculate with one factor as first PC is
# so "dominant":
b_c <- (pcaScores[,1]) %*% t(r_rRM@model$svd$v[,1]*eig_value[1])
b_c <- colMeans(b_c)

#Build a data frame so that b_c can be added to edx_test set later: 
temp_factorization1 <- edx_train %>% group_by(movieId) %>% summarize(movieAverage = mean(rating)) %>% 
  cbind(b_c)

#Calculate RMSE for 5th intermediate result based on factorization:
Fac_rating1 <- edx_test %>% 
  left_join(bi_reg_avg, by='movieId') %>%
  left_join(bu_reg_avg, by='userId') %>%
  left_join(temp_factorization1, by='movieId') %>%
  mutate(pred = mu + b_i_reg + b_u_reg + b_c) %>%
  .$pred

RMSE_InterFive <- RMSE(Fac_rating1, edx_test$rating)
RMSE_InterFive

#Improvement compared to RMSE after Regularization:
Imp1 <- RMSE_InterFour-RMSE_InterFive

#Visualize the RMSE result:
rmse_results <- bind_rows(rmse_results,
                          tibble(method="'PCA'",
                                 RMSE = RMSE_InterFive, Improvement = Imp1))

rmse_results %>% knitr::kable()

#     Now creat full rating matrix via SVD:

#How much variability is explained with first factor:
FirstFactor <- sum(r_rRM@model$svd$d[1]^2)/sum((r_rRM@model$svd$d)^2)

#Visualize how cumsum of each factor explains variability:
Variability <- cumsum(r_rRM@model$svd$d^2)/sum((r_rRM@model$svd$d)^2)
plot(c(0:10),append(0,Variability), type="b", xlab = "Factor", ylab="explained Variability",
     ylim = c(0,1), cex.lab=.8)
text(1,0.9, paste("Explained variability \nwith first factor",round(FirstFactor,3)), cex=.8)

#Calculate matrix Y with only with u,d,v of first factor:
resid <- with(r_rRM@model$svd,(u[, 1, drop=FALSE]*d[1]) %*% t(v[, 1, drop=FALSE]))

#Show that there are no blanks anymore in resid:
resid[1:10, 1:5]

#Calculate the means per movieId and generate b_i2:
b_i2 <- colMeans(resid)

#Build a data frame so that b_i2 can be added to edx_test set later: 
temp_factorization2 <- edx_train %>% group_by(movieId) %>% summarize(movieAverage = mean(rating)) %>% 
  cbind(b_i2)

#Calculate RMSE for 6h intermediate result based on full Rating-Matrix (SVD calculation): 
Fac_rating2 <- edx_test %>% 
  left_join(bi_reg_avg, by='movieId') %>%
  left_join(bu_reg_avg, by='userId') %>%
  left_join(temp_factorization2, by='movieId') %>%
  mutate(pred = mu + b_i_reg + b_u_reg + b_i2) %>%
  .$pred

RMSE_InterSix <- RMSE(Fac_rating2, edx_test$rating)
RMSE_InterSix

#Improvement compared to RMSE after Regularization:
Imp2 <- RMSE_InterFour-RMSE_InterSix 

#Visualize the RMSE result:
rmse_results <- bind_rows(rmse_results,
                          tibble(method= "'Full rating matrix'",
                                     RMSE = RMSE_InterSix, Improvement = Imp2))
                        
rmse_results %>% knitr::kable()

##################### Final RMSE result against validation set #####################################

#Final step: Test the Algorithm against validation set:
Overall_rating <- validation %>% 
  left_join(bi_reg_avg, by='movieId') %>% 
  left_join(bu_reg_avg, by='userId') %>%
  left_join(temp_factorization2, by='movieId') %>%
  mutate(pred = mu + b_i_reg + b_u_reg + b_i2) %>%
  .$pred

RMSE_Overall <- RMSE(Overall_rating, validation$rating)

#Visualize the RMSE result:
rmse_final <- tibble(RMSE_Validation = RMSE_Overall)
rmse_final %>% knitr::kable()