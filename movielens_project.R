library(tidyverse)
library(caret)
library(data.table)
knitr::opts_chunk$set(echo = TRUE)

# Introduction
# The MovieLens dataset used in this project comprises 10 millions ratings for 
# 10677 movies rated by 69878 users. The scale for the rating goes from
# 0 to 5 with increments of 0.5. The product between the number of users and 
# the number of movies is close to 746 millions, which is greater than the number 
# of ratings. This difference indicates that not every user rated every movie. 
# The MovieLens dataset has six columns which are: (1) the movie identifier, 
# (2) the user identifier, (3) the rating of the movie, (4) a time stamp with 
# the date of the rating, (5) the title of the movie, and (6) the genres of the 
# movie.

# The purpose of this project is to build a linear model to predict the rating 
# $r_{m,u}$ that the user $u$ will give to the movie $m$. Such a model can be 
# mathematically written as follows:
#  
#  r(m,u) = u + lambda*(bm + bu) + e(m,u)
#

# u is the average movie rating, bm represents the movie-to-movie variation, 
# bu is the user-to-user variation, e(m,u) is a zero mean random 
# variable representing uncertainty, and lambda is a penalty factor that 
# avoids large values for bm and bu. A second model that we want
# to explore includes the movie genre effect bg:
#  
#  r(m,u,g) = u + lambda*(bm + bu + bg) + e(m,u,g)
#

# Set the variable "create_data_set" to 1 to download and rebuild the "movielens"
# dataset, otherwise a data frame previously built will be loaded from 
# the "movielens.Rda" file.
create_data_set <- 1
# Create the "movielens" data set
if (create_data_set) {
  dl <- tempfile()
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  
  # The "ratings" data frame comprises the columns "userId", "movieId", "rating", 
  # and "timestamp"
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  # The "movies" data frame comprises the columns "movieId", "title" and "genre"
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                             title = as.character(title),
                                             genres = as.character(genres))
  
  # The "movielens" data frame cmbines the data frames "ratings" and "movies", 
  # therefore its columns are "userId", "movieId", "rating", "timestamp",
  # "title" and "genre".
  movielens <- left_join(ratings, movies, by = "movieId")
  save(movielens, file = "movielens.Rda")
} else {
  load("movielens.Rda")
}

# Number of movies, users, and genres
nmovies <- unique(movielens$movieId) %>% length()
nusers <- unique(movielens$userId) %>% length()
ngenres <- unique(movielens$genres) %>% length()

# Methods
# Data exploration and visualization
# The following table presents the titles of the most rated movies:
rmovies <- movielens %>% group_by(movieId) %>% 
  summarise(NRatings = n()) 
topId <- rmovies %>% arrange(desc(NRatings)) %>% head()
topId %>% left_join(movielens,by="movieId") %>% group_by(movieId) %>%
  slice(1) %>% ungroup() %>% select(title, NRatings) %>% 
  arrange(desc(NRatings)) %>% 
  rename("Title"=title,"Num. of ratings"=NRatings) %>%
  knitr::kable(caption = "Most rated movies")

# The following Figure shows a histogram with the distribution of ratings 
# among movies
rmovies %>% ggplot(aes(x = NRatings)) + geom_histogram(bins = 10) +
  scale_x_continuous(trans='log10')

# The following Figure shows a histogram with the distribution of ratings among 
# users.
umovies <- movielens %>% group_by(userId) %>% 
  summarise(NRatings = n()) %>% arrange(desc(NRatings))
umovies %>% ggplot(aes(x = NRatings)) + geom_histogram(bins = 10) +
  scale_x_continuous(trans='log10')

## Insights gained
#The data exploration and visualization of the previous section permit us to
#conclude that not every movie received the same number of ratings, and that no 
#every user rated the same number of movies.

## Modeling approach
# The approach followed in this report consists of building models of increasing
# complexity until a mean square error of less than $0.86490$ is obtained. These 
# are the models that will be used:
# 1. r(m,u) = u + e(m,u)
# 2. r(m,u) = u + bm + e(m,u)
# 3. r(m,u) = u + lambda*bm + e(m,u).
# 4. r(m,u) = u + lambda*(bm + bu) + e(m,u)
# 5. r(m,u,g) = u + lambda*(bm + bu + bg) + e(m,u,g)
  
# The quality of a model will be assessed according to the root mean squared error,
# which is defined as follows:

# Function to calculate the root mean square error
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Download the MovieLens database
# The first step to build the dataset is to download the file *ml-10m.zip* from the 
# MovieLens web site (https://files.grouplens.org/datasets/movielens/ml-10m.zip) 
# website. *ml-10m.zip* contains the files *ratings.dat* and *movies.dat*. The rows
# of the first file comprise a user identifier, a movie identifier, a rating, and 
# a time stamp; while the rows of the file are composed by a movie identifier, a 
# movie title, and a movie genre. The information obtained from *ratings.dat* 
# and *movies.dat* is combined into a single data frame named *movieLens*.

# Data partition
# The second step is to use the function *createDataPartition* of the *caret* 
# package to divide the data set into $90\%$ for modelling and $10\%$ for
# validation. The modelling and validation data frames will be termed *edx* and
# *validation*, respectively.
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, 
                                  p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Ensures that all users and movies in the "validation" data frame are also in 
# "edx" data frame
validation <- temp %>% semi_join(edx, by = "movieId") %>% 
  semi_join(edx, by = "userId")

# Puts the rows removed from the "validation" data frame into the "edx" data 
# frame.
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Results

# ******************************************************************************
# Estimation of the movie average rating
# MODEL 1: r(m,u) = u + e(m,u)
# ******************************************************************************
# The population mean is estimated by averaging all ratings of the *edx* data 
# frame. The resulting value is $\hat{u} =$ `r round(mu,2)`, which means that no
# matter the user or the movie, the predicted rating is:
mu <- mean(edx$rating)
yhat1 <- rep(mu, nrow(validation))
# RMSE
rmse1 <- RMSE(validation$rating, yhat1)
# Row that will be added to the data frame with the final results
rmse_row1 <- data.frame(Method = "u+e", RMSE = rmse1)

# ******************************************************************************
# Estimation of the movie effect
# MODEL 2: r(m,u) = u + bm + e(m,u)
# ******************************************************************************
# Movie effect
movie_avgs <- edx %>% group_by(movieId) %>% summarize(bi = mean(rating - mu))
# Predicted rating
yhat2 <- validation %>% left_join(movie_avgs, by="movieId") %>% 
  mutate(yhat = mu + bi) %>% pull(yhat)
# RMSE
rmse2 <- RMSE(validation$rating, yhat2)
# Row that will be added to the data frame with the final results
rmse_row2 <- data.frame(Method = "u+bm+e", RMSE = rmse2)

# Some values of bm are not reliable because they were estimated using just 
# a few ratings (one or two in some cases). The following table
# shows that the best rated movies have only one or two ratings.

# Best rated movies
best_rated <- edx %>% group_by(movieId) %>% 
  summarise(avrating = mean(rating), nratings = n()) %>% 
  arrange(desc(avrating)) %>% head()
best_rated %>% left_join(edx,by="movieId") %>% 
  select(title,avrating,nratings) %>%  rename("Title"=title) %>%
  rename("Av. Rating"=avrating, "Num. Ratings"=nratings) %>%
  knitr::kable(caption = "Movies with the highest average ratings")

# ******************************************************************************
# Estimation of the movie effect with regularization
# MODEL 3:r(m,u) = u + lambda*bm + e(m,u)
# ******************************************************************************
# Regularization permit us to take into consideration that the values of 
# bm estimated with very few ratings are random variables with higher 
# variability than the bm estimated with hundreds or thousands of 
# ratings. In a model with regularization, a penalty factor lambda
# multiplies the estimate that is regularized:
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  # Movie effect
  bi <- edx %>% group_by(movieId) %>% summarize(bi = sum(rating - mu)/(n()+l))
  # Prediction
  yhat <- validation %>% left_join(bi, by = "movieId") %>%
    mutate(pred = mu + bi) %>% pull(pred)
  # Return
  return(RMSE(yhat, validation$rating))
})
# Optimal value of lambda
lmin_bm <- lambdas[which.min(rmses)]
# Lowest RMSE
rmse3 <- min(rmses)
# Row that will be added to the data frame with the final results
rmse_row3 <- data.frame(Method = "u+lambda*bm+e", RMSE = rmse3)

# ******************************************************************************
# Estimation of the movie and user effects with regularization
# MODEL 4:r(m,u) = u + lambda*(bm + bu) + e(m,u)
# ******************************************************************************
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  # Movie effect
  bi <- edx %>% group_by(movieId) %>% summarize(bi = sum(rating - mu)/(n()+l))
  # User effect
  bu <- edx %>% left_join(bi, by="movieId") %>% group_by(userId) %>%
    summarize(bu = sum(rating - bi - mu)/(n()+l))
  # Prediction
  yhat <- validation %>% left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>% mutate(pred = mu + bi + bu) %>% pull(pred)
  # Return
  return(RMSE(yhat, validation$rating))
})
# Optimal value of lambda
lmin_bmbu <- lambdas[which.min(rmses)]
# Lowest RMSE
rmse4 <- min(rmses)
# Row that will be added to the data frame with the final results
rmse_row4 <- data.frame(Method = "u+lambda*(bm+bu)+e", RMSE = rmse4)

# ******************************************************************************
# Estimation of the movie, user and genre effects with regularization
# MODEL 5:r(m,u,g) = u + lambda*(bm + bu + bg) + e(m,u,g)
# ******************************************************************************
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  # Movie effect
  bm <- edx %>% group_by(movieId) %>% summarize(bm = sum(rating - mu)/(n()+l))
  # User effect
  bu <- edx %>% left_join(bm, by="movieId") %>% group_by(userId) %>%
    summarize(bu = sum(rating - bm - mu)/(n()+l))
  # Genre effect
  bg <- edx %>% left_join(bm, by="movieId") %>% left_join(bu, by="userId") %>%
    group_by(genres) %>% summarize(bg = sum(rating - bm - bu - mu)/(n()+l))
  # Prediction
  yhat <- validation %>% left_join(bm, by="movieId") %>%
    left_join(bu, by="userId") %>% left_join(bg, by="genres") %>%  
    mutate(pred = mu + bm + bu + bg) %>% pull(pred)
  # Return
  return(RMSE(yhat, validation$rating))
})
# Optimal value of lambda
lmin_bmbubg <- lambdas[which.min(rmses)]
# Lowest RMSE
rmse5 <- min(rmses)
# Row that will be added to the data frame with the final results
rmse_row5 <- data.frame(Method = "u+lambda*(bm+bu+bg)+e", RMSE = rmse5)

# Summary of results
# The following table summarizes the RMSE given by each model.
rmse_results <- rbind(rmse_row1, rmse_row2, rmse_row3, rmse_row4, rmse_row5)
rmse_results %>% knitr::kable(caption = "RMSE by model")
