library(xgboost)
library(Metrics)
library(text2vec)
library(tensorflow)
library(keras)
library(stringr)
library(tidyverse)
library(data.table) 
library(tm)
library("SnowballC")
library("wordcloud")
library("RColorBrewer")


yelp = read_csv('train_data.csv')
# total # of yelp is 1546379 

# data description
# 392720 different words
star_prop = table(yelp$stars)
star_p = prop.table(table(yelp$stars))
label_v = round(star_p * 100, 1)
star_prop_df = data.frame(star = c("1", "2", "3", "4", "5"), prop = as.numeric(star_prop))

label = paste(star_prop_df$star, "(", label_v, "%", ")", sep = '')

ggplot(data = star_prop_df, mapping = aes(x = "", y = prop, fill = star)) + 
  geom_bar(stat = 'identity', position = 'stack', width = 1)+
  coord_polar(theta = "y") + labs(x = "", y = "")

# data matrix
prep_fun = tolower  
tok_fun = word_tokenizer  
it_words = itoken(yelp$text,   
                  preprocessor = prep_fun,   
                  tokenizer = tok_fun)
vocab = create_vocabulary(it_words) 

# first view top 200 which have the most frequent
top_200 =  tail(vocab,200)
top_200 = data.frame(word = top_200$term, freq = top_200$term_count)
set.seed(1234)
wordcloud(words = top_200$word, freq = top_200$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

# by our prior experience, we would like to drop :'the' 'to' 'that' 'this' 'on' 'at' 'with' 'of'


pruned_vocab_5 = prune_vocabulary(vocab,   
                                  term_count_min = 5, 
) 
pruned_vocab_10 = prune_vocabulary(vocab,   
                                   term_count_min = 10,   
pruned_vocab_50 = prune_vocabulary(vocab,   
                                   term_count_min = 50,   
) 

prior_stopwords = c('the', 'to', 'that' ,'this', 'on', 'at' ,'with' ,'of')
vocab = filter(vocab, term %in% setdiff(vocab$term, prior_stopwords))
# 99764 words appear more than 5  times
# 68890 words appear more than 10 times
# 30826 words appear more than 50 times

stop_words= c(setdiff(vocab$term,pruned_vocab_50$term), prior_stopwords)

# read the data
review = read.csv("~/Downloads/train_data.csv", as.is = TRUE)
test_set = read.csv("~/Downloads/testval_data.csv", as.is = TRUE)


# use a text tokenizer to transform the texts into a word index matrix
u = text_tokenizer(num_words = 40000)
fit_text_tokenizer(u, review$text)

x_train = texts_to_sequences(u, review$text)
x_final = texts_to_sequences(u, test_set$text)

x_train = pad_sequences(x_train, 300)
x_final = pad_sequences(x_final, 300)


# change the labels to a matrix form to fit the lstm model
y_train = review$stars

to_star_matrix = function(y) {
  if(is.matrix(y)) {
    return(y)
  }
  l = length(y)
  s = matrix(0, nrow = l, ncol = 5)
  for(i in 1:l) {
    s[i, y[i]] = 1
  }
  return(s)
}

y_train = to_star_matrix(y_train)


# construct a lstm model
model = keras_model_sequential()

model %>%
  layer_embedding(input_dim = 40000, output_dim = 128) %>%
  layer_lstm(units = 64, dropout = 0.5, recurrent_dropout = 0.2, return_sequences = TRUE) %>% 
  layer_lstm(units = 64, dropout = 0.5, recurrent_dropout = 0.2) %>%
  layer_dense(units = 5, activation = 'sigmoid')

model %>% compile(
  loss = 'mse',
  optimizer = 'adam',
  metrics = c('accuracy')
)

model %>% keras::fit(
  x_train, y_train,
  epochs = 4
)


# predictions on test set and train set with the lstm model
final_probs = predict_proba(model, x_final, verbose = 1)
a = predict_proba(model, x_train, verbose = 1)


# use some other information (time, sentence length, lon, lat, ...)
t0 = as.numeric(unclass(as.POSIXct(review$date)))
nwords = numeric(dim(review)[1])

for(i in 1:dim(review)[1]) {
  nwords[i] = length(word_tokenizer(review$text[i])[[1]])
}

review = review[c("stars", "longitude", "latitude")]

train = cbind(review, a, t0, nwords)

train_data = train[, 2:10]
train_label = train[, 1]
train_data = matrix(unlist(train_data), ncol = 9)


# fit a decision tree model
bst = xgboost(data = train_data, label = train_label, max.depth = 7, eta = 0.5, nround = 80,  nthread = 2, eval_metric = "rmse", nestimators = 50, learning_rate = 0.1, subsample = 0.7, colsample_bytree = 0.8) 


# predictions with the decision tree model
t1 = as.numeric(unclass(as.POSIXct(test_set$date)))
nwords2 = numeric(dim(test_set)[1])

for(i in 1:dim(test_set)[1]) {
  nwords2[i] = length(word_tokenizer(test_set$text[i])[[1]])
}

test_set = test_set[c("longitude", "latitude")]
test = cbind(test_set, final_probs, t1, nwords2)

test_data = test[, 1:9]
test_data = matrix(unlist(test_data), ncol = 9)

pred = predict(bst, test_data)

result = data.frame(Id = 1:1016664, Prediction1 = pred)
#write.csv(result, "~/Desktop/M2/result.csv", row.names = FALSE) 


# semantic analysis
semantic = function(mytext)
{
  mytext = str_split(mytext, pattern = " ")
  mytext = unlist(mytext)
  mytext.length = length(mytext)
  
  for(i in 1:mytext.length)
  { 
    sub_text = paste(mytext[1:i], collapse = " ")
    print(sub_text)
    t = c( sub_text, " ")
    x_t = texts_to_sequences(u, t)
    x_t = pad_sequences(x_t, 300)
    mypred = predict_proba(model, x_t)
    mypred = mypred[1,]
    print(which.max(mypred))
  }
}

mytext1 = "11.99 Steak and Lobster! That's what got us in here when we were checking out Fremont St. just to see what it was like the steak was a 10oz sirloin that was medium well not medium rare and the lobster was the tail from a lobster that was probably under 1lb definitely frozen veggies on the place too like you had when you were a kid The place was filled with people that were partying and gambling earlier in the night since the special started at 9pm. Service was super prompt and efficient not sure I'll ever be back here since Fremont St. is only cool to check out once but it wasn't as bad as i expected"
semantic(mytext1)

mytext2 = c("the food is great but the sevice is terrible  so even if i very very love their food i'll never come here again")
semantic(mytext2)

t = c("came here for clockwork coffee which shares the same space and the people behind the bar were super friendly in educating us about pour over coffee they were great at telling us how it was and explaining the finer details of the process but it is pretty pricey for what you get and the coffee is good but not amazing i personally would not go back", " ")
x_t = texts_to_sequences(u, t)
x_t = pad_sequences(x_t, 300)
model %>% predict_proba(x_t)