library(tidyverse)
library(text2vec)
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
#代表词语划分到什么程度
tok_fun = word_tokenizer  
#步骤1.设置分词迭代器
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
                                   term_count_min = 5,   #词频，低于10个都删掉
) 

pruned_vocab_10 = prune_vocabulary(vocab,   
                                  term_count_min = 10,   #词频，低于10个都删掉
) 
pruned_vocab_50 = prune_vocabulary(vocab,   
                                   term_count_min = 50,   #词频，低于50个都删掉
) 

prior_stopwords = c('the', 'to', 'that' ,'this', 'on', 'at' ,'with' ,'of')
vocab = filter(vocab, term %in% setdiff(vocab$term, prior_stopwords))
# 99764 words appear more than 5  times
# 68890 words appear more than 10 times
# 30826 words appear more than 50 times



stop_words= c(setdiff(vocab$term,pruned_vocab_50$term), prior_stopwords)

#for stars 1

#164676
stars_1 = filter(yelp, stars == 1)
it_1 = itoken(stars_1$text,   
                  preprocessor = prep_fun,   
                  tokenizer = tok_fun,   
                 )

vocab_1 = create_vocabulary(it_1, stopwords = stop_words) 
n_1 = sum(vocab_1$term_count)
vocab_1 = tibble(key = vocab_1$term, prop_1 = vocab_1$term_count/n_1)

#for stars = 2
stars_2 = filter(yelp, stars == 2)
it_2 = itoken(stars_2$text,   
              preprocessor = prep_fun,   
              tokenizer = tok_fun,   
)
vocab_2 = create_vocabulary(it_2, stopwords = stop_words) 
n_2 = sum(vocab_2$term_count)
vocab_2 = tibble(key = vocab_2$term, prop_2 = vocab_2$term_count/n_2)


#for stars = 3
stars_3 = filter(yelp, stars == 3)
it_3 = itoken(stars_3$text,   
              preprocessor = prep_fun,   
              tokenizer = tok_fun,   
)
vocab_3 = create_vocabulary(it_3, stopwords = stop_words) 
n_3 = sum(vocab_3$term_count)
vocab_3 = tibble(key = vocab_3$term, prop_3 = vocab_3$term_count/n_3)

#for stars = 4
stars_4 = filter(yelp, stars == 4)
it_4 = itoken(stars_4$text,   
              preprocessor = prep_fun,   
              tokenizer = tok_fun,   
)
vocab_4 = create_vocabulary(it_4, stopwords = stop_words) 
n_4 = sum(vocab_4$term_count)
vocab_4 = tibble(key = vocab_4$term, prop_4 = vocab_4$term_count/n_4)


#for stars = 5
stars_5 = filter(yelp, stars == 5)
it_5 = itoken(stars_5$text,   
              preprocessor = prep_fun,   
              tokenizer = tok_fun,   
)
vocab_5 = create_vocabulary(it_5, stopwords = stop_words) 
n_5 = sum(vocab_5$term_count)
vocab_5 = tibble(key = vocab_5$term, prop_5 = vocab_5$term_count/n_5)

combine_12 = full_join(vocab_1,vocab_2,by ='key')
combine_123= full_join(combine_12,vocab_3,by ='key')
combine_1234= full_join(combine_123,vocab_4,by ='key')
combine_all= full_join(combine_1234,vocab_5,by ='key')

combine_all$prop_1[is.na(combine_all$prop_1)] = 0
combine_all$prop_2[is.na(combine_all$prop_2)] = 0
combine_all$prop_3[is.na(combine_all$prop_3)] = 0
combine_all$prop_4[is.na(combine_all$prop_4)] = 0
combine_all$prop_5[is.na(combine_all$prop_5)] = 0

combine_proportion = select(combine_all,prop_1:prop_5)

sd_list= rep(0,length(combine_all$key))
for(i in 1:length(combine_all$key))
    {
  sd_list[i] = sd(combine_proportion[i,])
  
}
sd_tibble = tibble(index = 1:length(combine_all$key), sd = sd_list)

ggplot(data = sd_tibble,mapping = aes(x = sd, y = ..density..))+
  geom_freqpoly(binwidth = 0.0001)

a = arrange(sd_tibble,desc(sd))
a$index[1:100]

work_words =  combine_all$key[a$index[1:100]]

