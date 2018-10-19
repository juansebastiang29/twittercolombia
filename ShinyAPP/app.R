

library(shiny)


ui <- fluidPage(
    fluidRow(class="myRow1",
             sidebarLayout(sidebarPanel(
               ## tweet activity html ----
               selectInput("activity_tweet", label = h3("Tweet activity"), 
                           choices = list("Tweet" = "count_tweet",
                                          "Retweet" = "reweet_count",
                                          "Reply" = "reply_count",
                                          "Favorite" = "favorite_count",
                                          "All" = "all"),
                           selected = "count_tweet"),
               selectInput("x_axis", label = h3("X-axis varible"), 
                           choices = list("Tweet" = "count_tweet",
                                          "Retweet" = "reweet_count",
                                          "Reply" = "reply_count",
                                          "Favorite" = "favorite_count",
                                          "All" = "all"),
                           selected = "count_tweet"),
               selectInput("y_axis", label = h3("Y-axis variable"), 
                           choices = list("Tweet" = "count_tweet",
                                          "Retweet" = "reweet_count",
                                          "Reply" = "reply_count",
                                          "Favorite" = "favorite_count",
                                          "All" = "all"),
                           selected = "favorite_count")),
               mainPanel(tabsetPanel(tabPanel("Time Series",plotOutput("time_s")),
                                     tabPanel("Top Users",plotOutput("hist")),
                                     tabPanel("Activity interactions",
                                              plotOutput("sctr"))))),
             titlePanel("Tweets from Top Users"),
             DT::dataTableOutput("mytable_1")
    ),
    fluidRow(class="myRow2",
             titlePanel("Word Clouds"),
             
             ## word clouds html ----
             sidebarLayout(
               sidebarPanel(
                 selectInput("n_gram",label=h3("Select the N-gram size"),
                             choices=list("1"=1,"2"=2,"3"=3),
                             selected = "1"),
                 sliderInput("sparsity", label = h3("Set the Sparsity for the Term Matrix"), 
                             min = 0.95,max = 0.99, value = 0.99,step=0.01),
                 sliderInput("n_words", label = h3("Set the Maximum number of words"), 
                             min = 15,max = 100, value = 75))
               ,
               mainPanel(
                 tabsetPanel(
                   tabPanel(title = "Tweets",plotOutput("WC_tweets")),
                   tabPanel(title = "users",plotOutput("WC_users")))))),
    fluidRow(class="myRow3",
             titlePanel("\nClustering Results \nPlotting Text Dissimilarity (1-cosine similatiry)"),
             plotOutput("cluster"),
             tabsetPanel(
               tabPanel(title = "Cluster 0",DT::dataTableOutput("mytable_2")),
               tabPanel(title = "Cluster 1",DT::dataTableOutput("mytable_3")),
               tabPanel(title = "Cluster 2",DT::dataTableOutput("mytable_4")))))
server <- function(input,output,session){
        library(shiny)
        library(ggplot2)
        library(dplyr)
        library(xts)
        library(scales)
        library(zoo)
        library(DT)
        library(tm)
        library(SnowballC)
        library(wordcloud)
        library(RWeka)
        library(NLP)
  
        
        
        options(scipen=999)
        
        #read the data ----
        
        tweets<-read.csv('./data/tweets_cluster.csv',header = TRUE,sep=',')
        rt_s<-read.csv('./data/db_RT.csv')
        users_data <- read.csv('./data/user_clusters.csv')
        
        
        #counting cube ----
        id_tweet<- cbind(tweets[,c('created_at','tweet_id','user_id')],1)
        names(id_tweet)<-c('created_at','tweet_id','user_id','count_tweet')
        id_tweet<-id_tweet[complete.cases(id_tweet),]
        id_rt_rep<-rt_s[,c('created_at','tweet_id',
                           'reweet_count','reply_count','favorite_count')]
        
        #select the last observation in the retweet table
        id_rt_rep <- id_rt_rep %>% 
          arrange(desc(tweet_id,created_at)) %>%
          distinct(tweet_id,.keep_all = TRUE)
        id_rt_rep <- id_rt_rep[,c('tweet_id','reweet_count',
                                  'reply_count','favorite_count')]
        
        #merge with tweets table
        cube_twitter <- merge(id_tweet,id_rt_rep,
                              by.x='tweet_id',
                              by.y='tweet_id',
                              all.x=TRUE)
        
        cube_twitter$all<-cube_twitter$count_tweet+cube_twitter$reweet_count+
          cube_twitter$reply_count+cube_twitter$favorite_count
        
        remove(id_rt_rep,id_tweet)
        dat_usr_tweet <- aggregate(. ~ user_id, cube_twitter, sum)
        user_names <- unique(tweets[,c('user_id','screen_name')])
        user_names<- user_names[complete.cases(user_names) &
                                  duplicated(user_names$user_id)==FALSE,]
        dat_usr_tweet <- merge(dat_usr_tweet,user_names,
                               by.x='user_id',
                               by.y='user_id',
                               all.x=TRUE)
        
        list_var=c("count_tweet","reweet_count","reply_count","favorite_count","all")
        
        for (item in list_var){
          dat_usr_tweet <- dat_usr_tweet %>% 
            arrange_(.dots=c(paste0('desc(',item,')')))
          dat_usr_tweet$top_ <- 1
          df <- dat_usr_tweet[1:20,]
          cube_twitter <- merge(cube_twitter,df[,c('user_id','top_')],
                                by.x='user_id',
                                by.y='user_id',
                                all.x=TRUE)
          names(cube_twitter)[names(cube_twitter)=='top_'] <- paste0('top_',item)
        }
        remove(user_names,df)
        gc()
        cube_twitter <- merge(cube_twitter,
                              tweets[,c('tweet_id','screen_name')] %>% 
                                distinct(tweet_id,.keep_all = TRUE),
                              by.x='tweet_id',
                              by.y='tweet_id',
                              all.x=TRUE)
        
        #time series processing ----
        
        cube_twitter$created_at<-as.POSIXct(cube_twitter$created_at,format="%Y-%m-%d %H:%M:%S")
        rt_s$created_at<- as.POSIXct(rt_s$created_at,format="%Y-%m-%d %H:%M:%S")
        
        convert_hourly <- function(text){
          if (text=="count_tweet"){
            data_<-cube_twitter
            ts.data <- xts(data_[,c(text)],
                           order.by = data_$created_at)
            hourly_<-(period.apply(ts.data, endpoints(ts.data, "hours"), sum))
          }
          else{
            data_<-rt_s
            ts.data <- xts(data_[,c(text)],
                           order.by = data_$created_at)
            hourly_<-period.apply(ts.data,endpoints(ts.data,"hours"),mean)
          }
          
          ind=index(hourly_)
          values=as.vector(hourly_)
          hourly_<-data.frame(y=values,date=ind)
          remove(ts.data,ind,values,data_)
          return(hourly_)
        }
        
        MA_zoo<-function(ts,k_){
          como_asi<-rollmean(zoo(ts$y,ts$date),k=k_)
          roll_mean<- data.frame(date=index(como_asi),y=as.vector(como_asi))
          return(roll_mean)
        }
        
        # NLTK processing ----
        
        #words
        words_tweets <- tweets$clean_tweet
        words_users <- users_data$user_desc
        
        cleaning_text <- function(data){
          #corpus
          element_corpus <- Corpus(VectorSource(data))
          
          #text cleaning
          
          my_list=c('cuál','pm','am','va','p m','a m','q','ver',
                    'acá','aca','aqui','da','m','p','tal','tan',
                    'v','u','cómo','ve','retweeted','fm','usted',
                    'responde','espere','tambien','dice','dicen','dijo',
                    'segun','segun','cada','anos','aun','aunque','cree',
                    'creen','creer','creo','decir','demas','estan','retwit',
                    'hace','hacen','hacer','hecha','hicieron' ,'hizo',
                    'hoy','puede','quiere','ser','sera','si','van','asi',
                    'ahi','ahora','vez','via','vea','mas','b')
          
          clean_text<-tm_map(element_corpus,stripWhitespace)
          clean_text<-tm_map(clean_text,tolower)
          clean_text<-tm_map(clean_text,removeNumbers)
          clean_text<-tm_map(clean_text,removePunctuation)
          clean_text<-tm_map(clean_text,removeWords, stopwords('spanish'))
          clean_text<-tm_map(clean_text,removeWords,my_list)
          clean_text<-tm_map(clean_text,stripWhitespace)
          clean_text<- VCorpus(VectorSource(clean_text))
          return(clean_text)
        }
        
        clean_text_tweets=cleaning_text(data=words_tweets)
        clean_text_user=cleaning_text(data=words_users)
        # inspect(clean_text_tweets)[1:10]
        # tokenization n-grams
        
        
        
        # Create an n-gram Word Cloud -----
        matrices_wc <- function(data,n,sparsity){
          Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = n, max = n))
          tdm <- TermDocumentMatrix(data, control = list(tokenize = Tokenizer))
          dtm <- DocumentTermMatrix(data, control = list(tokenize = Tokenizer))
          tdm<-removeSparseTerms(tdm, sparsity)
          results <- list(tdm=tdm,dtm=dtm)
        }
        
        
        #####ROW1 ----
        
        #reactive values ROW1 ----
        
        #outputs ROW1 ----
        
        # Time series ----
        
        
        output$time_s <- renderPlot({
          base_size = 12
          if (input$activity_tweet=="all"){
            hourly_1<-convert_hourly(text="count_tweet")
            hourly_2<-convert_hourly(text="reweet_count")
            hourly_3<-convert_hourly(text="reply_count")
            hourly_4<-convert_hourly(text="favorite_count")
            ggplot(data = hourly_1,aes(x=date,y=y))+
              geom_line(aes(colour="# Tweets"),size = 1.25)+
              geom_line(data = hourly_2,aes(x=date,y=y,colour="Retweets cum."))+
              geom_line(data = hourly_3,aes(x=date,y=y,colour="Reply cum."))+
              geom_line(data = hourly_4,aes(x=date,y=y,colour="Favorite cum."))+
              scale_x_datetime(breaks = date_breaks("8 hour"),
                               labels=date_format("%d %H:%M"))+
              theme_bw()+
              theme(panel.border = element_blank(),
                    axis.line =  element_line(colour="gray70"),
                    axis.text.x = element_text(size = base_size * 1.25 , 
                                               lineheight = 0.9, 
                                               colour = "black", 
                                               vjust = 1,angle = 90),
                    axis.text.y = element_text(size = base_size * 1.25, 
                                               lineheight = 0.9, 
                                               colour = "black", hjust = 1),
                    axis.title.x = element_text(size = base_size * 1.5, 
                                                angle = 0),
                    legend.text = element_text(size = base_size * 1.25),
                    panel.grid.minor =  element_blank())
          }
          else{
            hourly_<-convert_hourly(input$activity_tweet)
            
            MA_2<-MA_zoo(hourly_,k=2)
            
            ggplot(data = hourly_,aes(x=date,y=y))+
              geom_line(aes(colour="Activity"), size = 1,alpha=0.4)+
              geom_line(data=MA_2,aes(x=date,y=y,colour="Moving_AV_2"),
                        size = 0.75)+
              scale_x_datetime(breaks = date_breaks("8 hour"),
                               labels=date_format("%d %H:%M"))+
              scale_colour_manual(name="Time Series",
                                  values=c(Activity="dodgerblue4",
                                           Moving_AV_2="grey55"))+
              theme_bw()+
              theme(panel.border = element_blank(),
                    axis.line =  element_line(colour="gray70"),
                    axis.text.x = element_text(size = base_size * 1.25 , 
                                               lineheight = 0.9, 
                                               colour = "black", 
                                               vjust = 1,angle = 90),
                    axis.text.y = element_text(size = base_size * 1.25, 
                                               lineheight = 0.9, 
                                               colour = "black", hjust = 1),
                    axis.title.x = element_text(size = base_size * 1.5, 
                                                angle = 0),
                    legend.text = element_text(size = base_size * 1.25),
                    legend.title = element_text(size = base_size * 1.25, 
                                                face = "bold", hjust = 0),
                    panel.grid.minor =  element_blank())
          }
          
          
        })
        
        
        
        
        #histogram ----
        
        output$hist <- renderPlot({
          df <- cube_twitter[,c('tweet_id','screen_name',paste0('top_',input$activity_tweet))]
          df<-df[complete.cases(df),]
          base_size = 12
          df_1 <- within(df, screen_name <- factor(screen_name, 
                                                   levels=names(sort(table(screen_name), 
                                                                     decreasing=TRUE))))
          ggplot(df_1,aes(screen_name))+
            geom_bar(stat="count",width=0.75,
                     fill='lightblue3')+
            ylab(paste0("Counts"))+
            scale_y_continuous(breaks=seq(0,90,10))+
            coord_flip()+
            theme_bw()+theme(panel.border = element_blank(),
                             axis.line =  element_line(colour="gray70"),
                             axis.text.x = element_text(size = base_size * 1.25 , 
                                                        lineheight = 0.9, 
                                                        colour = "black", vjust = 1),
                             axis.text.y = element_text(size = base_size * 1.25, 
                                                        lineheight = 0.9, 
                                                        colour = "black", hjust = 1),
                             axis.title.x = element_text(size = base_size * 1.25, angle = 0),
                             panel.grid.major.x =  element_line(colour = "lightblue4", 
                                                                size = 0.10),
                             panel.grid.major.y = element_blank(),
                             panel.grid.minor =  element_blank())
        })
        
        
        output$mytable_1 <- DT::renderDataTable({
          df <- cube_twitter[,c('tweet_id',paste0('top_',input$activity_tweet))]
          
          df <- df[complete.cases(df),]
          
          df <- merge(df,tweets[,c('tweet_id','screen_name','text_tweet','user_description')],
                      by.x='tweet_id',
                      by.y='tweet_id',
                      all.x=TRUE)
          df <- df[,c('screen_name','user_description','text_tweet')]
          df <- df[sample(nrow(df), 20), ]
        })
        
        #scatter plt ----
        
        output$sctr <- renderPlot({
          base_size = 12
          df_ <- dat_usr_tweet[,c(input$x_axis,input$y_axis)]
          names(df_) <- c('x_var','y_var')
          ggplot(data = df_, aes(x = x_var, y = y_var))+
            geom_point(aes(colour=y_var),size=2)+
            scale_colour_gradient(low = "dodgerblue", high = "dodgerblue4")+
            theme_bw()+
            labs(title="Scatter plot activity by user\n", color = "Activity\n")+
            theme(panel.border = element_blank(),
                  axis.line =  element_line(colour="gray70"),
                  axis.text.x = element_text(size = base_size * 1.25, 
                                             lineheight = 0.9, 
                                             colour = "black", vjust = 1),
                  axis.text.y = element_text(size = base_size * 1.25, 
                                             lineheight = 0.9, 
                                             colour = "black", hjust = 1),
                  axis.title.x = element_text(size = base_size * 1.25),
                  panel.grid.major.x =  element_line(colour = "lightblue4", 
                                                     size = 0.10),
                  panel.grid.minor =  element_blank())
        })
        
        # word cloud ----
        
        output$WC_tweets <- renderPlot({
          n_=input$n_gram
          sparsity_=input$sparsity
          n_words_=input$n_words
          
          hola=matrices_wc(data=clean_text_tweets,n=n_,sparsity = sparsity_)
          m=as.matrix(hola$tdm)
          v=sort(rowSums(m),decreasing=TRUE)
          names(v)
          d=data.frame(word=names(v),freq=v)
          pal = brewer.pal(9,"BuPu")[4:9]
          wordcloud(words = d$word,
                    freq = d$freq,
                    scale = c(3,0.75),
                    max.words=n_words_,
                    random.order = F,
                    colors = pal)
          remove(d,v,m)
        })
        
        output$WC_users <- renderPlot({
          n_=input$n_gram
          sparsity_=input$sparsity
          n_words_=input$n_words
          
          hola=matrices_wc(data=clean_text_user,n=n_,sparsity = sparsity_)
          m=as.matrix(hola$tdm)
          v=sort(rowSums(m),decreasing=TRUE)
          names(v)
          d=data.frame(word=names(v),freq=v)
          pal = brewer.pal(9,"BuPu")[4:9]
          wordcloud(words = d$word,
                    freq = d$freq,
                    scale = c(3,0.75),
                    max.words=n_words_,
                    random.order = F,
                    colors = pal)
          remove(d,v,m)
        })
        
        #clustering ----
        
        output$cluster <- renderPlot({
          base_size = 12
          df<- tweets[,c('MDS_tweets_comp1','MDS_tweets_comp2','cluster')]
          ggplot(data=df,aes(x=MDS_tweets_comp1,y=MDS_tweets_comp2))+
            geom_point(aes(colour=factor(cluster)))+
            theme_bw()+
            labs(title = "Text Tweet Clustering\n", 
                 x = "1st Component", 
                 y = "2nd Component", color = "Cluster ID\n")+
            theme(panel.border = element_blank(),
                  axis.line =  element_line(colour="gray70"),
                  axis.text.x = element_text(size = base_size * 1.5, 
                                             lineheight = 0.9, 
                                             colour = "black", vjust = 1),
                  axis.text.y = element_text(size = base_size * 1.5, 
                                             lineheight = 0.9, 
                                             colour = "black", hjust = 1),
                  axis.title.x = element_text(size = base_size * 1.5, angle = 0),
                  axis.title.y = element_text(size = base_size * 1.5, angle = 90),
                  panel.grid.major.x =  element_line(colour = "lightblue4", 
                                                     size = 0.10),
                  legend.text = element_text(size = base_size * 1.25),
                  legend.title = element_text(size = base_size * 1.25, 
                                              face = "bold", hjust = 0),
                  panel.grid.minor =  element_blank())
          
        })
        
        output$mytable_2 <- DT::renderDataTable({
          df <- tweets[tweets$cluster==0,c('text_tweet','cluster')]
          df <- df[sample(nrow(df), 20), ]
        })
        
        output$mytable_3 <- DT::renderDataTable({
          df <- tweets[tweets$cluster==1,c('text_tweet','cluster')]
          df <- df[sample(nrow(df), 20), ]
        })
        
        output$mytable_4 <- DT::renderDataTable({
          df <- tweets[tweets$cluster==2,c('text_tweet','cluster')]
          df <- df[sample(nrow(df), 20), ]
        })
        
      }

shinyApp(ui, server)