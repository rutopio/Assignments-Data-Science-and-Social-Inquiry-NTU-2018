
library(tidyverse)
library(rvest)
library(httr)
library(dplyr) 
options(stringsAsFactors = F)



url2 <- 'https://dq.yam.com/list.php'
doc2   <- read_html(url2)
class(doc2)
browseURL(url2)


# path <- '//*[@id="pixList"]/div/a/figure/figcaption/h3'
path <- '//*[@id="pixList"]//div[contains(@class,"imgWarp masonry-brick")]/a'
node.a2 <- html_nodes(doc2, xpath = path)
length(node.a)
links <- html_attr(node.a, "href")
length(links)