
library(httr)
library(jsonlite)
library(dplyr)
options(stringsAsFactors = FALSE)
options(encoding = "UTF-8")


mainurl <- 'https://www.dcard.tw/_api/forums/relationship/posts?popular=false'

resdata <- fromJSON(content(GET(mainurl), "text"))
end <- resdata$id[length(id)]

for(i in 1:5){
  url <- paste0(mainurl,"&before=",end)
  print(url)
  tmpres <- fromJSON(content(GET(url), "text"))
  end <- tmpres$id[length(id)]
  resdata <- bind_rows(resdata,tmpres)
  
}

rm(tmpres)

View(resdata)

saveRDS(resdata, file = "data.rds")
