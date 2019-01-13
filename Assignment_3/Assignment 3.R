library(httr)
library(jsonlite)
library(ggplot2)
library(ggmap)
options(stringsAsFactors = F)

# google api
# ref: https://github.com/dkahle/ggmap/issues/51
register_google(key = "AIzaSyAvMcKUuEy5_xdT8h4h-rnSkaNwHU9NEP8") 

url <- 'https://taqm.epa.gov.tw/taqm/aqs.ashx?lang=tw&act=aqi-epa&ts=1538931940046'
air <- fromJSON(content(GET(url), "text", encoding = "utf-8"))
air.df <- as.data.frame(air$Data)

air.df$lat <- as.numeric(air.df$lat,na.rm='TRUE')
air.df$lng <- as.numeric(air.df$lng,na.rm='TRUE')

air.df$PM10_AVG <- as.numeric(air.df$PM10_AVG,na.rm='TRUE',defalt=0)
air.df$PM25_AVG <- as.numeric(air.df$PM25_AVG,na.rm='TRUE',defalt=0)
air.df$O3 <- as.numeric(air.df$O3,na.rm='TRUE',defalt=0)
air.df$O3_8 <- as.numeric(air.df$O3_8, na.rm='TRUE',defalt=0)
air.df$CO_8 <- as.numeric(air.df$CO_8,na.rm='TRUE',defalt=0)
air.df$NO2 <- as.numeric(air.df$NO2,na.rm='TRUE',defalt=0)
air.df$SO2 <- as.numeric(air.df$SO2,na.rm='TRUE',defalt=0)

air.df[is.na(air.df)] <- 0

# https://www.google.com.tw/maps/@23.6619723,121.0565847,7.74z?hl=zh-TW


map <- get_map(location = c(lon = 121.0565847,lat = 23.6619723),zoom = 7, language = "zh-TW")



O3_8_color <- function(ratio){
  if(ratio <= 54){
    return("#00e800") # green
  }
  else if(ratio >= 55 && ratio <= 70 ){
    return("#ffff00") # yello
  }
  else if(ratio >= 71 && ratio <= 85 ){
    return("#ff7e00") # orange
  }
  else if(ratio >= 86 && ratio <= 105 ){
    return("#ff0000") # red
  }
  else if(ratio >= 86 && ratio <= 105 ){
    return("#8f3f97") # purple
  }
  else{
    return("#7e0023") # china_purple
  }
}

air.df$O3_8_col <- sapply(air.df$O3_8, O3_8_color)
  
ggmap(map)+geom_point(data=air.df,aes(x=lng, y=lat), colour=(air.df$O3_8_col), size=air.df$O3/10, alpha=0.4)+labs(title="O3_avg_8")

bp <- ggplot(data=PlantGrowth, aes(x=group, y=weight, fill=group)) + geom_boxplot()


ll <- list(a = list(x = 1:10, y = 2:11, group = 1), 
           b = list(x = 11:20, y = 12:21, group = 2))
dfll <- do.call(rbind,lapply(ll, data.frame))
ll1 <- list(a = list(x = 1:10, y = 2:11, z = 1:3), 
            b = list(x = 11:20, y = 12:21, z = 1:3))

installed.packages('jsonlite')
library(jsonlite)
url <- 'https://data.fda.gov.tw/opendata/exportDataList.do?method=ExportData&InfoId=19&logType=3'
data <- fromJSON(url)

data.v <- unlist(data)
data.v <- data.v[!is.na(data.v)]
data.m <- matrix(data.v, byrow = T, ncol = 11)
data.df <- as.data.frame(data.m)
names(data.df) <- c(
  '許可證字號','類別','中文品名','核可日期','申請商','證況','保健功效相關成分','保健功效','保健功效宣稱','警語','注意事項')


class(data)
df <- data.frame(matrix(unlist(data),nrow =1770 , byrow=T),stringsAsFactors=FALSE)
df
