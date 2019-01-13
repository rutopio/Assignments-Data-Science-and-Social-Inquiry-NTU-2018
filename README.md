# 資料科學與社會研究 Data Science and Social Inquiry
## Assignment written by ChingRu
### 2018, The Department of Economics, National Taiwan University 

***

As title，就是放我寫的Assignment的地方喇

# Assignment #2 No later than 10/07 23:59 (3 points)

- 繳交方式：統一上傳. rmd 檔案與. rmd 檔案所 generate 出來的. nb.html 或. html 兩個檔案（務必成功 generate 出其中一種. html 檔）。將兩個檔案壓縮後上傳。如果不會操作 R markdown 可以參考這個[影片說明](https://youtu.be/xVXUZShYfEI)。
- 完成課堂 R01_2 vector.Rmd 與 R01_4 dataframe tp theft.Rmd 最末的練習。請務必用 R notebook 或 R Markdown 編寫，助教屆時將會優先看. html 檔，沒有成功編譯成. html 檔的會扣分。
- 下載 R01_5prac load and summarize tweet data.Rmd，練習完，並嘗試對這筆資料做點分析，並提出你的發現。

```
x.a <- rnorm(1000, 1, 10)
# 1.1 Filter out extreme values out of two standard deviations
a1 <- x.a[!abs(x.a - mean(x.a)) > 2*sd(x.a)]

# 1.2 Plotting the distribution of the remaining vector x.a
# 1.3 Calculate the 25% 50% and 75% quantile of vector x.a. You may google "quantile r"
# 1.4 Get the number between 25% to 75% and assign to x.a1
# 1.5 Plotting x.a1
x.b <- c("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k")
# 2.1 Get only elements at odd positions and assign to x.b1
x.b1 <- x.b[(1:length(x.b)) %% 2 == 1]

# 2.2 Truncate the first 2 elements and the last 2 elements and assign to x.b2
```

# Assignment #3 No later than 10/14 23:59 (3 points)

請在下列三題中選兩題做：

- 第一題：在 Paid maternity leave 的練習中，我們繪製了最後調查年代為 level 5 的兩張不同國家長條圖。請嘗試修改程式碼，畫出 level 4 的那兩張長條圖。
- 第二題：下載或直接讀取[空氣品質資料](https://taqm.epa.gov.tw/taqm/aqs.ashx?lang=tw&act=aqi-epa&ts=1538931940046)，並依照 **R02_3 read json ubike.Rmd** 的方法繪製空氣品質地圖。上色應依照空氣品質監測網的上色方式。
- 第三題：讀取以下資料集中的任一個並轉成 Data.frame（當然是 json 檔，不然 csv 編錯還得了）
  - 違規藥品廣告資料 (<https://data.gov.tw/dataset/14196>)
  - 違規化妝品廣告資料（[https://data.gov.tw/dataset/14198）](https://data.gov.tw/dataset/14198%EF%BC%89)
  - 健康食品資料集（<https://data.gov.tw/dataset/6951>)

# Assignment #4 Try to scrape at least one, at most 10 pages of one sites listed as follows:

- retrieve essential data to data.frame then dump those data to `.rds` file by `saveRDS(data, f)`.
- url_dcard <- "<https://www.dcard.tw/f/relationship>"
- url_104 <- "<https://www.104.com.tw/jobs/search/?ro=0&keyword=%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90&area=6001001000&order=1&asc=0&kwop=7&page=9&mode=s&jobsource=n104bank1>"
- url_cnyes <- "<https://news.cnyes.com/api/v3/search?q=%E9%B4%BB%E6%B5%B7>"
- Try to find at least one website whose webpage is rendered from json files.

# Assignment #5 Scraping news report

- Try to scrape and parse one news website (必須要是非得剖析背後的 html 不可的網站，例如鉅亨網背後是 json，那就沒必要用 html)
- e.g., ltn.news, apple news, udn news, ...
- with one query to get at least 100 news reports
- Store your data as .rds or .rda
- Zip your data, .rmd, .html file into a zipped file, then upload

# Assignment #6 text analysis basic using dplyr, tidytext, and ggplot2

- Try to analyze typhoon disaster news data in Chinese 
- (options) Try to implement the following posts [Who wrote the anti-Trump New York Times op-ed? Using tidytext to find document similarity](http://varianceexplained.org/r/op-ed-text-analysis/)
- (options) Scrape ptt posts and comments and try to do some analysis.