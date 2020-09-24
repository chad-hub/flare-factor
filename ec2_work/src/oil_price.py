# %%
import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime
from selenium import webdriver
from bs4 import BeautifulSoup
import urllib3
from requests import get
import time
import re
import os
import math

pd.options.display.max_rows = 8
# %%
def oil_futures():
  #write url based on current time
  url = "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/425/G?pageSize=50&_="
  urlext = np.round(time.time(), decimals=0)
  urlext = str(urlext)
  url = url + urlext

  # create a new  session
  # System.setProperty("webdriver.chrome.driver","/usr/bin/chromedriver");
  # WebDriver driver = new ChromeDriver();
  driver = webdriver.Chrome(executable_path='/usr/bin/chromedriver')
  driver.get(url)
  soup = BeautifulSoup(url)

  #scrape site source page
  urltext = soup.findAll(text=True)
  urltext = driver.current_url
  httptext = urllib3.PoolManager()
  responsetext = httptext.request('GET', urltext)

  #close browser
  driver.close()
  #convert BeautifulSoup object to text
  souptext = BeautifulSoup(responsetext.data)
  souptext2 = str(souptext)
  #create lists for pricing and dates
  settlePri = []
  settleDate = []

  #state the patterns to search on within raw data
  pricetxt2 = '"priorSettle":"\d+.\d\d"'
  pricetxt3 = '"priorSettle":"-"'

  #find the above patterns in the text and append them to their corresponding lists from above
  p = re.compile("(%s|%s)" % (pricetxt2, pricetxt3)).findall(souptext2)
  d = re.findall('"expirationDate":"\d{8}"', souptext2)
  if p:
      settlePri.append(p)
  if d:
      settleDate.append(d)
  #combine those two lists
  pricelist = {'Date':d,'Oil_Price':p}
  #convert to dataframe
  pricelist = pd.DataFrame(pricelist)
  #remove unnecessary text, format numbers and dates, remove non-numeric price place holders from web site
  pricelist['Date'] = pricelist['Date'].map(lambda x: x.replace('"expirationDate":"', ""))
  pricelist['Date'] = [datetime(year=int(x[0:4]), month=int(x[4:6]), day=int(x[6:8])) for x in pricelist['Date']]
  pricelist['Month'] = pricelist['Date'].dt.month.astype(int)
  pricelist['Year'] = pricelist['Date'].dt.year.astype(int)
  pricelist['Oil_Price'] = pricelist['Oil_Price'].map(lambda x: x.replace('"priorSettle":"', "").rstrip('"'))
  pricelist['Oil_Price'] = pricelist['Oil_Price'].replace('-', '0.00')
  pricelist['Oil_Price'] = pricelist['Oil_Price'].astype(float)

  #final dataframe
  pricelist = pricelist[['Year', 'Month', 'Oil_Price']]
  return pricelist



# %%
def oil_past():
  url = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=pet&s=rwtc&f=m"
  driver = webdriver.Chrome(executable_path='/usr/bin/chromedriver')
  driver.get(url)

  soup = BeautifulSoup(url)
  urltext = soup.findAll(text=True)
  urltext = driver.current_url
  httptext = urllib3.PoolManager()
  responsetext = httptext.request('GET', urltext)
  souptext = BeautifulSoup(responsetext.data)
  Price_table = souptext.find("table", {"class": "FloatTitle"})
  driver.close()


  Histprice = pd.DataFrame()
  rowsp = Price_table.find_all('tr')

  for row in rowsp:
      cols=row.find_all('td')
      cols=[x.text.strip() for x in cols]
      cols = pd.DataFrame(cols)
      Histprice = pd.concat([Histprice,cols], ignore_index=True, axis=1)

  Histprice = Histprice.dropna(axis = 1)
  Header = Histprice.iloc[0]
  Histprice = Histprice[1:]
  Histprice.columns = Header

  months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  monthsdict = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
  months = np.asarray(months)

  Histprice = Histprice.set_index(months, drop=True)
  Histprice2 = pd.DataFrame(Histprice.unstack(level=0))
  Histprice2.index = Histprice2.index.set_names(['Year', 'Monthstr'])
  Histprice2 = Histprice2.reset_index()
  Histprice2['Month']= Histprice2['Monthstr'].map(monthsdict)
  Histprice2.columns = ['Year', 'Monthstr', 'Oil_Price', 'Month']
  Histprice2 = Histprice2[['Year', 'Month', 'Oil_Price']]
  Histprice2[['Year', 'Month', 'Oil_Price']] = Histprice2[['Year', 'Month','Oil_Price']].apply(pd.to_numeric)
  Histprice2 = Histprice2.dropna()

  return Histprice2

def price_concat(df1, df2, year):
  df = pd.concat((df1, df2))
  df = df[df['Year'] >= year]
  df.columns = ['CYCLE_YEAR', 'CYCLE_MONTH', 'OIL_PRICE']
  return df

def main():
  oil_future_df = oil_futures()
  oil_past_df = oil_past()
  price_concat(oil_future_df, oil_past_df, 2010)

# %%
if __name__ =='__main__':
  oil_df = main()
# %%
