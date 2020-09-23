# Flare-Factor

Deploy Machne Learning technologies, trained on publically avaiable data from the Texas Railroad Commission, to predict flaring volumes for oil and gas leases in Texas.
<p align="center">
  <img src="ec2_work/plots/flaring_image.jpg" alt="drawing" width="400"/>
</p>

## Why Flaring
Flaring, the burning of natural gas that is more expensive to get to market than to burn, has been a decades-long concern for environmental groups, and a frustrating problem for those who understand the energy industry. In my many years working in the oil and gas industry, I always understood why flaring happened and the cost/benefit of doing so - but it always bothered me. In my previous project I sought to quantify flaring energy wasted at the state level, and put it into perspective. Knowing the opportunity cost of the energy wasted inspired me to keep digging.  Many enterprising groups are developing innovative solutions for using the energy wasted to flaring. Identifying, locating and quantifying current flaring levels and locations is a cumbersome challenge with hundreds of thousands of wells reporting production and flaring monthly. This project will lay the ground work for building a 'marketplace' for current flaring energy in Texas by investigating the feasibility of *predicting* flaring levels down to the lease level. 

## The Tech

![Tech Stack](ec2_work/plots/ff_tech.png)

## The Data
As previously mentioned, the Texas Railraod Commission (TRRC) dumps all the data they collect monthly. This amounts to about 135,000 reports monthy in 2019/2020 from all active leases in Texas. The unzipped dataset exceeds 25 GB, and contains upwards of 65,000,000 entries for production and ~35,000,000 entries that contain information on flaring, dating back to 1993. After initally reading in the full dataset, I determined that there was a good reason to focus on 2010 - Present data, as there is a clear departure from what was the normal in 2010, made evident in the plot below. 

<p align="center">
  <img src="ec2_work/plots/flare_by_district.png" alt="drawing" width="700"/>
</p>

The year 2010 is significant because it represents the year when shale oil revolution (crude oil extraction from non-permeable formations),  truly began in Texas. This move expanded the boundaries of where oil production was possible, which in turn expanded where flaring would occur. The increasing production of oil from locations further removed from any sort of gas processing facility or market for the gas has led to a massive increase in flaring volumes, and a 'new normal' in Texas. One important feature about the data: every volume is self-reported by the operating companies the the TRRC. 

## First Approach
I have a great deal of domain knowledge in the Oil & Gas industry, so I understood inherently what features would weigh heavily on the flaring volumes observed each month. The first phase was to digest the mass quantity of data received from the TRRC, and create a reliable pipeline of the features I wanted to keep. Below is a summary of what features I included in my analysis and if they were engineered or taken directly from TRRC data. 

|Feature      | Description | Units   | Feature Engineering |
| ----------- | ----------- |---------| ------------------- |
|District     | TRRC Regional District | ID Number | Pulled straight from TRRC |
|County       | Name and ID number of County in Texas | Name / ID Number | Pulled straight from TRRC |
|Operator     | Name and ID number of the company listed as the Operator | Name / ID Number | Pulled straight from TRRC |
|Lease        | ID number of lease | ID Number |  Pulled straight from TRRC |
|Oil Produced | Volume of oil produced on the lease in the given year / month | Barrel (bbl) |   Pulled straight from TRRC |
|Gas Produced | Volume of natural gas produced on the lease in the given year / month | Thousand cubic feet (Mcf) |    Pulled straight from TRRC |
|Casinghead  Gas Produced | Volume of casinghead gas produed on the lease in the given year / month | Thousand cubic feet (Mcf) |  Pulled straight from TRRC |
|Condensate produced | Volume of condensate produced on the lease in the given year / month | Barrel (bbl) |    Pulled straight from TRRC |
|Flaring Volume | Volume of gas flared or vented on the lease in the given year / month | thousand cubic feet (Mcf) |    Pulled straight from TRRC 
|Months from First Report| Tracking how many reporting periods have passed from first reporting production | Number of months | With this feature I wasnted to capture the effects of decaying production over time. I simply converted all month / year entries from type(int) to datetime, the subtracted the current report date from the first report date.|
|Price of Oil | Dollars per Barrel in the given year / month | USD | Price of oil informs the oil produced. Oil produced informs the flaring volume. Created web scraping script to concatenate the year / month with the average price for the given cycle. Also includes availablility to include forecasted oil prices in the future. 





