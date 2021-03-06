# Flare-Factor

Deploy Machine Learning technologies, trained on publicly available data from the Texas Railroad Commission, to predict flaring volumes for oil and gas leases in Texas.
<p align="center">
  <img src="ec2_work/plots/flaring_image.jpg" alt="drawing" width="400"/>
</p>

## Why Flaring
Flaring, the burning of natural gas that is more expensive to get to market than to burn, has been a decades-long concern for environmental groups, and a frustrating problem for those who understand the energy industry. In my many years working in the oil and gas industry, I always understood why flaring happened and the cost/benefit of doing so - but it always bothered me. In my previous project I sought to quantify flaring energy wasted at the state level and put it into perspective. Knowing the opportunity cost of the energy wasted inspired me to keep digging.  Many enterprising groups are developing innovative solutions for using the energy wasted to flaring. Identifying, locating, and quantifying current flaring levels and locations is a cumbersome challenge with hundreds of thousands of wells reporting production and flaring monthly. This project will lay the groundwork for building a 'marketplace' for current flaring energy in Texas by investigating the feasibility of *predicting* flaring levels down to the lease level. 

## The Tech

![Tech Stack](ec2_work/plots/ff_tech.png)

## The Data
As previously mentioned, the Texas Railroad Commission (TRRC) dumps all the data they collect monthly. This amounts to about 135,000 reports every month in 2019/2020 from all active leases in Texas. The unzipped dataset exceeds 25 GB and contains upwards of 65,000,000 entries for production and ~35,000,000 entries that contain information on flaring, dating back to 1993. After initially reading in the full dataset, I determined that there was a good reason to focus on 2010 - Present data, as there is a clear departure from what was the normal in 2010, made evident in the plot below. 

<p float="center">
  <img src="ec2_work/plots/districts_colorsm.jpg" alt="drawing" width="300"/>
  <img src="ec2_work/plots/flare_by_district.png" alt="drawing" width="500"/>
</p>

The year 2010 is significant because it represents the year when shale oil revolution (crude oil extraction from non-permeable formations), truly began in Texas. This move expanded the boundaries of where oil production was possible, which in turn expanded where flaring would occur. The increasing production of oil from locations further removed from any sort of gas processing facility or market for the gas has led to a massive increase in flaring volumes, and a 'new normal' in Texas. One important feature about the data: every volume is self-reported by the operating companies the TRRC. 

## First Approach
I have a great deal of domain knowledge in the Oil & Gas industry, so I understood inherently what features would weigh heavily on the flaring volumes observed each month. The first phase was to digest the mass quantity of data received from the TRRC and create a reliable pipeline of the features I wanted to keep. Below is a summary of what features I included in my analysis and if they were engineered or taken directly from TRRC data. 

|Feature      | Description | Units   | Feature Engineering |
| ----------- | ----------- |---------| ------------------- |
|District     | TRRC Regional District | ID Number | Pulled straight from TRRC. District is the highest level geographical category. |
|County       | Name and ID number of County in Texas | Name / ID Number | Pulled straight from TRRC. Further classifies the lease geographically. |
|Operator     | Name and ID number of the company listed as the Operator | Name / ID Number | Pulled straight from TRRC. Company reporting the production / flaring. |
|Lease        | ID number of lease | ID Number |  Pulled straight from TRRC. Most granular geographical classification. Lease will be used to concatenate the lease location with the latitude / Longitude shape files.|
|Oil Produced | Volume of oil produced on the lease in the given year / month | Barrel (bbl) |   Pulled straight from TRRC |
|Gas Produced | Volume of natural gas produced on the lease in the given year / month | Thousand cubic feet (Mcf) |    Pulled straight from TRRC. natural gas withdrawn from the hydrocarbon reservoir |
|Casinghead  Gas Produced | Volume of casinghead gas produced on the lease in the given year / month | Thousand cubic feet (Mcf) |  Pulled straight from TRRC. Natural gas produced along with crude oil from oil wells. It contains either dissolved or associated gas or both  |
|Condensate produced | Volume of condensate produced on the lease in the given year / month | Barrel (bbl) |   Pulled straight from TRRC. Low-density mixture of hydrocarbon liquids that are present as gaseous components in the raw natural gas produced from the reservoir|
|Flaring Volume | Volume of gas flared or vented on the lease in the given year / month | thousand cubic feet (Mcf) |    Pulled straight from TRRC. This is the amount of energy wasted, and the focus of this project |
|Months from First Report| Tracking how many reporting periods have passed from first reporting production | Number of months | With this feature I wanted to capture the effects of decaying production over time. I simply converted all month / year entries from type(int) to datetime, the subtracted the current report date from the first report date. |
|Price of Oil | Dollars per Barrel in the given year / month | USD | Price of oil informs the oil produced. Oil produced informs the flaring volume. Created web scraping script to concatenate the year / month with the average price for the given cycle. Also includes availability to include forecasted oil prices in the future. 

## Data: Additional Considerations

While I will investigate flare volumes in an absolute sense (i.e. pure flare volume), it is important to understand that flaring only occurs because oil and gas are produced. If there is a lease that produces more oil than another one, it is likely that there will be more flaring. To account for this, I created some factors that will better depict the 'energy stewardship' of a particular lease. A few terms that will be introduced to account for this:
  1. **Waste ratio**: This is a unit-less metric that will capture, from a pure chemical potential energy standpoint, how much energy (in kilo-watt of giga-watt hours) was vented or flared, over total energy produced (in Kwh of Gwh) from a combination of oil, gas and condensate. 
  2. **Flaring Intensity**: This will capture the flaring volume (Mcf) as a factor of oil produced (in bbl). Because oil production is the main target of production in Texas, it will weigh most heavily on the flaring volumes. 
  
  These ratios will illuminate areas that may flare less gas, but are in fact more wasteful as a factor of how much is produced. 
  
 ## EDA: Finding the Variance
 
 ### District Level
  
 - The Boxplots break down the aggregate production and flaring values by district  
 - Also provided the flaring volumes normalized by both oil production and gas production
 - Normalized values show which districts are better than other at handling the gas
<p align="center"> 
  <img src="ec2_work/plots/second_pass/boxplotFlare Volumes by District (MMcf).png" width="460" /> 
</p>

<p float="center">
  <img src="ec2_work/plots/second_pass/boxplotOil Production by District (bbl).png" width="460" /> 
  <img src="ec2_work/plots/second_pass/boxplotGas Production by District (MMcf).png" width="460" />
</p>

<p float="center">
  <img src="ec2_work/plots/second_pass/boxplotFlare Gas - Oil Production Ratio by District.png" width="460" />
  <img src="ec2_work/plots/second_pass/boxplotFlare Gas - Gas Production Ratio by District.png" width="460" />
</p>

### Follow the Money
- Adding the price of oil provides insight into how economics play into production and flaring volume
- The dark red line is the average of all the districts, while the lightly shaded red area is 95% confidence interval

<p align="center"> 
  <img src="ec2_work/plots/second_pass/flare_price.png" width="500" /> 
</p>

<p float="center">
  <img src="ec2_work/plots/second_pass/oil_prd_price.png" width="475" /> 
  <img src="ec2_work/plots/second_pass/gas_prd_price.png" width="475" />
</p>

<p float="center">
  <img src="ec2_work/plots/second_pass/flare_oil_ratio.png" width="475" />
  <img src="ec2_work/plots/second_pass/flare_gas_ratio_price.png" width="475" />
</p>

## Operators
- Top 25 operators are defined as the 25 companies that contributed most to the total flaring volume
- We can see that the companies that contributed most to flaring, contributed a similar amount to oil production

<p float="center"> 
  <img src="ec2_work/plots/second_pass/operator_contribution.png" width="500" />
  <img src="ec2_work/plots/second_pass/operator_contribution_oil.png" width="500" />
</p>

## Decay
- To observe how decay effects produced energy and waste ratio, these plots capture a small sample of these values. 
<p float="center"> 
  <img src="ec2_work/plots/waste_months.jpg" width="500" />
  <img src="ec2_work/plots/energy_months.jpg" width="500" />
</p>

## Model
In picking a model to run to try to predict flaring volumes or energy wasted, there are a few important items to consider:
  1) Flaring often happens randomly. If equipment fails and gas has nowhere else to go, flaring will happen at a high rate until the equipment is fixed
  2) There are millions of observations...any model will take a LONG time to train. 
  3) While predicting how much gas is flared in any given month based on historical data is interesting, the likelihood that this will provide anything usable is low. This is       because even if the model predicts well, it won't provide direct insight into the areas that will be flaring in the future. 

I landed on using a Gradient Boosting model because I wanted an ensemble method less prone to over-fitting (if possible), and the non-linearity of the data. The initial results were predictable. 

<p align="center"> 
  <img src="ec2_work/plots/gbr_staged_predict_0.1.png" width="600" /> 
</p>
 What we get is a VERY overfit model. No matter the amount of tweaking, the predictability of precise flaring volumes is not feasible. 
 
 Best Results:  
 
 - **Train R2: 0.572**
 - **Test R2: 0.014**
 
## Confounding Data
-Some problems that I uncovered after getting poor predictive results. The most problematic is below. This shows that, on the same lease and same reporting period, different operators report production and flaring separately. I will have to address this moving forward. 
<p align="center"> 
  <img src="ec2_work/plots/lease_two_ops.png" width="900" /> 
</p>

## First Approach Conclusions
While predicting exact flare volumes has proven to be unfeasible, there is value in creating a model that can predict whether or not a lease in the future is likely to flare in various categories (high, medium, low, for example). I will have to change my approach, but I now have an effective data pipeline and ability to access all the information I need to pivot my strategy. 

