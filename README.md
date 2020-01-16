## HS Code Identification Model


### HS CODE: 'Harmonized Commodity Description and Coding System' (무역 관세코드)

- HS Code is a international product naming System  
- Facilitate the collection of trade-related statistics and tariffs in International Trade  
- The first six digits of HS CODE are common worldwide  


- Current practice to HS CODE Identification
	- Use of integrated tariff statistics classification table: Because table is composed of academic and technical terms, it is difficult to search
	- Database-based search1): When searching for Products that are not in the transaction data base, such as new products, product that never been traded, and names that have been misspelled
	
	- → When exporting in small and medium enterprises, it is difficult to identify HS CODE by existing methods for product developers unfamiliar with trade. And There is lack of research on automatic HS CODE identification technique


#### Objectives
- Development of HS CODE Identification Model using actual trade item name
	- Word Embedding – Embedding actual trade item name using Word2Vec and fastText method
	- Deep Learning Model – Apply Classification model such as FCN, LSTM, CNN


#### Data Example

|Product name|HS CODE|
|---|---|
|MAYPRIDE CANNED SWEET KERNELS CORN VACUUM PACKED PACKING OZ NW DW NORMAL LID HTS INTENDED TRANSHIPMANT SINGAPORE|200580|
|MITSUBISHI MITSUBISHI PASSENGER ELEVATORPARTS MODEL SHOP ORDERNO SHOP ORDER NO|843139|
|BRAVO ONE THOUSAND EIGHT HUNDRED CARTON SARDINES TOMATO SAUCE EOE ONE THOUSAND EIGHT HUNDRED CARTON SARDINES TOMATO SAUCE|210320|

  
    
#### Process

- Product name:HS CODE = 1:1 Matched Data
- Preprocessing (modules.py)
- Using Word2vec or FastText, Product names --> Weight matrix
- FCN, LSTM, CNN Model Train
- New Data Input -> Predict 2 or 4 digits of HS Code(6digits) (6 digits has too many classes and there is not enough data for each class)
- Expand to 6 digits(Using Cosine Similarity)



