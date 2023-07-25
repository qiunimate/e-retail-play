# E-retail-play

With the dataset of online retailer (https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci), in this work, 2 tasks were studied, namely

- basket analysiss
- returned customers

The codes were firstly finished in ipynb files, and then transferred manually to python files, to soomthen the operations.

This work started from Sumday 23/07 and done during some spare times

## Basket analysis
To analyze which product is most likely to be purchased knowing that one specific product is already bought.

`python ./main.py --basket --top=5 --product=82482`
* product: the StockCode of the product
* top: show the top n choises knowing product is chosen

### Explanation:

#### Data cleaning:
Drop unwanted products, like negative price, cancelled invoice, etc.
I did not give negative points to the cancelled invoice because they could be related with porducts out of the invoice, so it is not sure. It would be better that we **only consider sure things**, like valid invoice.

#### Conditional probability
The task is basically a conditioanl probability, in other words, knowing A is bought, which product possesses a highest probability.

There are 2 dicts, count_dict for how many times a product appears in all invoices and prob_dict to count in invoices containing A, how many B/C/D are taken seperately.

e.g.   
 count_dict = {A:50, B:20, C:100,...}   
 prob_dict = {A:{B:10, C:30,...}, B:{A:10,C:20},...}

There is another way to mesure which is to use the apriori algo, but it only takes into account the probability the appearance of some sets overall, it is not pricise in this case because it more like a conditional probability (P(A|B))

### Discussion
In this task, traditional statistical method was utilized to complete the target, because the relation is already clear.  


With regard to the results, with the json file, we can get the results within very short time (less than )

- **Product Bundling**: Based on the conditional probabilities, bundle product A with the most likely associated products (e.g., B and C) as a package deal. 

- **Selling recomandation**: When a customer adds product A to their cart or completes a purchase, recommend the top-n associated products. 

- **Promotions and Discounts**

In the future, we can further work on recommandations based on multi-product, classification of costomers, etc. Also, we can select the top results with number n or or with **a threshold of the probability**.

## Returned customers
In this part, we will try to predict if a customer is going to return to us and buy something in the following quarter. We only take the data of the **last quarter** to do this prediction.

`python ./main.py --returning`

Unlike the previous one, we take machine learning models to do this prediction, so we have to some data preparations for the model.

After some cleaning, the input of the data comtains:
- sum of bought quantities in the last quarter
- mean of bought quantities per invoice in the last quarter
- sum of cost in the last quarter
- mean of cost per invoice in the last quarter
- The quarter of the training data (one hot encoded)
- types contained in the training data (binary encoderd)
- number of types

Notably, the latest month is not taken into account because there is no data in the following month. The number of data of label 1 is a little bit more than label 0, a simple down sampling was taken to balance the dataset.

### Difficulties
2 models (random forest and xgboost) was tried to do this prediction, the default versions did not gave us satisfying results, so I started hyperparameter tuning. But when trees go deeper, overfitting shows up. To solve that, I tried to do K-fold validation and reset the min_samples_leaf. 

### Discussion
Though the overfitting is a little bit better than with k-fold validation, the score is still not good enough. (around 68%). More networks and even neuron networks could be tried to improve the performance, moreover, we can increase the training information, e.g., not only use the last quarter, but use the last year or even all history to do the prediction.

