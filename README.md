# e-retail-play



## basket analysis
To analyze which product is most likely to be purchased knowing that one specific product is already bought

In this part we have an result file: **prob_dict.json**


`python ./main.py --basket --top=5 --product=82482`
- product: the StockCode of the product
- top: show the top n choises knowing product is chosen

### explanation:

#### data cleaning:
Drop unwanted products, like negative price, cancelled invoice, etc.
I did not give negative points to the cancelled invoice because they could be related with porducts out of the invoice, so it is not sure. It would be better that we only consider sure things, like valid invoice.

#### conditional probability
There are 2 dicts, count_dict for how many times a product appears in all invoices and prob_dict to count in invoices containing A, how many B/C/D are taken seperately.

 count_dict = {A:50, B:20, C:100,...}

 prob_dict = {A:{B:10, C:30,...}, B:{A:10,C:20},...}

There is another way to mesure which is to use the apriori algo, but it only takes into account the probability the appearance of some sets overall, it is not pricise in this case because it more like a conditional probability (P(A|B))

### discussion
In this task, traditional statistical method was utilized to complete the target, because the relation is already clear.  


With regard to the results, with the json file, we can get the results within very short time (less than )

- **Product Bundling**: Based on the conditional probabilities, bundle product A with the most likely associated products (e.g., B and C) as a package deal. 

- **Selling recomandation**: When a customer adds product A to their cart or completes a purchase, recommend the top-n associated products. 

- **Promotions and Discounts**

In the future, we can further work on recommandations based on multi-product, classify costomers, etc.
Also, we can select the top results with number n or with **the threshold of the probability**.


