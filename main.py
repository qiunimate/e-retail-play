import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

from option import get_args

csv_path = "./online_retail_II.csv"
json_path = "./prob_dict.json"
model_path = "./model.pkl"
num_top = 5

def first_step_clean(df):
    # 1. drop rows non UK and then drop country column
    df = df.drop(df[df["Country"] != "United Kingdom"].index)
    df = df.drop(columns=["Country"])
    # 2. drop description column
    df = df.drop(columns=["Description"])
    # 3. drop rows with negative or 0 quantity
    df = df.drop(df[df["Quantity"] <= 0].index)
    # 4. drop rows with negative or 0 price
    df = df.drop(df[df["Price"] <= 0].index)
    # 5. drop invoice column starting with letters (cancelations)
    df = df.drop(df[df["Invoice"].str.contains("^[a-zA-Z]", regex=True)].index)
    # 6. only keep rows with stock code starting with 5 digits
    df = df[df["StockCode"].str.contains("^[0-9]{5}", regex=True)]

    # 7. convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df

def conditional_probability(invoice_products):
    count_dict = {}
    prob_dict = {}

    # iterrate over all invoices
    for _, row in tqdm(invoice_products.iterrows(), total=len(invoice_products), ascii=True):
        products = row['products']
        for product_a in products:
            count_dict[product_a] = count_dict.get(product_a, 0) + 1
            if product_a not in prob_dict:
                prob_dict[product_a] = {}
            for product_b in products:
                if product_a != product_b:
                    prob_dict[product_a][product_b] = prob_dict[product_a].get(product_b, 0) + 1

    # calculate conditional probability
    for product_a in tqdm(prob_dict, total=len(prob_dict), ascii=True):
        for product_b in prob_dict[product_a]:
            prob_dict[product_a][product_b] /= count_dict[product_a]

    # sort by probability (descending) and only keep the top5
    for product in tqdm(prob_dict, total=len(prob_dict), ascii=True):
        prob_dict[product] = {k: v for k, v in sorted(prob_dict[product].items(), key=lambda item: item[1], reverse=True)}
        prob_dict[product] = {k: v for k, v in list(prob_dict[product].items())[:num_top]}

    return prob_dict

def show_top(prob_dict, product, num=num_top):
    # make sure the product is in the dictionary, num is positive and smaller than num_top
    assert product in prob_dict.keys()
    assert num > 0
    assert num <= num_top

    interesting_dict = prob_dict[product]
    key_list = list(interesting_dict.keys())[:num]
    val_list = list(interesting_dict.values())[:num]

    # display the top n products with probability
    print(f"For the product {product}, the top {num} products with highest probability are:")
    for i in range(num):
        print(f"{key_list[i]}: {val_list[i]}")

    # visualize
    plt.bar(key_list, val_list)
    plt.xlabel("product")
    plt.ylabel("Probability")
    plt.show()

def basket_analyse(df):
    # load json if already calculated
    if os.path.exists(json_path):
        with open(json_path, 'r') as fp:
            prob_dict = json.load(fp)
        return prob_dict
    else:
        # we only need to know which Invoice contains which product
        df = first_step_clean(df)
        df_basket = df[["Invoice", "StockCode"]]
        # show in each invoice which products were bought
        invoice_products = df_basket.groupby('Invoice')['StockCode'].unique().to_frame(name='products')

        # calculation of invoice support (conditional probability) and save as json
        prob_dict = conditional_probability(invoice_products)

        # save as json
        with open(json_path, 'w') as fp:
            json.dump(prob_dict, fp)

        return prob_dict

def return_analyse(df):
    pass

if __name__=="__main__":
    args = get_args()
    if args.basket:
        if not os.path.exists(json_path):
            """load data"""
            df = pd.read_csv(csv_path)

            """basket analyse"""
            prob_dict = basket_analyse(df) # analysis of the basket
        else:
            with open(json_path, 'r') as fp:
                prob_dict = json.load(fp)
        show_top(prob_dict, args.product, num=args.top) # show top 5 products with highest probability for product 82482
    elif args.returning:
        if not os.path.exists(model_path):
            """load data"""
            df = pd.read_csv(csv_path)

            """returning customer analyse"""
            return_analyse(df)
        else:
            pass
    else:
        print("Please use choose a working mode")
        print("python main.py -h to see more info")
    

