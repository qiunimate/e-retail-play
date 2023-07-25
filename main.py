import pandas as pd
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import xgboost as xgb
import pickle
import os
from functools import reduce

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
    
def clean(df):
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
    # 8. drop rows with missing values
    df = df.dropna()
    # 9. convert CustomerID to int
    df["Customer ID"] = df["Customer ID"].astype(int)
    # 10. calculate total price
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    # 11. convert StockCode to its respective index
    # get a unique list of StockCode
    stock_code_list = df["StockCode"].unique().tolist()
    # get a dict of StockCode and its index and the reverse
    stock_code_dict = {stock_code_list[i]: i for i in range(len(stock_code_list))}
    df["StockCode"] = df["StockCode"].apply(lambda x: stock_code_dict[x])
    return df, stock_code_dict

def get_customer_invoice(df):
    grouped_data = df.groupby('Customer ID').agg({
        'Invoice': lambda invoices: list(invoices.unique())  # Convert invoices to lists
    }).reset_index()
    grouped_data.columns = ['Customer ID', 'Invoice']
    return grouped_data

def get_invoice_info(df):
    invoice_data = df.groupby(['Invoice']).agg({
        'Quantity': "sum",
        "TotalPrice": "sum",
        "InvoiceDate": "first",
        "StockCode": lambda x: list(x.unique())
    }).reset_index()
    return invoice_data

# get the quarter count of a date (from 2099)
def quarter_count(date, start_year=2009):
    date = pd.to_datetime(date)
    return (date.year - start_year) * 4 + date.quarter

# tell if a costomer has bought in the next certain quarter
def get_label(quater_count, quarter_list):
    if quater_count + 1 in quarter_list:
        return 1
    else:
        return 0

def feature_preparing(df, grouped_data, invoice_data):
    feature_list = []
    max_quater = max(list(invoice_data["InvoiceDate"].apply(lambda x: quarter_count(x))))

    for i, row in tqdm(grouped_data.iterrows(), total=len(grouped_data), ascii=True):
        invoice_list = row['Invoice']
        date_list = invoice_data[invoice_data['Invoice'].isin(invoice_list)]['InvoiceDate'].tolist()
        date_list = [quarter_count(date) for date in date_list]
        date_list_unique = sorted(list(set(date_list)))

        # for each unique quarter, get the corresponding invoices and calculate the features
        for date in date_list_unique:
            # the last quarter is not used for training
            if date == max_quater:
                continue
            list_index = [invoice_list[i] for i, x in enumerate(date_list) if x == date]
            selected_invoices = invoice_data[invoice_data['Invoice'].isin(list_index)]
            union_list = reduce(lambda x, y: set(x).union(y), selected_invoices['StockCode'])
            feature_list.append([selected_invoices['Quantity'].sum(), selected_invoices['Quantity'].mean(), 
                                selected_invoices['TotalPrice'].sum(), selected_invoices['TotalPrice'].mean(),
                                date%4, len(union_list), union_list, get_label(date, date_list_unique)])

    # convert to dataframe
    feature_df = pd.DataFrame(feature_list, columns=['Quantity_sum', 'Quantity_mean', 'TotalPrice_sum', 'TotalPrice_mean', 'Quarter', 'types', 'type_list', 'Label'])
    
    """encode the types and quarter because they are categorical"""
    mlb = MultiLabelBinarizer()
    # encode the types (1 if bought, 0 if not)
    one_hot_encoded_df = pd.DataFrame(mlb.fit_transform(feature_df['type_list']), columns=mlb.classes_)
    feature_df = pd.concat([feature_df, one_hot_encoded_df], axis=1).drop(columns=['type_list'])

    # encode the quarter (one hot)
    one_hot = pd.get_dummies(feature_df['Quarter'])
    # set one hot column names "Quarter_0", "Quarter_1", etc. to avoid confusion
    one_hot.columns = ["Quarter_" + str(i) for i in range(4)]
    feature_df = pd.concat([feature_df, one_hot], axis=1).drop(columns=['Quarter'])

    # drop 2000 random data with label 1 to balance the data
    # c.f. notebook to see more about details of decisions
    feature_df = feature_df.drop(feature_df[feature_df['Label'] == 1].sample(n=2000, random_state=42).index)

    return feature_df

def cross_validation(model, _X, _y, _cv=5):
      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True,
                               return_estimator=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean(),
              "estimator": results['estimator']
              }

def plot_result(x_label, y_label, plot_title, train_data, val_data):
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

def return_analyse(df):
    # need the dict for future use
    df, stock_code_dict = clean(df)
    # get mapping between customer id and invoice
    grouped_data = get_customer_invoice(df)
    # get mapping between invoice and invoice info
    invoice_data = get_invoice_info(df)
    # get feature dataframe
    feature_df = feature_preparing(df, grouped_data, invoice_data)

    labels = feature_df['Label']
    features = feature_df.drop(columns=['Label'])
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)
    # c.f. notebook to see more about details of decisions
    xgb_best_i = 0.05
    xgb_best_j = 3
    xgb_result = cross_validation(xgb.XGBClassifier(objective="binary:logistic", random_state=42, verbose=1, learning_rate=xgb_best_i, max_depth=xgb_best_j), X_train, y_train, _cv=5)

    for i in range(len(xgb_result["Training Accuracy scores"])):
        print(f"Training Accuracy: {xgb_result['Training Accuracy scores'][i]}, Validation Accuracy: {xgb_result['Validation Accuracy scores'][i]}")
        print(f"Training Precision: {xgb_result['Training Precision scores'][i]}, Validation Precision: {xgb_result['Validation Precision scores'][i]}")
        print(f"Training Recall: {xgb_result['Training Recall scores'][i]}, Validation Recall: {xgb_result['Validation Recall scores'][i]}")
        print(f"Training F1: {xgb_result['Training F1 scores'][i]}, Validation F1: {xgb_result['Validation F1 scores'][i]}")
        print("="*50)

    for i in range(len(xgb_result["Training Accuracy scores"])):
        plot_result("Folds", "Accuracy", "Accuracy Scores", xgb_result["Training Accuracy scores"], xgb_result["Validation Accuracy scores"])
        plot_result("Folds", "Precision", "Precision Scores", xgb_result["Training Precision scores"], xgb_result["Validation Precision scores"])
        plot_result("Folds", "Recall", "Recall Scores", xgb_result["Training Recall scores"], xgb_result["Validation Recall scores"])
        plot_result("Folds", "F1", "F1 Scores", xgb_result["Training F1 scores"], xgb_result["Validation F1 scores"])

    # save the model
    pickle.dump(xgb_result["estimator"][2], open(model_path, 'wb'))



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
        df = pd.read_csv(csv_path)
        return_analyse(df)
    else:
        print("Please use choose a working mode")
        print("python main.py -h to see more info")
    

