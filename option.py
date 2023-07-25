import argparse

def get_args():    
    parser = argparse.ArgumentParser(description='fun with online retail datapy')
    parser.add_argument('--basket', action='store_true', help='basket analyse')
    parser.add_argument('--returning', action='store_true', help='predict if a customer will return in a quarter')
    parser.add_argument('--top', type=int, default=5, help='show top 5 products with highest probability for product 82482')
    parser.add_argument('--product', type=str, default="82482", help='product id')
    args = parser.parse_args()
    return args