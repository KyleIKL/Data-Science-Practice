# Retail Product Availability Prediction Report

## 1. Objective
The objective of this study is to analyze the factors influencing product size availability in a retail dataset and to build machine learning models capable of predicting the number of available sizes (`available_size_count`) based on product attributes and pricing information.

## 2. Dataset Overview
The dataset contains 311 observations and 9 predictive variables, including product information, pricing, and categorical attributes. Key variables include price_local (original product price), sale_price_local (discounted sale price), category and subcategory (product classification), gender_segment (target customer segment), size_label (size category information), product_name (product identifier or SKU), color_name (product color), and seen_market_count (the number of markets where the product appears). The target variable used in this study is available_size_count, which represents the number of sizes currently available for a product and reflects inventory availability and demand conditions.

## 3. Methodology
Two ensemble machine learning models were applied in this study: Random Forest Regressor and Gradient Boosting Regressor. The dataset was divided into training and testing subsets using an 80/20 split, where 80% of the data was used for training the models and 20% for evaluating performance. Model performance was assessed using three metrics: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R² (Coefficient of Determination).

## 4. Model Performance
The Random Forest model achieved an RMSE of 1.8578, an MAE of 0.5273, and an R² of 0.911, indicating that it explained approximately 91% of the variance in the target variable. The Gradient Boosting model performed better overall, achieving an RMSE of 1.1822, an MAE of 0.2679, and an R² of 0.964. This means that the Gradient Boosting model explained about 96% of the variance and provided the most accurate predictions among the tested models.

## 5. Feature Importance Analysis
Feature importance analysis based on the Random Forest model indicates that pricing variables play the most significant role in predicting available sizes. The most influential predictors include price_local, subcategory, sale_price_local, gender_segment, and size_label. Less influential variables include product_name, color_name, seen_market_count, and category. This suggests that both pricing strategies and product segmentation characteristics contribute to explaining inventory availability.

## 6. Key Findings
The analysis reveals several important findings. First, pricing variables strongly influence the number of sizes remaining in stock. Both original price and sale price are among the most influential predictors, indicating that pricing strategies affect product sell-through rates. Second, product segmentation variables such as subcategory and gender segment play a major role, suggesting that different product categories and customer segments follow different inventory patterns. Third, SKU-level differences exist, as indicated by the influence of product_name, implying that certain individual products may experience different demand dynamics.

## 7. Business Implications
From a business perspective, these results suggest that pricing strategies have a direct impact on inventory turnover and size availability. Retailers may use predictive models like this to anticipate potential stock shortages and adjust inventory allocation accordingly. Additionally, understanding which product segments experience faster size depletion can help retailers optimize inventory planning and improve pricing strategies.

## 8. Conclusion
This study demonstrates that machine learning models can effectively predict product size availability using product attributes and pricing information. Among the models tested, Gradient Boosting achieved the best performance, while pricing variables and product segmentation features were identified as the most influential predictors. These findings highlight the importance of pricing strategies and product structure in managing retail inventory availability and suggest that predictive analytics can provide valuable support for retail decision-making.