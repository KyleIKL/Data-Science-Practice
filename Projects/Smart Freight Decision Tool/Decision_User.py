# Initiailize:
# Change the path to your own absolute path where the model and scaler are saved.
# Format of the input features should match the order and type:
# Category Feature Input Guidelines

# 1. Transport Mode 
# Valid options: 'Air', 'Rail', 'Road', 'Sea'
# Corresponding encoding:
#   'Air': 0
#   'Rail': 1
#   'Road': 2
#   'Sea': 3
# Example Input: Choose one of the options: 'Air', 'Rail', 'Road', 'Sea'

# 2. Product Category 
# Valid options: 'Automotive', 'Electronics', 'Perishables', 'Pharmaceuticals', 'Textiles'
# Corresponding encoding:
#   'Automotive': 0
#   'Electronics': 1
#   'Perishables': 2
#   'Pharmaceuticals': 3
#   'Textiles': 4
# Example Input: Choose one of the options: 'Automotive', 'Electronics', 'Perishables', 'Pharmaceuticals', 'Textiles'

# 3. Weather Condition 
# Valid options: 'Clear', 'Fog', 'Hurricane', 'Rain', 'Storm'
# Corresponding encoding:
#   'Clear': 0
#   'Fog': 1
#   'Hurricane': 2
#   'Rain': 3
#   'Storm': 4
# Example Input: Choose one of the options: 'Clear', 'Fog', 'Hurricane', 'Rain', 'Storm'

# 4. Origin Port 
# Valid options: 'Antwerp', 'Busan', 'Dubai', 'Hamburg', 'Los Angeles', 'Rotterdam', 'Shanghai', 'Singapore'
# Corresponding encoding:
#   'Antwerp': 0
#   'Busan': 1
#   'Dubai': 2
#   'Hamburg': 3
#   'Los Angeles': 4
#   'Rotterdam': 5
#   'Shanghai': 6
#   'Singapore': 7
# Example Input: Choose one of the options: 'Antwerp', 'Busan', 'Dubai', 'Hamburg', 'Los Angeles', 'Rotterdam', 'Shanghai', 'Singapore'

# 5. Destination Port 
# Valid options: 'Antwerp', 'Busan', 'Dubai', 'Hamburg', 'Los Angeles', 'Marseille', 'Rotterdam', 'Shanghai', 'Singapore'
# Corresponding encoding:
#   'Antwerp': 0
#   'Busan': 1
#   'Dubai': 2
#   'Hamburg': 3
#   'Los Angeles': 4
#   'Marseille': 5
#   'Rotterdam': 6
#   'Shanghai': 7
#   'Singapore': 8
# Example Input: Choose one of the options: 'Antwerp', 'Busan', 'Dubai', 'Hamburg', 'Los Angeles', 'Marseille', 'Rotterdam', 'Shanghai', 'Singapore'

# Number Feature Input Guidelines

# 1. Distance_km (运输距离，单位：千米)
#   - Valid input range: positive number
#   - Example range: from 10 km to 10000 km
#   - Description: Represents the transport distance. A higher value means longer transport distance, which affects the transportation cost and time.

# 2. Weight_MT (重量，单位：公吨)
#   - Valid input range: positive number
#   - Example range: from 0.1 to 5000 metric tons
#   - Description: Represents the weight of the cargo. A higher value means heavier cargo, which influences transport costs.

# 3. Fuel_Price_Index (燃油价格指数)
#   - Valid input range: positive number
#   - Example range: from 0 to 10
#   - Description: A higher index means higher fuel prices, which increase the transport costs.

# 4. Geopolitical_Risk_Score (地缘政治风险分数)
#   - Valid input range: from 0 to 10
#   - Example range: from 0 (no risk) to 10 (high risk)
#   - Description: Represents the geopolitical risk in the transport region. A higher value indicates more political instability or conflict.

# 5. Carrier_Reliability_Score (承运商可靠性评分)
#   - Valid input range: from 0 to 1
#   - Example range: from 0 to 1
#   - Description: Represents the reliability of the carrier. A value closer to 1 indicates higher reliability and better punctuality.




import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. 加载训练好的模型和标准化器
classifier = joblib.load('E:\\Project95\\Projects\\Smart Freight Decision Tool\\best_xgb_model.joblib')  # 加载训练好的XGBoost模型
scaler = joblib.load('E:\\Project95\\Projects\\Smart Freight Decision Tool\\scaler.joblib')  # 加载训练时使用的标准化器


# 定义11个特征
features = [
    'Distance_km', 'Weight_MT', 'Fuel_Price_Index', 'Geopolitical_Risk_Score',
    'Carrier_Reliability_Score', 'Lead_Time_Days', 'Disruption_Occurred',
    'Transport_Mode', 'Product_Category', 'Weather_Condition',
    'Origin_Port', 'Destination_Port'
]

def get_user_input():
    user_input = []
    print("Please enter the following details:")

    # 获取用户输入的11个特征
    for feature in features:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))  # 获取输入并转换为浮动数
                user_input.append(value)
                break  # 如果输入正确则跳出循环
            except ValueError:
                print("Invalid input. Please enter a valid number.")  # 如果输入无效则提示并重新输入
    
    return np.array(user_input).reshape(1, -1)  # 转换为numpy数组，保持输入为1行11列的格式

def preprocess_input(input_data):
    return scaler.transform(input_data)  # 对输入数据进行标准化处理

def predict(input_data):
    processed_data = preprocess_input(input_data)  # 先处理输入数据
    prediction = model.predict(processed_data)  # 使用模型进行预测
    return prediction

# 交互主函数
def main():
    user_input = get_user_input()  # 获取用户输入的特征数据
    prediction = predict(user_input)  # 使用模型进行预测
    print(f"The predicted result is: {prediction[0]}")  # 输出预测结果

# 运行程序
if __name__ == "__main__":
    main()