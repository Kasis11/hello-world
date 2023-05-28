

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\KASHISH RANA\Downloads\titanic\Medical Price Dataset.csv")
#print(np.sum(df.isna()))
df['sex']=df['sex'].map({'female':1,'male':0}) 
df['region']=df['region'].map({'southeast':0,'southwest':1,'northwest':2,'northeast':3})   
df['smoker']=df['smoker'].map({'yes':1,'no':0})
data=df[['smoker','age','sex','bmi','children','region','charges']]

data.hist(figsize=(10,10))
plt.show()


figure,axis=plt.subplots(2,4)

axis[0,0].plot(df['age'],df['smoker'])
axis[0,0].set_title("age")
axis[0,1].plot(df['sex'],df['smoker'])
axis[0,1].set_title("sex")
axis[0,2].plot(df['bmi'],df['smoker'])
axis[0,2].set_title("bmi")
axis[0,3].plot(df['region'],df['smoker'])
axis[0,3].set_title("region")
axis[1,0].plot(df['charges'],df['smoker'])
axis[1,0].set_title("charges")
axis[1,2].plot(df['children'],df['smoker'])
axis[1,2].set_title("children")
plt.show()

plt.scatter(df['age'],df['smoker'])
plt.xlabel("age")
plt.ylabel("smoker")
plt.show() 

plt.scatter(df['sex'],df['smoker'])
plt.xlabel("sex")
plt.ylabel("smoker")
plt.show() 

plt.scatter(df['region'],df['smoker'])
plt.xlabel("region")
plt.ylabel("smoker")
plt.show() 

plt.scatter(df['charges'],df['smoker'])
plt.xlabel("charges")
plt.ylabel("smoker")
plt.show()

plt.scatter(df['children'],df['smoker'])
plt.xlabel("children")
plt.ylabel("smoker")
plt.show() 

plt.scatter(df['bmi'],df['smoker'])
plt.xlabel("bmi")
plt.ylabel("smoker")
plt.show() 

train=df[:(int((len(data)*0.8)))]  #dividing train and test data
test=df[(int((len(data)*0.8))):]
total=data.shape

print("train",train.shape)
print("test",test.shape)
print("total",data.shape)

def slr(i, o):
    x = i  
    y = o  

    # Calculate the product of the sum of 'y' and the sum of 'x', divided by the length of 'x'
    yx_byn = (y.sum() * x.sum()) / len(x)

    # Calculate the sum of the squares of each element in 'x'
    s_xx = (x * x).sum()

    # Calculate the square of the sum of 'x', divided by the length of 'x'
    xx_byn = ((x.sum()) ** 2) / len(x)

    # Calculate the sum of the element-wise multiplication between 'y' and 'x'
    s_yx = (y * x).sum()

    # Calculate the slope of the regression line
    slope = (s_yx - yx_byn) / (s_xx - xx_byn)

    # Calculate the intercept of the regression line
    intercept = (y.mean()) - slope * (x.mean()) 
    return intercept,slope

c_a, c_s = slr(train['age'], train['smoker']) 
#- Assuming the function 'slr' returns the intercept and slope values of the regression line, this line calculates the intercept and slope using the 'age' column as the independent variable and the 'smoker' column as the dependent variable. The calculated values are assigned to the variables 't_i' and 't_s'.

# Scatter plot of 'age' vs 'smoker'
plt.scatter(train['age'], train['smoker'])
# Plot the regression line using the calculated slope and intercept
plt.plot(train['age'], c_s * train['age'] + c_a, '-r')

# Set the y-axis label
plt.ylabel("smoker") 
# Set the x-axis label
plt.xlabel("age")


# Display the plot
plt.show()


def test(t,intercept,slope):
    return (t*slope)+intercept
t_y=[]   
x=data['age'].values
for i in range(train.shape[0],total[0]):
    test_y=test(x[i],c_a,c_s)
    t_y.append(test_y)
print(t_y)