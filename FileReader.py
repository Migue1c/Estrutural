import pandas as pd

# Read the data into a DataFrame with space as the delimiter
#df = pd.read_csv("Input_Data.txt", delimiter="  ", comment='#', encoding='utf-8')
#The comment parameter is set to '#' to indicate that any line starting with the '#' character should be treated as a comment and skipped.

columns_read = ['Points','z','r','t','v','w','beta','Ne','Interp','v1','w1','beta1','Material']

# usecols -> Defines what columns the code reads

df = pd.read_excel('Livro1.xlsx', sheet_name='1', usecols=columns_read) 

#df = pd.read_excel('Livro1.xlsx', sheet_name='1', usecols=['Points'])
print(df)

# Access a specific value by label (column name) / .loc for label-based access 
#value = df.loc[1, 'z']  
#print("By label:", value)

# Access a specific value by integer index / .iloc[] for integer-based access
#value2 = df.iloc[0, 0]  
#print("By integer index:", value2)