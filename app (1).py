#!/usr/bin/env python
# coding: utf-8

# In[11]:


from flask import Flask, render_template, request
import xgboost
import pickle as pkl

app = Flask(__name__)

with open('rf_pickle.pkl', 'rb') as file:
    model = pkl.load(file)

# In[12]:


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        balance = int(request.form['balance'])
        day = int(request.form['day'])
        duration = int(request.form['duration'])
        campaign = int(request.form['campaign'])
        PDays = int(request.form['PDays'])
        previous = int(request.form['previous'])
        
        df = [[age, balance, day,duration, campaign, PDays, previous]]
        output = model.predict(df)
        
        return render_template('index.html', prediction_text = 'predicted deposite {}'.format(output))
    
if __name__ == '__main__' :
    app.run(host = '0.0.0.0', port = 5000, debug = True)
        
                  
        
    


# In[ ]:




