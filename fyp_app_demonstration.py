import streamlit as st
import pandas as pd
from sklearn import datasets
from prophet import Prophet
import matplotlib as plt
import datetime as dt
import itertools

#prophet diagnostic lib
from prophet.diagnostics import performance_metrics, cross_validation
from prophet.plot import plot_cross_validation_metric

#login authenticator
import  streamlit_authenticator as stauth

st.set_page_config(
     page_title="FYP ARIMA demonstration application",
     page_icon="🧊",
     layout="centered",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.fyp_demo.com/help',
         'Report a bug': "https://www.fyp_demo.com/bug",
         'About': "#This is an application to show ARIMA forecasting"
     }
 )


def user_input_features():
    month_frequency = st.sidebar.slider('Months forecast', 1,2,12)
    
    return month_frequency


global df
def main():

    menu = ["Main Page","Input page"]
    choice = st.sidebar.selectbox("Navigation menu",menu)

    if choice == "Main Page":

        st.title('Expenses forecaster application')

        names = ['User_Admin']
        username = ['Luqman']
        passwords = ['Admin123']

        hashed_password = stauth.hasher(passwords).generate()

        authenticator = stauth.authenticate(names,username,hashed_password,
                        'password_cookie','password_signature', cookie_expiry_days=14)
        
        name, authentication_status = authenticator.login('Login','main')

        if authentication_status:
            
            st.write('Welcome *%s*' % (username))

            st.write("""
            
            ### This app predicts future expenses based on ARIMA concept
            """)

            st.sidebar.title('User Input Parameters')

            st.sidebar.write("Select how many months forward you want to forecast:")

            st.write("This is the page for outputting data")
            df = pd.read_csv('https://raw.githubusercontent.com/SuperHacka/fyp-web-demo/main/user_data_4_prophet.csv')

            df.head()

            df.info()

            #streamlit display dataset
            st.dataframe(df)

            #grid search parameterization tuning

                

            #prophet df 
            m = Prophet()
            m.fit(df)

            #plotting the graph for forecast
            st.subheader('Monthly graph for user 4 data')
            m_2 = Prophet(seasonality_mode = 'multiplicative').fit(df)
            future_2 = m_2.make_future_dataframe(periods=user_input_features(), freq='M')
            forecast_2 = m_2.predict(future_2)
            fig2 = m_2.plot(forecast_2)
            plt.pyplot.axvline(dt.datetime(2021, 1, 1),color='red')
            st.pyplot(fig2)

            #performance metrics evaluation
            df_cv = cross_validation(m, initial='183 days', period='15 days', horizon = '30 days')

            df_p = performance_metrics(df_cv)
            df_p.head()

            st.subheader('MAPE analysis for user 4 data')
            st.write('MAPE(Mean Absolute Percentage Error) is a measure in predicting accuracy of forecasting' )
            fig_metric = plot_cross_validation_metric(df_cv, metric='mape')
            st.pyplot(fig_metric)

            #hyperparameter tuning module
            param_grid = {  
                'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            }

            # Generate all combinations of parameters
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            rmses = []  # Store the RMSEs for each params here

            # Use cross validation to evaluate all parameters
            cutoffs = pd.to_datetime(['2020-01-01'])
            df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days')

            for params in all_params:
                m = Prophet(**params).fit(df)  # Fit model with given params
                df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].values[0])

            # Find the best parameters
            tuning_results = pd.DataFrame(all_params)
            tuning_results['rmse'] = rmses
            print(tuning_results)
            st.write('Shown below is the tuning results for the hyperparameter tuning')
            st.write(tuning_results)
            
            
        elif authentication_status == False:
            st.error('The username or password entered is wrong')
        elif authentication_status == None:
            st.warning('Please enter the username or password')


    elif choice == "Input page":
        st.title('Expenses forecaster application')
        st.write("""
        
        ### This app predicts future expenses based on ARIMA concept
        """)
        st.write("This is the page for inputting data")

        #file upload
          
        file = st.file_uploader(label='Please upload a CSV file', type='csv', accept_multiple_files=False, key=None, help=None, on_change=None)

     
        if file is not None:
            print(file)

            try: 
                df = pd.read_csv(file)
            except Exception as e:
                print(e)
                st.error("Please upload a CSV file")
                df = pd.read_csv(file)
        
        try:
            st.write(df)
        except Exception as e:
            print(e)
            st.warning("Please upload file to the application")

    

if __name__ == '__main__':
	main()
        

