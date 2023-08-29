import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
from scipy.stats import mannwhitneyu
import streamlit as st



def plot_pie_chart(df, column_name):
    column_values = df[column_name]
    value_counts = column_values.value_counts()

    limit = 11
    # Ограничение количества категорий с помощью параметра limit
    other_category = 'Other'
    limited_value_counts = value_counts[:limit]
    if len(value_counts) > limit:
        other_count = value_counts[limit:].sum()
        limited_value_counts[other_category] = other_count

    fig, ax = plt.subplots()
    proptease = fm.FontProperties()
    proptease.set_size('xx-small')
    patches, texts, autotexts = ax.pie(limited_value_counts, labels=None, autopct='%1.1f%%')
    labels = [f'{label} ({autotext.get_text()})' for label, autotext in zip(limited_value_counts.index, autotexts)]
    ax.axis('equal')
    ax.set_title('Распределение значений столбца')
    plt.setp(autotexts, fontproperties=proptease)
    ax.legend(patches, labels, title=column_name)
    st.pyplot(fig)



def run():
    st.title("Я сделаль")
    html_temp="""
    
    """
 
    st.markdown(html_temp)

    # Загрузка файла
    uploaded_file = st.file_uploader("Выберите файл", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(df)
            
            # Список переменных для выбора
            variable_list = df.columns.tolist()
            # Выбор переменных для исследования
            variable1 = st.selectbox("Выберите переменную 1", variable_list)
            variable2 = st.selectbox("Выберите переменную 2", variable_list)
            
            # Визуализация распределения переменных
            # Первая
            st.subheader("Распределение переменной 1")
            if df[variable1].dtype == "object":
                # Pie chart для категориальных переменных
                plot_pie_chart(df, variable1)
            
            else:
                # Гистограмма для числовых переменных
                fig, ax = plt.subplots()
                sns.histplot(df[variable1], ax=ax)
                st.pyplot(fig)
                
                # Гистограмма ящика с усами 
                fig, ax = plt.subplots()
                sns.boxplot(df[variable1], ax=ax)
                st.pyplot(fig)
                

            # Вторая 
            st.subheader("Распределение переменной 2")
            if df[variable2].dtype == "object":
                # Pie chart для категориальных переменных
                plot_pie_chart(df, variable2)
            else:
                # Гистограмма для числовых переменных
                fig, ax = plt.subplots()
                sns.histplot(df[variable2], ax=ax)
                st.pyplot(fig)
                
                # Гистограмма ящика с усами 
                fig, ax = plt.subplots()
                sns.boxplot(df[variable2], ax=ax)
                st.pyplot(fig)
            
            
            
            
            
            
            
                # Выбор и применение проверочного алгоритма
            statistical_test = st.selectbox("Выберите проверочный алгоритм", ["t-test", "u-test"])
            
            if statistical_test == "t-test":
                # Проверка гипотезы t-test
                t_stat, p_value = stats.ttest_ind(df[variable1], df[variable2])
                
                # Вывод результатов
                st.write(f"t-stat: {t_stat}")
                st.write(f"p-value: {p_value}")

                if p_value < 0.05:
                    st.write("Существует статистически значимая разница между двумя группами.")
                else:
                    st.write("Нет статистически значимой разницы между двумя группами.")

                    
            elif statistical_test == "u-test":
                stat, p_value = mannwhitneyu(df[variable1], df[variable2])
                
                st.write(f"u-stat: {stat}")
                st.write(f"p-value: {p_value}")

                if p_value < 0.05:
                    st.write("Существует статистически значимая разница между двумя группами.")
                else:
                    st.write("Нет статистически значимой разницы между двумя группами.")
                
            
            
            
        except Exception as e:
            st.write("Ну вот не получилось что-то: ", e)

   
    
if __name__=='__main__':
    run()